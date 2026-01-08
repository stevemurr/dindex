//! HNSW index implementation using USearch

use crate::config::IndexConfig;
use crate::types::{ChunkId, Embedding};
use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// Vector index for storing and querying embeddings
pub struct VectorIndex {
    /// USearch index
    index: Index,
    /// Mapping from internal key to chunk ID
    key_to_chunk: RwLock<HashMap<u64, ChunkId>>,
    /// Mapping from chunk ID to internal key
    chunk_to_key: RwLock<HashMap<ChunkId, u64>>,
    /// Next available key
    next_key: AtomicU64,
    /// Index configuration (reserved for future tuning)
    #[allow(dead_code)]
    config: IndexConfig,
    /// Number of dimensions
    dimensions: usize,
}

/// Search result from the vector index
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub chunk_id: ChunkId,
    pub key: u64,
    pub distance: f32,
    pub similarity: f32,
}

impl VectorIndex {
    /// Create a new vector index
    pub fn new(dimensions: usize, config: &IndexConfig) -> Result<Self> {
        info!(
            "Creating vector index: {} dimensions, M={}, ef_construction={}",
            dimensions, config.hnsw_m, config.hnsw_ef_construction
        );

        let options = IndexOptions {
            dimensions,
            metric: MetricKind::Cos, // Cosine similarity
            quantization: ScalarKind::F32, // Will use INT8 for storage
            connectivity: config.hnsw_m,
            expansion_add: config.hnsw_ef_construction,
            expansion_search: config.hnsw_ef_search,
            multi: false,
        };

        let index = Index::new(&options).context("Failed to create USearch index")?;
        index
            .reserve(config.max_capacity)
            .context("Failed to reserve index capacity")?;

        Ok(Self {
            index,
            key_to_chunk: RwLock::new(HashMap::new()),
            chunk_to_key: RwLock::new(HashMap::new()),
            next_key: AtomicU64::new(0),
            config: config.clone(),
            dimensions,
        })
    }

    /// Load index from disk
    pub fn load(path: impl AsRef<Path>, config: &IndexConfig) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading vector index from: {}", path.display());

        let options = IndexOptions {
            dimensions: 0, // Will be read from file
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: config.hnsw_m,
            expansion_add: config.hnsw_ef_construction,
            expansion_search: config.hnsw_ef_search,
            multi: false,
        };

        let index = Index::new(&options)?;
        let path_str = path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
        index.load(path_str).context("Failed to load index")?;

        let dimensions = index.dimensions();

        // Load mappings
        let mappings_path = path.with_extension("mappings.json");
        let (key_to_chunk, chunk_to_key, next_key) = if mappings_path.exists() {
            let data = std::fs::read_to_string(&mappings_path)?;
            let mappings: SavedMappings = serde_json::from_str(&data)?;
            let chunk_to_key: HashMap<ChunkId, u64> = mappings
                .key_to_chunk
                .iter()
                .map(|(k, v)| (v.clone(), *k))
                .collect();
            (
                mappings.key_to_chunk,
                chunk_to_key,
                mappings.next_key,
            )
        } else {
            (HashMap::new(), HashMap::new(), 0)
        };

        Ok(Self {
            index,
            key_to_chunk: RwLock::new(key_to_chunk),
            chunk_to_key: RwLock::new(chunk_to_key),
            next_key: AtomicU64::new(next_key),
            config: config.clone(),
            dimensions,
        })
    }

    /// Save index to disk
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        info!("Saving vector index to: {}", path.display());

        let path_str = path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid path"))?;
        self.index.save(path_str).context("Failed to save index")?;

        // Save mappings
        let mappings = SavedMappings {
            key_to_chunk: self.key_to_chunk.read().clone(),
            next_key: self.next_key.load(Ordering::SeqCst),
        };
        let mappings_path = path.with_extension("mappings.json");
        let data = serde_json::to_string_pretty(&mappings)?;
        std::fs::write(&mappings_path, data)?;

        Ok(())
    }

    /// Add a single embedding to the index
    pub fn add(&self, chunk_id: &ChunkId, embedding: &Embedding) -> Result<u64> {
        assert_eq!(
            embedding.len(),
            self.dimensions,
            "Embedding dimension mismatch"
        );

        let key = self.next_key.fetch_add(1, Ordering::SeqCst);

        self.index
            .add(key, embedding)
            .context("Failed to add to index")?;

        self.key_to_chunk.write().insert(key, chunk_id.clone());
        self.chunk_to_key.write().insert(chunk_id.clone(), key);

        debug!("Added chunk {} with key {}", chunk_id, key);
        Ok(key)
    }

    /// Add multiple embeddings in batch
    pub fn add_batch(&self, items: &[(ChunkId, Embedding)]) -> Result<Vec<u64>> {
        let mut keys = Vec::with_capacity(items.len());

        for (chunk_id, embedding) in items {
            let key = self.add(chunk_id, embedding)?;
            keys.push(key);
        }

        Ok(keys)
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &Embedding, k: usize) -> Result<Vec<VectorSearchResult>> {
        assert_eq!(
            query.len(),
            self.dimensions,
            "Query dimension mismatch"
        );

        let results = self.index.search(query, k).context("Search failed")?;

        let key_to_chunk = self.key_to_chunk.read();
        let search_results: Vec<VectorSearchResult> = results
            .keys
            .iter()
            .zip(results.distances.iter())
            .filter_map(|(&key, &distance)| {
                key_to_chunk.get(&key).map(|chunk_id| VectorSearchResult {
                    chunk_id: chunk_id.clone(),
                    key,
                    distance,
                    // Convert distance to similarity (for cosine, similarity = 1 - distance)
                    similarity: 1.0 - distance,
                })
            })
            .collect();

        Ok(search_results)
    }

    /// Remove an embedding by chunk ID
    pub fn remove(&self, chunk_id: &ChunkId) -> Result<bool> {
        let key = match self.chunk_to_key.write().remove(chunk_id) {
            Some(key) => key,
            None => return Ok(false),
        };

        self.key_to_chunk.write().remove(&key);
        self.index.remove(key).context("Failed to remove from index")?;

        debug!("Removed chunk {} with key {}", chunk_id, key);
        Ok(true)
    }

    /// Check if a chunk exists in the index
    pub fn contains(&self, chunk_id: &ChunkId) -> bool {
        self.chunk_to_key.read().contains_key(chunk_id)
    }

    /// Get the number of items in the index
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get the key for a chunk ID
    pub fn get_key(&self, chunk_id: &ChunkId) -> Option<u64> {
        self.chunk_to_key.read().get(chunk_id).copied()
    }

    /// Get the chunk ID for a key
    pub fn get_chunk_id(&self, key: u64) -> Option<ChunkId> {
        self.key_to_chunk.read().get(&key).cloned()
    }
}

/// Serializable mappings for persistence
#[derive(serde::Serialize, serde::Deserialize)]
struct SavedMappings {
    key_to_chunk: HashMap<u64, ChunkId>,
    next_key: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> VectorIndex {
        let config = IndexConfig {
            hnsw_m: 8,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 50,
            memory_mapped: false,
            max_capacity: 1000,
        };
        VectorIndex::new(4, &config).unwrap()
    }

    #[test]
    fn test_add_and_search() {
        let index = create_test_index();

        // Add some vectors
        index
            .add(&"chunk1".to_string(), &vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        index
            .add(&"chunk2".to_string(), &vec![0.0, 1.0, 0.0, 0.0])
            .unwrap();
        index
            .add(&"chunk3".to_string(), &vec![0.9, 0.1, 0.0, 0.0])
            .unwrap();

        // Search for nearest to [1, 0, 0, 0]
        let results = index.search(&vec![1.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_id, "chunk1");
        assert!(results[0].similarity > 0.99); // Should be almost 1
    }

    #[test]
    fn test_remove() {
        let index = create_test_index();

        index
            .add(&"chunk1".to_string(), &vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        assert!(index.contains(&"chunk1".to_string()));

        index.remove(&"chunk1".to_string()).unwrap();
        assert!(!index.contains(&"chunk1".to_string()));
    }
}
