//! HNSW index implementation using USearch
//!
//! Uses sled for mapping persistence instead of JSON for better scalability.

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
    /// Mapping from internal key to chunk ID (in-memory for fast lookup)
    key_to_chunk: RwLock<HashMap<u64, ChunkId>>,
    /// Mapping from chunk ID to internal key (in-memory for fast lookup)
    chunk_to_key: RwLock<HashMap<ChunkId, u64>>,
    /// Next available key
    next_key: AtomicU64,
    /// Sled database for persistent mappings
    mappings_db: Option<sled::Db>,
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
            mappings_db: None, // Will be set on first save/load
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

        // Reserve additional capacity for new vectors
        // The loaded index only has capacity for existing vectors
        let current_size = index.size();
        let target_capacity = config.max_capacity.max(current_size + 100_000);
        index
            .reserve(target_capacity)
            .context("Failed to reserve additional capacity after loading")?;

        // Open sled database for mappings
        let mappings_db_path = path.with_extension("mappings.sled");
        let json_mappings_path = path.with_extension("mappings.json");

        // Check if we need to migrate from JSON
        let (key_to_chunk, chunk_to_key, next_key, mappings_db) =
            if json_mappings_path.exists() && !mappings_db_path.exists() {
                // Migrate from JSON to sled
                info!("Migrating vector index mappings from JSON to sled...");
                let data = std::fs::read_to_string(&json_mappings_path)?;
                let old_mappings: SavedMappings = serde_json::from_str(&data)?;

                // Open new sled database
                let db = sled::open(&mappings_db_path)
                    .context("Failed to open mappings database")?;

                // Write all mappings to sled
                for (key, chunk_id) in &old_mappings.key_to_chunk {
                    db.insert(&key.to_le_bytes(), chunk_id.as_bytes())?;
                }
                // Store next_key with special prefix
                db.insert(b"__next_key__", &old_mappings.next_key.to_le_bytes())?;
                db.flush()?;

                // Build in-memory maps
                let chunk_to_key: HashMap<ChunkId, u64> = old_mappings
                    .key_to_chunk
                    .iter()
                    .map(|(k, v)| (v.clone(), *k))
                    .collect();

                // Backup old JSON file
                let backup_path = path.with_extension("mappings.json.backup");
                std::fs::rename(&json_mappings_path, &backup_path)?;
                info!("Migration complete. Old JSON backed up to {:?}", backup_path);

                (old_mappings.key_to_chunk, chunk_to_key, old_mappings.next_key, Some(db))
            } else if mappings_db_path.exists() {
                // Load from sled
                let db = sled::open(&mappings_db_path)
                    .context("Failed to open mappings database")?;

                let mut key_to_chunk = HashMap::new();
                let mut chunk_to_key = HashMap::new();

                // Load next_key
                let next_key = db
                    .get(b"__next_key__")?
                    .map(|v| {
                        let bytes: [u8; 8] = v.as_ref().try_into()
                            .context("Corrupt next_key entry in mappings database")?;
                        Ok::<u64, anyhow::Error>(u64::from_le_bytes(bytes))
                    })
                    .transpose()?
                    .unwrap_or(0);

                // Load all mappings
                for result in db.iter() {
                    let (k, v) = result?;
                    // Skip special keys
                    if k.starts_with(b"__") {
                        continue;
                    }
                    let key = u64::from_le_bytes(
                        k.as_ref().try_into()
                            .with_context(|| format!("Corrupt key in mappings database: {} bytes", k.len()))?
                    );
                    let chunk_id = String::from_utf8(v.to_vec())
                        .context("Invalid chunk ID in mappings")?;
                    chunk_to_key.insert(chunk_id.clone(), key);
                    key_to_chunk.insert(key, chunk_id);
                }

                info!("Loaded {} mappings from sled", key_to_chunk.len());
                (key_to_chunk, chunk_to_key, next_key, Some(db))
            } else {
                // No existing mappings
                (HashMap::new(), HashMap::new(), 0, None)
            };

        Ok(Self {
            index,
            key_to_chunk: RwLock::new(key_to_chunk),
            chunk_to_key: RwLock::new(chunk_to_key),
            next_key: AtomicU64::new(next_key),
            mappings_db,
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

        // Save mappings to sled
        let mappings_db_path = path.with_extension("mappings.sled");
        let db = if let Some(ref db) = self.mappings_db {
            db.clone()
        } else {
            sled::open(&mappings_db_path).context("Failed to open mappings database")?
        };

        // Write all mappings (in case of new index or changes)
        let key_to_chunk = self.key_to_chunk.read();
        for (key, chunk_id) in key_to_chunk.iter() {
            db.insert(&key.to_le_bytes(), chunk_id.as_bytes())?;
        }

        // Save next_key
        let next_key = self.next_key.load(Ordering::SeqCst);
        db.insert(b"__next_key__", &next_key.to_le_bytes())?;

        db.flush().context("Failed to flush mappings database")?;

        info!("Saved {} mappings to sled", key_to_chunk.len());
        Ok(())
    }

    /// Add a single embedding to the index
    pub fn add(&self, chunk_id: &ChunkId, embedding: &Embedding) -> Result<u64> {
        anyhow::ensure!(
            embedding.len() == self.dimensions,
            "Embedding dimension mismatch: expected {}, got {}",
            self.dimensions,
            embedding.len()
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
        anyhow::ensure!(
            query.len() == self.dimensions,
            "Query dimension mismatch: expected {}, got {}",
            self.dimensions,
            query.len()
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

/// Serializable mappings for JSON format (used only for migration from old format)
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

    #[test]
    fn test_persistence() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test.index");

        let config = IndexConfig {
            hnsw_m: 8,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 50,
            memory_mapped: false,
            max_capacity: 1000,
        };

        // Create and populate index
        {
            let index = VectorIndex::new(4, &config).unwrap();
            index
                .add(&"chunk1".to_string(), &vec![1.0, 0.0, 0.0, 0.0])
                .unwrap();
            index
                .add(&"chunk2".to_string(), &vec![0.0, 1.0, 0.0, 0.0])
                .unwrap();

            index.save(&index_path).unwrap();
            assert_eq!(index.len(), 2);
        }

        // Reload and verify
        {
            let index = VectorIndex::load(&index_path, &config).unwrap();
            assert_eq!(index.len(), 2);
            assert!(index.contains(&"chunk1".to_string()));
            assert!(index.contains(&"chunk2".to_string()));

            // Verify mappings work
            let key1 = index
                .get_key(&"chunk1".to_string())
                .expect("chunk1 key should exist");
            let chunk_id = index.get_chunk_id(key1);
            assert_eq!(chunk_id, Some("chunk1".to_string()));
        }

        // Verify sled file exists (not JSON)
        assert!(temp_dir.path().join("test.mappings.sled").exists());
        assert!(!temp_dir.path().join("test.mappings.json").exists());
    }
}
