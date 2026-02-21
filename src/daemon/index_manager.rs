//! Index Manager
//!
//! Centralized index management for the daemon. Owns all index components
//! and provides thread-safe access for concurrent read/write operations.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use parking_lot::RwLock;
use tracing::{debug, info};

use crate::config::Config;
use crate::embedding::{hash_based_embedding, EmbeddingEngine};
use crate::index::{ChunkStorage, VectorIndex};
use crate::retrieval::{Bm25Index, HybridIndexer, HybridRetriever};
use crate::types::{Chunk, Query, QueryFilters, SearchResult};

use super::protocol::IndexStats;

/// Centralized index management for the daemon
pub struct IndexManager {
    config: Config,
    vector_index: Arc<VectorIndex>,
    /// BM25 index - held for ownership. Accessed indirectly via `retriever` and `indexer`
    /// which hold their own Arc references. Stored here to ensure the index outlives
    /// any potential future direct access needs.
    #[allow(dead_code)]
    bm25_index: Arc<Bm25Index>,
    chunk_storage: Arc<ChunkStorage>,
    retriever: Arc<HybridRetriever>,
    indexer: Arc<HybridIndexer>,
    /// Embedding engine for generating query embeddings
    embedding_engine: RwLock<Option<Arc<EmbeddingEngine>>>,
    /// Tracks pending chunks that haven't been committed yet
    pending_count: RwLock<usize>,
}

impl IndexManager {
    /// Load or create indexes from the configured data directory
    pub fn load(config: &Config) -> Result<Self> {
        let data_dir = &config.node.data_dir;
        std::fs::create_dir_all(data_dir)?;

        info!("Loading indexes from: {}", data_dir.display());

        // Load or create vector index
        let index_path = data_dir.join("vector.index");
        let vector_index = if index_path.exists() {
            info!("Loading existing vector index");
            Arc::new(VectorIndex::load(&index_path, &config.index)?)
        } else {
            info!("Creating new vector index");
            Arc::new(VectorIndex::new(config.embedding.dimensions, &config.index)?)
        };

        // Load or create BM25 index
        let bm25_path = data_dir.join("bm25");
        let bm25_index = Arc::new(Bm25Index::new(&bm25_path)?);
        info!("BM25 index loaded");

        // Load or create chunk storage (auto-migrates from JSON if needed)
        let chunk_storage = Arc::new(ChunkStorage::load(data_dir)?);
        info!("Chunk storage loaded with {} chunks", chunk_storage.len());

        // Create retriever for search operations
        let retriever = Arc::new(HybridRetriever::new(
            vector_index.clone(),
            bm25_index.clone(),
            chunk_storage.clone(),
            None,
            config.retrieval.clone(),
        ));

        // Create indexer for write operations
        let indexer = Arc::new(HybridIndexer::new(
            vector_index.clone(),
            bm25_index.clone(),
            chunk_storage.clone(),
        ));

        Ok(Self {
            config: config.clone(),
            vector_index,
            bm25_index,
            chunk_storage,
            retriever,
            indexer,
            embedding_engine: RwLock::new(None),
            pending_count: RwLock::new(0),
        })
    }

    /// Search the index with a query
    pub fn search(&self, query_text: &str, top_k: usize) -> Result<(Vec<SearchResult>, u64)> {
        self.search_with_filters(query_text, top_k, None)
    }

    /// Search the index with a query and optional filters
    pub fn search_with_filters(
        &self,
        query_text: &str,
        top_k: usize,
        filters: Option<&QueryFilters>,
    ) -> Result<(Vec<SearchResult>, u64)> {
        let start = Instant::now();
        debug!("Searching for: {} (top_k={}, filters={:?})", query_text, top_k, filters.is_some());

        // Generate query embedding (uses real engine if available, hash-based fallback otherwise)
        let query_embedding = self.generate_embedding(query_text);

        // Create query with filters
        let mut query = Query::new(query_text, top_k);
        query.filters = filters.cloned();

        // Execute search - request more results if filtering
        let fetch_k = if filters.is_some() { top_k * 3 } else { top_k };
        let search_query = Query {
            top_k: fetch_k,
            ..query.clone()
        };

        let mut results = self.retriever.search(&search_query, Some(&query_embedding))?;

        // Apply metadata filtering if specified
        if let Some(filters) = filters {
            results = Self::filter_by_metadata(results, filters);
            results.truncate(top_k);
        }

        let query_time_ms = start.elapsed().as_millis() as u64;
        debug!(
            "Search completed in {}ms, found {} results",
            query_time_ms,
            results.len()
        );

        Ok((results, query_time_ms))
    }

    /// Filter search results by metadata constraints
    fn filter_by_metadata(results: Vec<SearchResult>, filters: &QueryFilters) -> Vec<SearchResult> {
        results
            .into_iter()
            .filter(|result| {
                let extra = &result.chunk.metadata.extra;

                // Check source_url_prefix
                if let Some(ref prefix) = filters.source_url_prefix {
                    if let Some(ref url) = result.chunk.metadata.source_url {
                        if !url.starts_with(prefix) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }

                // Check metadata_equals (all must match)
                if let Some(ref equals) = filters.metadata_equals {
                    for (key, value) in equals {
                        if extra.get(key) != Some(value) {
                            return false;
                        }
                    }
                }

                // Check metadata_contains (value must be in allowed list)
                if let Some(ref contains) = filters.metadata_contains {
                    for (key, allowed_values) in contains {
                        if let Some(actual) = extra.get(key) {
                            // For comma-separated category values, check if any allowed value is present
                            let actual_values: std::collections::HashSet<&str> =
                                actual.split(',').map(|s| s.trim()).collect();
                            let has_match = allowed_values.iter().any(|v| actual_values.contains(v.as_str()));
                            if !has_match {
                                return false;
                            }
                        } else {
                            return false;
                        }
                    }
                }

                true
            })
            .collect()
    }

    /// Index a batch of chunks with their embeddings
    pub fn index_batch(&self, chunks_with_embeddings: &[(Chunk, Vec<f32>)]) -> Result<Vec<u64>> {
        debug!("Indexing batch of {} chunks", chunks_with_embeddings.len());

        let keys = self.indexer.index_batch(chunks_with_embeddings)?;

        // Update pending count
        {
            let mut pending = self.pending_count.write();
            *pending += chunks_with_embeddings.len();
        }

        Ok(keys)
    }

    /// Index chunks without embeddings (will generate them)
    pub fn index_chunks(&self, chunks: Vec<Chunk>) -> Result<Vec<u64>> {
        // Generate embeddings for each chunk (uses real engine if available, hash-based fallback otherwise)
        let chunks_with_embeddings: Vec<_> = chunks
            .into_iter()
            .map(|c| {
                let embedding = self.generate_embedding(&c.content);
                (c, embedding)
            })
            .collect();

        self.index_batch(&chunks_with_embeddings)
    }

    /// Commit pending changes to disk
    pub fn commit(&self) -> Result<()> {
        info!("Committing index changes");

        let index_path = self.config.node.data_dir.join("vector.index");
        self.vector_index.save(&index_path)?;
        self.indexer.save()?;

        // Reset pending count
        {
            let mut pending = self.pending_count.write();
            *pending = 0;
        }

        info!("Index committed successfully");
        Ok(())
    }

    /// Get index statistics
    pub fn stats(&self) -> Result<IndexStats> {
        let data_dir = &self.config.node.data_dir;

        // Calculate sizes
        let vector_index_size = self.file_size(data_dir.join("vector.index"));
        let bm25_index_size = self.dir_size(data_dir.join("bm25"));
        let storage_size = self.file_size(data_dir.join("chunks.json"));

        // Count chunks and unique documents
        let total_chunks = self.chunk_storage.len();
        let total_documents = self.chunk_storage.document_count();

        Ok(IndexStats {
            total_documents,
            total_chunks,
            vector_index_size_bytes: vector_index_size,
            bm25_index_size_bytes: bm25_index_size,
            storage_size_bytes: storage_size,
        })
    }

    /// Get the number of pending (uncommitted) chunks
    pub fn pending_count(&self) -> usize {
        *self.pending_count.read()
    }

    /// Get the data directory
    pub fn data_dir(&self) -> &PathBuf {
        &self.config.node.data_dir
    }

    /// Get the hybrid retriever for external use (e.g., QueryExecutor)
    pub fn retriever(&self) -> Arc<HybridRetriever> {
        self.retriever.clone()
    }

    /// Get all embeddings for advertisement generation
    pub fn all_embeddings(&self) -> Vec<Vec<f32>> {
        self.chunk_storage
            .all_embeddings()
            .into_iter()
            .map(|(_, emb)| emb)
            .collect()
    }

    /// Get the configured embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.config.embedding.dimensions
    }

    /// Set the embedding engine for generating query embeddings
    pub fn set_embedding_engine(&self, engine: Arc<EmbeddingEngine>) {
        let mut guard = self.embedding_engine.write();
        *guard = Some(engine);
    }

    /// Generate embedding for content using real embedding engine
    fn generate_embedding(&self, content: &str) -> Vec<f32> {
        if let Some(ref engine) = *self.embedding_engine.read() {
            match engine.embed(content) {
                Ok(embedding) => return embedding,
                Err(e) => {
                    tracing::warn!("Embedding generation failed, using fallback: {}", e);
                    return hash_based_embedding(content, engine.dimensions());
                }
            }
        }

        // Fallback: generate deterministic fake embedding if no engine available
        tracing::warn!("No embedding engine available for search, using hash-based fallback");
        hash_based_embedding(content, self.config.embedding.dimensions)
    }

    /// Get file size, returning 0 if file doesn't exist
    fn file_size(&self, path: PathBuf) -> u64 {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    }

    /// Get total size of files in a directory
    fn dir_size(&self, path: PathBuf) -> u64 {
        if !path.exists() {
            return 0;
        }

        walkdir::WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(data_dir: &std::path::Path) -> Config {
        let mut config = Config::default();
        config.node.data_dir = data_dir.to_path_buf();
        config
    }

    #[test]
    fn test_index_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(temp_dir.path());

        let manager = IndexManager::load(&config).unwrap();
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_index_manager_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(temp_dir.path());

        let manager = IndexManager::load(&config).unwrap();
        let stats = manager.stats().unwrap();

        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_documents, 0);
    }
}
