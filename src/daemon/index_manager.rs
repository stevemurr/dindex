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
use crate::embedding::{generate_embedding, EmbeddingEngine};
use crate::index::{ChunkStorage, IndexStack, VectorIndex};
use crate::retrieval::{Bm25Index, HybridIndexer, HybridRetriever};
use crate::routing::CentroidGenerator;
use crate::types::{Chunk, Query, QueryFilters, SearchResult};
use crate::util::{normalize_in_place, truncate_str};

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
        info!("Loading indexes from: {}", config.node.data_dir.display());

        let stack = IndexStack::open(config)?;

        let retriever = Arc::new(stack.retriever(None, config.retrieval.clone()));
        let indexer = Arc::new(stack.indexer());

        Ok(Self {
            config: config.clone(),
            vector_index: stack.vector_index,
            bm25_index: stack.bm25_index,
            chunk_storage: stack.chunk_storage,
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

        // Generate query embedding
        let query_embedding = self.generate_embedding(query_text)?;

        // Create query with filters — HybridRetriever handles filtering internally
        let mut query = Query::new(query_text, top_k);
        query.filters = filters.cloned();

        let results = self.retriever.search(&query, Some(&query_embedding))?;

        let query_time_ms = start.elapsed().as_millis() as u64;
        debug!(
            "Search completed in {}ms, found {} results",
            query_time_ms,
            results.len()
        );

        Ok((results, query_time_ms))
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
        // Generate embeddings for each chunk, skipping chunks that fail
        let mut chunks_with_embeddings = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            match self.generate_embedding(&chunk.content) {
                Ok(embedding) => chunks_with_embeddings.push((chunk, embedding)),
                Err(e) => {
                    tracing::warn!(
                        "Skipping chunk {}: embedding generation failed: {}",
                        chunk.metadata.chunk_id,
                        e
                    );
                }
            }
        }

        if chunks_with_embeddings.is_empty() {
            anyhow::bail!(
                "All chunks failed embedding generation. \
                 Ensure an embedding backend is configured in dindex.toml."
            );
        }

        self.index_batch(&chunks_with_embeddings)
    }

    /// Delete all chunks belonging to the given document IDs
    pub fn delete_documents(&self, document_ids: &[String]) -> Result<(usize, usize)> {
        let mut total_chunks = 0usize;
        let mut docs_deleted = 0usize;

        for doc_id in document_ids {
            match self.indexer.remove_document(doc_id) {
                Ok(count) => {
                    total_chunks += count;
                    if count > 0 {
                        docs_deleted += 1;
                    }
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Failed to delete document {}: {}",
                        doc_id,
                        e
                    ));
                }
            }
        }

        // Save vector index to disk
        let index_path = self.config.node.data_dir.join("vector.index");
        self.vector_index.save(&index_path)?;
        self.indexer.save()?;

        info!(
            "Deleted {} documents ({} chunks)",
            docs_deleted, total_chunks
        );
        Ok((docs_deleted, total_chunks))
    }

    /// Clear all entries from the index
    pub fn clear_all(&self) -> Result<usize> {
        let chunk_ids = self.chunk_storage.chunk_ids();
        let total = chunk_ids.len();

        for chunk_id in &chunk_ids {
            if let Err(e) = self.vector_index.remove(chunk_id) {
                tracing::warn!("Failed to remove chunk {} from vector index: {}", chunk_id, e);
            }
            self.chunk_storage.remove(chunk_id);
        }

        // Clear BM25 and commit
        self.bm25_index.clear()?;

        // Save vector index
        let index_path = self.config.node.data_dir.join("vector.index");
        self.vector_index.save(&index_path)?;
        self.chunk_storage.save()?;

        info!("Cleared all {} chunks from index", total);
        Ok(total)
    }

    /// Replace an existing document indexed under the given URL.
    /// Returns the old document ID if one was found and removed.
    pub fn replace_by_url(&self, url: &str) -> Result<Option<String>> {
        if let Some(old_doc_id) = self.chunk_storage.find_document_id_by_url(url) {
            let count = self.indexer.remove_document(&old_doc_id)?;
            if count > 0 {
                debug!("Replacing existing document {} ({} chunks) for URL {}", old_doc_id, count, url);
            }
            Ok(Some(old_doc_id))
        } else {
            Ok(None)
        }
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
        let storage_size = self.dir_size(data_dir.join("chunks.sled"));

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

    /// Cluster documents by their stored embeddings
    pub fn cluster_documents(
        &self,
        document_urls: &[String],
        max_clusters: usize,
    ) -> Result<ClusterDocumentsResult> {
        let mut matched_docs = Vec::new();
        let mut unmatched_urls = Vec::new();
        let mut doc_embeddings = Vec::new();

        for url in document_urls {
            let doc_id = match self.chunk_storage.find_document_id_by_url(url) {
                Some(id) => id,
                None => {
                    unmatched_urls.push(url.clone());
                    continue;
                }
            };

            let stored_chunks = self.chunk_storage.get_by_document(&doc_id);
            if stored_chunks.is_empty() {
                unmatched_urls.push(url.clone());
                continue;
            }

            // Average chunk embeddings into a document-level embedding
            let dim = stored_chunks[0].embedding.len();
            let mut avg = vec![0.0f32; dim];
            for sc in &stored_chunks {
                for (j, &val) in sc.embedding.iter().enumerate() {
                    if j < dim {
                        avg[j] += val;
                    }
                }
            }
            let n = stored_chunks.len() as f32;
            for val in avg.iter_mut() {
                *val /= n;
            }
            normalize_in_place(&mut avg);

            // Extract title from first chunk's metadata
            let title = stored_chunks[0].chunk.metadata.source_title.clone();
            let snippet = truncate_str(&stored_chunks[0].chunk.content, 200).into_owned();

            doc_embeddings.push(avg);
            matched_docs.push(MatchedDocument {
                url: url.clone(),
                doc_id,
                title,
                snippet,
            });
        }

        // Run k-means clustering on document embeddings
        let cluster_result = if matched_docs.len() <= 1 {
            // Single or no documents — trivial assignment
            crate::routing::ClusterResult {
                centroids: Vec::new(),
                assignments: vec![0; matched_docs.len()],
            }
        } else {
            let generator = CentroidGenerator::new(max_clusters);
            generator.generate_with_assignments(&doc_embeddings)
        };

        Ok(ClusterDocumentsResult {
            matched_docs,
            assignments: cluster_result.assignments,
            unmatched_urls,
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
            .embedding_iter()
            .map(|(_, emb)| emb)
            .collect()
    }

    /// Get the configured embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.config.embedding.dimensions
    }

    /// Get the embedding engine (if initialized)
    pub fn embedding_engine(&self) -> Option<Arc<EmbeddingEngine>> {
        self.embedding_engine.read().clone()
    }

    /// Set the embedding engine for generating query embeddings
    pub fn set_embedding_engine(&self, engine: Arc<EmbeddingEngine>) {
        let mut guard = self.embedding_engine.write();
        *guard = Some(engine);
    }

    /// Generate embedding for content using the real embedding engine.
    ///
    /// Returns an error if no engine is available or embedding generation fails.
    fn generate_embedding(&self, content: &str) -> Result<Vec<f32>> {
        let guard = self.embedding_engine.read();
        let engine_ref = guard.as_ref().map(|arc| arc.as_ref());
        generate_embedding(engine_ref, content)
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

/// A document matched by URL in cluster_documents()
pub struct MatchedDocument {
    pub url: String,
    pub doc_id: String,
    pub title: Option<String>,
    pub snippet: String,
}

/// Result of cluster_documents()
pub struct ClusterDocumentsResult {
    pub matched_docs: Vec<MatchedDocument>,
    pub assignments: Vec<usize>,
    pub unmatched_urls: Vec<String>,
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
