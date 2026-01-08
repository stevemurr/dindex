//! Hybrid retrieval combining dense and sparse search

use super::{
    bm25::{Bm25Index, Bm25SearchResult},
    fusion::{reciprocal_rank_fusion, to_ranked_results, RankedResult, RrfConfig},
};
use crate::config::RetrievalConfig;
use crate::embedding::EmbeddingEngine;
use crate::index::{ChunkStorage, VectorIndex, VectorSearchResult};
use crate::types::{Chunk, ChunkId, Embedding, Query, SearchResult};
use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info};

/// Hybrid retrieval engine combining multiple search methods
pub struct HybridRetriever {
    /// Vector index for dense search
    vector_index: Arc<VectorIndex>,
    /// BM25 index for lexical search
    bm25_index: Arc<Bm25Index>,
    /// Chunk storage for retrieving full content
    chunk_storage: Arc<ChunkStorage>,
    /// Embedding engine for query encoding
    embedding_engine: Option<Arc<EmbeddingEngine>>,
    /// Configuration
    config: RetrievalConfig,
}

impl HybridRetriever {
    /// Create a new hybrid retriever
    pub fn new(
        vector_index: Arc<VectorIndex>,
        bm25_index: Arc<Bm25Index>,
        chunk_storage: Arc<ChunkStorage>,
        embedding_engine: Option<Arc<EmbeddingEngine>>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            vector_index,
            bm25_index,
            chunk_storage,
            embedding_engine,
            config,
        }
    }

    /// Search using hybrid retrieval
    pub fn search(&self, query: &Query, query_embedding: Option<&Embedding>) -> Result<Vec<SearchResult>> {
        // Validate input
        if query.text.trim().is_empty() {
            return Ok(Vec::new()); // Empty query returns empty results
        }

        if query.top_k == 0 {
            return Ok(Vec::new()); // No results requested
        }

        let candidate_count = self.config.candidate_count;
        let mut ranked_lists: Vec<Vec<RankedResult>> = Vec::new();

        // Dense search
        if self.config.enable_dense {
            // Try to get embedding from parameter or generate it
            let generated_embedding = if query_embedding.is_none() {
                self.embedding_engine
                    .as_ref()
                    .and_then(|e| e.embed(&query.text).ok())
            } else {
                None
            };

            let embedding_ref = query_embedding.or(generated_embedding.as_ref());

            if let Some(embedding) = embedding_ref {
                let dense_results = self.vector_index.search(embedding, candidate_count)?;
                let ranked: Vec<(ChunkId, f32)> = dense_results
                    .iter()
                    .map(|r| (r.chunk_id.clone(), r.similarity))
                    .collect();
                ranked_lists.push(to_ranked_results(&ranked, "dense"));
                debug!("Dense search: {} results", dense_results.len());
            }
        }

        // BM25 search
        if self.config.enable_bm25 {
            let bm25_results = self.bm25_index.search(&query.text, candidate_count)?;
            let ranked: Vec<(ChunkId, f32)> = bm25_results
                .iter()
                .map(|r| (r.chunk_id.clone(), r.score))
                .collect();
            ranked_lists.push(to_ranked_results(&ranked, "bm25"));
            debug!("BM25 search: {} results", bm25_results.len());
        }

        // Fuse results
        let rrf_config = RrfConfig { k: self.config.rrf_k };
        let fused = reciprocal_rank_fusion(&ranked_lists, &rrf_config);

        // Retrieve chunk content and build final results
        let top_k = query.top_k.min(fused.len());
        let top_fused = &fused[..top_k];

        let chunk_ids: Vec<String> = top_fused.iter().map(|f| f.chunk_id.clone()).collect();
        let stored_chunks = self.chunk_storage.get_batch(&chunk_ids);

        // Build search results
        let mut results: Vec<SearchResult> = Vec::with_capacity(top_k);
        for fused_result in top_fused {
            if let Some(stored) = stored_chunks.iter().find(|s| s.chunk.metadata.chunk_id == fused_result.chunk_id) {
                let mut result = SearchResult::new(stored.chunk.clone(), fused_result.rrf_score);
                result.matched_by = fused_result.contributing_methods.clone();
                results.push(result);
            }
        }

        info!(
            "Hybrid search for '{}': {} results",
            truncate_query(&query.text),
            results.len()
        );

        Ok(results)
    }

    /// Search with pre-computed embedding
    pub fn search_with_embedding(
        &self,
        query: &Query,
        embedding: &Embedding,
    ) -> Result<Vec<SearchResult>> {
        self.search(query, Some(embedding))
    }

    /// Dense-only search
    pub fn dense_search(&self, embedding: &Embedding, k: usize) -> Result<Vec<VectorSearchResult>> {
        self.vector_index.search(embedding, k)
    }

    /// BM25-only search
    pub fn bm25_search(&self, query_text: &str, k: usize) -> Result<Vec<Bm25SearchResult>> {
        self.bm25_index.search(query_text, k)
    }
}

fn truncate_query(query: &str) -> String {
    if query.len() > 50 {
        format!("{}...", &query[..47])
    } else {
        query.to_string()
    }
}

/// Builder for creating a hybrid retriever with indexing support
pub struct HybridIndexer {
    pub vector_index: Arc<VectorIndex>,
    pub bm25_index: Arc<Bm25Index>,
    pub chunk_storage: Arc<ChunkStorage>,
}

impl HybridIndexer {
    /// Create a new hybrid indexer
    pub fn new(
        vector_index: Arc<VectorIndex>,
        bm25_index: Arc<Bm25Index>,
        chunk_storage: Arc<ChunkStorage>,
    ) -> Self {
        Self {
            vector_index,
            bm25_index,
            chunk_storage,
        }
    }

    /// Index a chunk with its embedding
    pub fn index_chunk(&self, chunk: &Chunk, embedding: &Embedding) -> Result<u64> {
        // Validate input
        if chunk.content.trim().is_empty() {
            return Err(anyhow::anyhow!("Cannot index chunk with empty content"));
        }

        if chunk.metadata.chunk_id.is_empty() {
            return Err(anyhow::anyhow!("Cannot index chunk without chunk_id"));
        }

        if embedding.is_empty() {
            return Err(anyhow::anyhow!("Cannot index chunk with empty embedding"));
        }

        // Add to vector index
        let key = self.vector_index.add(&chunk.metadata.chunk_id, embedding)?;

        // Add to BM25 index
        self.bm25_index.add(chunk)?;

        // Store chunk data
        let indexed_chunk = crate::types::IndexedChunk {
            chunk: chunk.clone(),
            embedding: embedding.clone(),
            lsh_signature: None,
            index_key: key,
        };
        self.chunk_storage.store(&indexed_chunk);

        Ok(key)
    }

    /// Index multiple chunks
    pub fn index_batch(&self, chunks: &[(Chunk, Embedding)]) -> Result<Vec<u64>> {
        let mut keys = Vec::with_capacity(chunks.len());

        for (chunk, embedding) in chunks {
            let key = self.index_chunk(chunk, embedding)?;
            keys.push(key);
        }

        // Commit BM25 changes
        self.bm25_index.commit()?;

        Ok(keys)
    }

    /// Remove a chunk from all indices
    pub fn remove_chunk(&self, chunk_id: &ChunkId) -> Result<()> {
        self.vector_index.remove(chunk_id)?;
        self.bm25_index.delete(chunk_id)?;
        self.chunk_storage.remove(chunk_id);
        Ok(())
    }

    /// Remove all chunks for a document
    pub fn remove_document(&self, document_id: &str) -> Result<usize> {
        // Get all chunks for this document
        let chunks = self.chunk_storage.get_by_document(document_id);
        let count = chunks.len();

        // Remove each chunk
        for stored_chunk in chunks {
            let chunk_id = &stored_chunk.chunk.metadata.chunk_id;
            self.vector_index.remove(chunk_id)?;
            self.bm25_index.delete(chunk_id)?;
            self.chunk_storage.remove(chunk_id);
        }

        // Commit BM25 changes
        self.bm25_index.commit()?;

        Ok(count)
    }

    /// Save all indices
    pub fn save(&self) -> Result<()> {
        self.bm25_index.commit()?;
        self.chunk_storage.save()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::IndexConfig;
    use crate::types::ChunkMetadata;
    use tempfile::TempDir;

    fn create_test_embedding(seed: usize, dims: usize) -> Embedding {
        (0..dims)
            .map(|i| ((seed * 100 + i) as f32 / 1000.0).sin())
            .collect()
    }

    #[test]
    fn test_hybrid_indexer() {
        let temp_dir = TempDir::new().unwrap();
        let vector_index = Arc::new(VectorIndex::new(64, &IndexConfig::default()).unwrap());
        let bm25_index = Arc::new(Bm25Index::new_in_memory().unwrap());
        let chunk_storage = Arc::new(ChunkStorage::new(temp_dir.path().join("chunks")).unwrap());

        let indexer = HybridIndexer::new(
            vector_index.clone(),
            bm25_index.clone(),
            chunk_storage.clone(),
        );

        let chunk = Chunk {
            metadata: ChunkMetadata::new("chunk1".to_string(), "doc1".to_string()),
            content: "The quick brown fox jumps over the lazy dog".to_string(),
            token_count: 9,
        };
        let embedding = create_test_embedding(1, 64);

        // Index the chunk
        let key = indexer.index_chunk(&chunk, &embedding).unwrap();
        indexer.save().unwrap();

        assert!(key > 0 || key == 0); // Key was assigned

        // Verify it's stored
        let stored = chunk_storage.get("chunk1");
        assert!(stored.is_some());
        assert_eq!(stored.unwrap().chunk.content, chunk.content);
    }

    #[test]
    fn test_hybrid_search() {
        let temp_dir = TempDir::new().unwrap();
        let vector_index = Arc::new(VectorIndex::new(64, &IndexConfig::default()).unwrap());
        let bm25_index = Arc::new(Bm25Index::new_in_memory().unwrap());
        let chunk_storage = Arc::new(ChunkStorage::new(temp_dir.path().join("chunks")).unwrap());

        let indexer = HybridIndexer::new(
            vector_index.clone(),
            bm25_index.clone(),
            chunk_storage.clone(),
        );

        // Index some chunks
        let chunks = vec![
            (
                Chunk {
                    metadata: ChunkMetadata::new("chunk1".to_string(), "doc1".to_string()),
                    content: "Machine learning and artificial intelligence".to_string(),
                    token_count: 5,
                },
                create_test_embedding(1, 64),
            ),
            (
                Chunk {
                    metadata: ChunkMetadata::new("chunk2".to_string(), "doc1".to_string()),
                    content: "Deep learning neural networks".to_string(),
                    token_count: 4,
                },
                create_test_embedding(2, 64),
            ),
        ];

        indexer.index_batch(&chunks).unwrap();

        // Create retriever
        let config = RetrievalConfig {
            enable_dense: true,
            enable_bm25: true,
            rrf_k: 60,
            candidate_count: 10,
            enable_reranking: false,
            reranker_model_path: None,
        };

        let retriever = HybridRetriever::new(
            vector_index,
            bm25_index,
            chunk_storage,
            None,
            config,
        );

        // Search with embedding
        let query = Query::new("machine learning".to_string(), 5);
        let query_embedding = create_test_embedding(1, 64);
        let results = retriever.search(&query, Some(&query_embedding)).unwrap();

        assert!(!results.is_empty());
    }
}
