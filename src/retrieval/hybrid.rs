//! Hybrid retrieval combining dense and sparse search

use super::{
    bm25::Bm25Index,
    fusion::{reciprocal_rank_fusion, to_ranked_results, RankedResult, RetrievalMethod, RrfConfig},
    reranker::SimpleReranker,
};
use crate::config::RetrievalConfig;
use crate::embedding::EmbeddingEngine;
use crate::index::{ChunkStorage, VectorIndex};
use crate::types::{Chunk, ChunkId, Embedding, Query, QueryFilters, SearchResult};
use crate::util::truncate_str;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info, warn};

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

        let has_filters = query.filters.as_ref().is_some_and(|f| !Self::filters_empty(f));

        // Over-fetch when filtering so we have enough results after filtering
        let candidate_count = if has_filters {
            self.config.candidate_count * 3
        } else {
            self.config.candidate_count
        };
        let mut ranked_lists: Vec<Vec<RankedResult>> = Vec::new();

        // Dense search
        if self.config.enable_dense {
            // Try to get embedding from parameter or generate it
            let generated_embedding = if query_embedding.is_none() {
                self.embedding_engine.as_ref().and_then(|e| {
                    e.embed(&query.text)
                        .map_err(|err| {
                            warn!("Failed to generate query embedding for dense search: {}", err);
                            err
                        })
                        .ok()
                })
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
                ranked_lists.push(to_ranked_results(&ranked, RetrievalMethod::Dense));
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
            ranked_lists.push(to_ranked_results(&ranked, RetrievalMethod::Bm25));
            debug!("BM25 search: {} results", bm25_results.len());
        }

        // Fuse results
        let rrf_config = RrfConfig { k: self.config.rrf_k };
        let fused = reciprocal_rank_fusion(&ranked_lists, &rrf_config);

        // Retrieve chunk content and build final results
        // When filtering, fetch more candidates to have enough after filtering
        let fetch_k = if has_filters {
            (query.top_k * 3).min(fused.len())
        } else {
            query.top_k.min(fused.len())
        };
        let top_fused = &fused[..fetch_k];

        let chunk_ids: Vec<String> = top_fused.iter().map(|f| f.chunk_id.clone()).collect();
        let stored_chunks = self.chunk_storage.get_batch(&chunk_ids);

        // Build HashMap for O(1) lookup instead of O(n) linear scan per result
        let chunk_map: HashMap<&str, _> = stored_chunks
            .iter()
            .map(|s| (s.chunk.metadata.chunk_id.as_str(), s))
            .collect();

        // Build search results
        let mut results: Vec<SearchResult> = Vec::with_capacity(fetch_k);
        for fused_result in top_fused {
            if let Some(stored) = chunk_map.get(fused_result.chunk_id.as_str()) {
                let mut result = SearchResult::new(stored.chunk.clone(), fused_result.rrf_score);
                result.matched_by = fused_result.contributing_methods.iter().map(|m| m.to_string()).collect();
                results.push(result);
            }
        }

        // Apply query filters
        if let Some(filters) = &query.filters {
            if !Self::filters_empty(filters) {
                let pre_filter_count = results.len();
                results = Self::apply_filters(results, filters);
                results.truncate(query.top_k);
                debug!("Filtered {} → {} results", pre_filter_count, results.len());
            }
        }

        // Apply reranking if enabled
        if self.config.enable_reranking {
            debug!("Reranking {} results", results.len());
            SimpleReranker::rerank(&query.text, &mut results);
        }

        info!(
            "Hybrid search for '{}': {} results",
            truncate_str(&query.text, 50),
            results.len()
        );

        Ok(results)
    }

    /// Check if all filter fields are empty/None
    fn filters_empty(filters: &QueryFilters) -> bool {
        filters.source_url_prefix.is_none()
            && filters.min_timestamp.is_none()
            && filters.max_timestamp.is_none()
            && filters.document_ids.is_none()
            && filters.metadata_equals.is_none()
            && filters.metadata_contains.is_none()
    }

    /// Apply query filters to search results
    fn apply_filters(results: Vec<SearchResult>, filters: &QueryFilters) -> Vec<SearchResult> {
        let document_id_set: Option<HashSet<&str>> = filters.document_ids.as_ref().map(|ids| {
            ids.iter().map(|s| s.as_str()).collect()
        });

        results
            .into_iter()
            .filter(|result| {
                let meta = &result.chunk.metadata;
                let extra = &meta.extra;

                // Filter by document IDs
                if let Some(ref id_set) = document_id_set {
                    if !id_set.contains(meta.document_id.as_str()) {
                        return false;
                    }
                }

                // Filter by source URL prefix
                if let Some(ref prefix) = filters.source_url_prefix {
                    match &meta.source_url {
                        Some(url) if url.starts_with(prefix) => {}
                        _ => return false,
                    }
                }

                // Filter by timestamp range
                if let Some(ref min_ts) = filters.min_timestamp {
                    if meta.timestamp < *min_ts {
                        return false;
                    }
                }
                if let Some(ref max_ts) = filters.max_timestamp {
                    if meta.timestamp > *max_ts {
                        return false;
                    }
                }

                // Filter by metadata exact match (all must match)
                if let Some(ref equals) = filters.metadata_equals {
                    for (key, value) in equals {
                        if extra.get(key) != Some(value) {
                            return false;
                        }
                    }
                }

                // Filter by metadata contains (value must be in allowed list)
                if let Some(ref contains) = filters.metadata_contains {
                    for (key, allowed_values) in contains {
                        if let Some(actual) = extra.get(key) {
                            let actual_values: HashSet<&str> =
                                actual.split(',').map(|s| s.trim()).collect();
                            if !allowed_values.iter().any(|v| actual_values.contains(v.as_str())) {
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

    fn make_chunk(id: &str, doc_id: &str, content: &str) -> Chunk {
        Chunk {
            metadata: ChunkMetadata::new(id.to_string(), doc_id.to_string()),
            content: content.to_string(),
            token_count: content.split_whitespace().count(),
        }
    }

    /// Helper to create a fully wired test indexer and retriever
    struct TestHarness {
        indexer: HybridIndexer,
        vector_index: Arc<VectorIndex>,
        bm25_index: Arc<Bm25Index>,
        chunk_storage: Arc<ChunkStorage>,
        _temp_dir: TempDir,
    }

    impl TestHarness {
        fn new(dims: usize) -> Self {
            let temp_dir = TempDir::new().unwrap();
            let vector_index =
                Arc::new(VectorIndex::new(dims, &IndexConfig::default()).unwrap());
            let bm25_index = Arc::new(Bm25Index::new_in_memory().unwrap());
            let chunk_storage =
                Arc::new(ChunkStorage::new(temp_dir.path().join("chunks")).unwrap());

            let indexer = HybridIndexer::new(
                vector_index.clone(),
                bm25_index.clone(),
                chunk_storage.clone(),
            );

            Self {
                indexer,
                vector_index,
                bm25_index,
                chunk_storage,
                _temp_dir: temp_dir,
            }
        }

        fn retriever(&self, enable_reranking: bool) -> HybridRetriever {
            let config = RetrievalConfig {
                enable_dense: true,
                enable_bm25: true,
                rrf_k: 60,
                candidate_count: 10,
                enable_reranking,
                reranker_model_path: None,
                ..Default::default()
            };
            HybridRetriever::new(
                self.vector_index.clone(),
                self.bm25_index.clone(),
                self.chunk_storage.clone(),
                None,
                config,
            )
        }
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
        let _key = indexer.index_chunk(&chunk, &embedding).unwrap();
        indexer.save().unwrap();

        // Key was assigned (any u64 value is valid)

        // Verify it's stored
        let stored = chunk_storage
            .get("chunk1")
            .expect("chunk should be stored");
        assert_eq!(stored.chunk.content, chunk.content);
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
            ..Default::default()
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

    #[test]
    fn test_remove_chunk_removes_from_vector_and_storage() {
        let harness = TestHarness::new(64);

        let chunk = make_chunk("c1", "doc1", "removable content alpha bravo");
        let embedding = create_test_embedding(1, 64);

        harness.indexer.index_chunk(&chunk, &embedding).unwrap();
        harness.indexer.save().unwrap();

        // Verify chunk exists in vector index and chunk storage
        assert!(harness.vector_index.contains(&"c1".to_string()));
        assert!(harness.chunk_storage.get("c1").is_some());

        // Remove the chunk
        harness.indexer.remove_chunk(&"c1".to_string()).unwrap();

        // Should be gone from vector index
        assert!(!harness.vector_index.contains(&"c1".to_string()));
        // Should be gone from chunk storage
        assert!(harness.chunk_storage.get("c1").is_none());
    }

    #[test]
    fn test_remove_document_removes_all_chunks() {
        let harness = TestHarness::new(64);

        let c1 = make_chunk("c1", "docX", "first chunk alpha content");
        let c2 = make_chunk("c2", "docX", "second chunk bravo content");
        let c3 = make_chunk("c3", "docY", "other document charlie content");

        harness
            .indexer
            .index_batch(&[
                (c1, create_test_embedding(1, 64)),
                (c2, create_test_embedding(2, 64)),
                (c3, create_test_embedding(3, 64)),
            ])
            .unwrap();

        // Remove all chunks for docX
        let removed_count = harness.indexer.remove_document("docX").unwrap();
        assert_eq!(removed_count, 2);

        // docX chunks should be gone
        assert!(!harness.vector_index.contains(&"c1".to_string()));
        assert!(!harness.vector_index.contains(&"c2".to_string()));
        assert!(harness.chunk_storage.get("c1").is_none());
        assert!(harness.chunk_storage.get("c2").is_none());

        // docY chunks should remain
        assert!(harness.vector_index.contains(&"c3".to_string()));
        assert!(harness.chunk_storage.get("c3").is_some());
    }

    #[test]
    fn test_remove_document_nonexistent_returns_zero() {
        let harness = TestHarness::new(64);
        let removed_count = harness.indexer.remove_document("no_such_doc").unwrap();
        assert_eq!(removed_count, 0);
    }

    #[test]
    fn test_empty_query_returns_empty_results() {
        let harness = TestHarness::new(64);

        let chunk = make_chunk("c1", "doc1", "some indexed content");
        harness
            .indexer
            .index_chunk(&chunk, &create_test_embedding(1, 64))
            .unwrap();
        harness.indexer.save().unwrap();

        let retriever = harness.retriever(false);
        let query = Query::new("".to_string(), 5);
        let results = retriever.search(&query, None).unwrap();
        assert!(results.is_empty(), "Empty query should return empty results");
    }

    #[test]
    fn test_whitespace_only_query_returns_empty() {
        let harness = TestHarness::new(64);

        let chunk = make_chunk("c1", "doc1", "some indexed content");
        harness
            .indexer
            .index_chunk(&chunk, &create_test_embedding(1, 64))
            .unwrap();
        harness.indexer.save().unwrap();

        let retriever = harness.retriever(false);
        let query = Query::new("   \t\n  ".to_string(), 5);
        let results = retriever.search(&query, None).unwrap();
        assert!(
            results.is_empty(),
            "Whitespace-only query should return empty results"
        );
    }

    #[test]
    fn test_top_k_zero_returns_empty() {
        let harness = TestHarness::new(64);

        let chunk = make_chunk("c1", "doc1", "searchable content");
        harness
            .indexer
            .index_chunk(&chunk, &create_test_embedding(1, 64))
            .unwrap();
        harness.indexer.save().unwrap();

        let retriever = harness.retriever(false);
        let query = Query::new("searchable".to_string(), 0);
        let embedding = create_test_embedding(1, 64);
        let results = retriever.search(&query, Some(&embedding)).unwrap();
        assert!(results.is_empty(), "top_k=0 should return empty results");
    }

    #[test]
    fn test_index_chunk_rejects_empty_content() {
        let harness = TestHarness::new(64);
        let chunk = make_chunk("c1", "doc1", "   ");
        let result = harness.indexer.index_chunk(&chunk, &create_test_embedding(1, 64));
        assert!(result.is_err(), "Should reject chunk with whitespace-only content");
    }

    #[test]
    fn test_index_chunk_rejects_empty_embedding() {
        let harness = TestHarness::new(64);
        let chunk = make_chunk("c1", "doc1", "valid content");
        let result = harness.indexer.index_chunk(&chunk, &vec![]);
        assert!(result.is_err(), "Should reject empty embedding");
    }

    fn make_chunk_with_url(id: &str, doc_id: &str, content: &str, url: &str) -> Chunk {
        let mut meta = ChunkMetadata::new(id.to_string(), doc_id.to_string());
        meta.source_url = Some(url.to_string());
        Chunk {
            metadata: meta,
            content: content.to_string(),
            token_count: content.split_whitespace().count(),
        }
    }

    fn make_chunk_with_metadata(id: &str, doc_id: &str, content: &str, extra: HashMap<String, String>) -> Chunk {
        let mut meta = ChunkMetadata::new(id.to_string(), doc_id.to_string());
        meta.extra = extra;
        Chunk {
            metadata: meta,
            content: content.to_string(),
            token_count: content.split_whitespace().count(),
        }
    }

    #[test]
    fn test_filter_by_document_ids() {
        let harness = TestHarness::new(64);

        let c1 = make_chunk("c1", "doc-a", "machine learning artificial intelligence");
        let c2 = make_chunk("c2", "doc-b", "machine learning deep neural networks");

        harness.indexer.index_batch(&[
            (c1, create_test_embedding(1, 64)),
            (c2, create_test_embedding(2, 64)),
        ]).unwrap();

        let retriever = harness.retriever(false);
        let mut query = Query::new("machine learning", 10);
        query.filters = Some(QueryFilters {
            document_ids: Some(vec!["doc-a".to_string()]),
            ..Default::default()
        });

        let embedding = create_test_embedding(1, 64);
        let results = retriever.search(&query, Some(&embedding)).unwrap();

        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.chunk.metadata.document_id, "doc-a");
        }
    }

    #[test]
    fn test_filter_by_source_url_prefix() {
        let harness = TestHarness::new(64);

        let c1 = make_chunk_with_url("c1", "doc1", "machine learning algorithms overview", "https://example.com/ml");
        let c2 = make_chunk_with_url("c2", "doc2", "machine learning deep networks overview", "https://other.com/dl");

        harness.indexer.index_batch(&[
            (c1, create_test_embedding(1, 64)),
            (c2, create_test_embedding(2, 64)),
        ]).unwrap();

        let retriever = harness.retriever(false);
        let mut query = Query::new("machine learning", 10);
        query.filters = Some(QueryFilters {
            source_url_prefix: Some("https://example.com".to_string()),
            ..Default::default()
        });

        let embedding = create_test_embedding(1, 64);
        let results = retriever.search(&query, Some(&embedding)).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.metadata.source_url.as_deref(), Some("https://example.com/ml"));
    }

    #[test]
    fn test_filter_by_metadata_equals() {
        let harness = TestHarness::new(64);

        let mut extra1 = HashMap::new();
        extra1.insert("lang".to_string(), "en".to_string());
        let c1 = make_chunk_with_metadata("c1", "doc1", "machine learning english content", extra1);

        let mut extra2 = HashMap::new();
        extra2.insert("lang".to_string(), "fr".to_string());
        let c2 = make_chunk_with_metadata("c2", "doc2", "machine learning french content", extra2);

        harness.indexer.index_batch(&[
            (c1, create_test_embedding(1, 64)),
            (c2, create_test_embedding(2, 64)),
        ]).unwrap();

        let retriever = harness.retriever(false);
        let mut query = Query::new("machine learning", 10);
        let mut equals = HashMap::new();
        equals.insert("lang".to_string(), "en".to_string());
        query.filters = Some(QueryFilters {
            metadata_equals: Some(equals),
            ..Default::default()
        });

        let embedding = create_test_embedding(1, 64);
        let results = retriever.search(&query, Some(&embedding)).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.metadata.extra.get("lang").unwrap(), "en");
    }

    #[test]
    fn test_filter_by_metadata_contains() {
        let harness = TestHarness::new(64);

        let mut extra1 = HashMap::new();
        extra1.insert("category".to_string(), "science, tech".to_string());
        let c1 = make_chunk_with_metadata("c1", "doc1", "machine learning algorithms science", extra1);

        let mut extra2 = HashMap::new();
        extra2.insert("category".to_string(), "sports".to_string());
        let c2 = make_chunk_with_metadata("c2", "doc2", "machine learning sports analytics", extra2);

        harness.indexer.index_batch(&[
            (c1, create_test_embedding(1, 64)),
            (c2, create_test_embedding(2, 64)),
        ]).unwrap();

        let retriever = harness.retriever(false);
        let mut query = Query::new("machine learning", 10);
        let mut contains = HashMap::new();
        contains.insert("category".to_string(), vec!["science".to_string(), "art".to_string()]);
        query.filters = Some(QueryFilters {
            metadata_contains: Some(contains),
            ..Default::default()
        });

        let embedding = create_test_embedding(1, 64);
        let results = retriever.search(&query, Some(&embedding)).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].chunk.metadata.extra.get("category").unwrap().contains("science"));
    }

    #[test]
    fn test_filter_by_timestamp_range() {
        use chrono::{TimeZone, Utc};

        let harness = TestHarness::new(64);

        let mut c1 = make_chunk("c1", "doc1", "machine learning old content article");
        c1.metadata.timestamp = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();

        let mut c2 = make_chunk("c2", "doc2", "machine learning recent content article");
        c2.metadata.timestamp = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();

        harness.indexer.index_batch(&[
            (c1, create_test_embedding(1, 64)),
            (c2, create_test_embedding(2, 64)),
        ]).unwrap();

        let retriever = harness.retriever(false);
        let mut query = Query::new("machine learning", 10);
        query.filters = Some(QueryFilters {
            min_timestamp: Some(Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap()),
            ..Default::default()
        });

        let embedding = create_test_embedding(1, 64);
        let results = retriever.search(&query, Some(&embedding)).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.metadata.chunk_id, "c2");
    }

    #[test]
    fn test_no_filters_returns_all() {
        let harness = TestHarness::new(64);

        let c1 = make_chunk("c1", "doc1", "machine learning algorithms overview");
        let c2 = make_chunk("c2", "doc2", "machine learning neural networks overview");

        harness.indexer.index_batch(&[
            (c1, create_test_embedding(1, 64)),
            (c2, create_test_embedding(2, 64)),
        ]).unwrap();

        let retriever = harness.retriever(false);
        let query = Query::new("machine learning", 10);

        let embedding = create_test_embedding(1, 64);
        let results = retriever.search(&query, Some(&embedding)).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_empty_filters_returns_all() {
        let harness = TestHarness::new(64);

        let c1 = make_chunk("c1", "doc1", "machine learning algorithms overview");
        let c2 = make_chunk("c2", "doc2", "machine learning neural networks overview");

        harness.indexer.index_batch(&[
            (c1, create_test_embedding(1, 64)),
            (c2, create_test_embedding(2, 64)),
        ]).unwrap();

        let retriever = harness.retriever(false);
        let mut query = Query::new("machine learning", 10);
        query.filters = Some(QueryFilters::default());

        let embedding = create_test_embedding(1, 64);
        let results = retriever.search(&query, Some(&embedding)).unwrap();

        assert_eq!(results.len(), 2, "Empty filters should not exclude any results");
    }
}
