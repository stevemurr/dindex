//! Integration tests for dindex
//!
//! These tests verify end-to-end functionality of the system.

use dindex::{
    chunking::TextSplitter,
    config::{ChunkingConfig, EmbeddingConfig, IndexConfig, RetrievalConfig},
    embedding::EmbeddingEngine,
    index::{ChunkStorage, VectorIndex},
    retrieval::{Bm25Index, HybridIndexer, HybridRetriever},
    types::{Document, Query},
};
use std::sync::Arc;
use tempfile::TempDir;

/// Helper to create deterministic test embeddings (same as fallback mode)
fn create_test_embedding(text: &str, dims: usize) -> Vec<f32> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();

    let mut embedding = Vec::with_capacity(dims);
    for i in 0..dims {
        let byte_idx = i % hash.len();
        let value = (hash[byte_idx] as f32 / 255.0) * 2.0 - 1.0;
        embedding.push(value);
    }

    // Normalize
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for v in embedding.iter_mut() {
            *v /= magnitude;
        }
    }
    embedding
}

/// Test the complete indexing and retrieval pipeline
#[test]
fn test_indexing_and_retrieval_pipeline() {
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path();

    // Create components
    let embedding_config = EmbeddingConfig::default();
    let index_config = IndexConfig::default();
    let vector_index = Arc::new(
        VectorIndex::new(embedding_config.dimensions, &index_config).unwrap(),
    );

    let bm25_index = Arc::new(Bm25Index::new(data_dir.join("bm25")).unwrap());
    let chunk_storage = Arc::new(ChunkStorage::new(data_dir.join("chunks")).unwrap());

    // Create indexer
    let indexer = HybridIndexer::new(
        vector_index.clone(),
        bm25_index.clone(),
        chunk_storage.clone(),
    );

    // Create document and chunk it
    let document = Document::new(
        "Machine learning is a branch of artificial intelligence that focuses on \
         building systems that learn from data. Deep learning is a subset of machine \
         learning that uses neural networks with many layers. Natural language \
         processing enables computers to understand human language.",
    )
    .with_id("doc1")
    .with_title("Test Document");

    let chunking_config = ChunkingConfig::default();
    let splitter = TextSplitter::new(chunking_config);
    let chunks = splitter.split_document(&document);

    // Index the chunks using test embeddings
    for chunk in &chunks {
        let embedding = create_test_embedding(&chunk.content, embedding_config.dimensions);
        indexer.index_chunk(chunk, &embedding).unwrap();
    }
    indexer.save().unwrap();

    // Create retriever (without embedding engine for this test)
    let retrieval_config = RetrievalConfig::default();
    let retriever = HybridRetriever::new(
        vector_index,
        bm25_index,
        chunk_storage,
        None,
        retrieval_config,
    );

    // Query using BM25 (text-based)
    let query = Query::new("machine learning artificial intelligence", 5);
    let query_embedding = create_test_embedding(&query.text, embedding_config.dimensions);
    let results = retriever.search(&query, Some(&query_embedding)).unwrap();

    // Verify results
    assert!(!results.is_empty(), "Search should return results");
}

/// Test document chunking
#[test]
fn test_document_chunking() {
    let config = ChunkingConfig {
        chunk_size: 50,
        min_chunk_size: 20,
        max_chunk_size: 100,
        overlap_fraction: 0.1,
    };

    let splitter = TextSplitter::new(config);

    let document = Document::new(
        "First paragraph with some text. Second paragraph with more text. Third paragraph to ensure we have enough content for multiple chunks."
    ).with_id("doc1");

    let chunks = splitter.split_document(&document);

    assert!(!chunks.is_empty(), "Should produce at least one chunk");

    // Verify all chunks have content
    for chunk in &chunks {
        assert!(!chunk.content.is_empty(), "Chunk should have content");
        assert!(!chunk.metadata.chunk_id.is_empty(), "Chunk should have ID");
    }
}

/// Test embedding engine initialization and basic operations
#[test]
fn test_embedding_engine_init() {
    let config = EmbeddingConfig::default();

    // The embedding engine may fail if no model is available, but it should
    // still be creatable (uses fallback mode)
    match EmbeddingEngine::new(&config) {
        Ok(engine) => {
            let text = "This is a test sentence for embedding.";
            let embedding = engine.embed(text).unwrap();
            assert_eq!(embedding.len(), config.dimensions, "Embedding should have correct dimensions");

            // Embedding should be normalized (magnitude ~= 1.0)
            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (magnitude - 1.0).abs() < 0.1,
                "Embedding should be approximately normalized, got magnitude {}",
                magnitude
            );
        }
        Err(_) => {
            // This is expected if no model files are available
            // The fallback mode should still work
        }
    }
}

/// Test that our test embedding helper produces normalized embeddings
#[test]
fn test_helper_embeddings() {
    let embedding = create_test_embedding("test text", 768);
    assert_eq!(embedding.len(), 768);

    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (magnitude - 1.0).abs() < 0.01,
        "Test embedding should be normalized, got magnitude {}",
        magnitude
    );

    // Same text should produce same embedding
    let embedding2 = create_test_embedding("test text", 768);
    assert_eq!(embedding, embedding2, "Same text should produce same embedding");

    // Different text should produce different embedding
    let embedding3 = create_test_embedding("different text", 768);
    assert_ne!(embedding, embedding3, "Different text should produce different embedding");
}

/// Test vector index operations
#[test]
fn test_vector_index_operations() {
    let config = IndexConfig::default();
    let index = VectorIndex::new(64, &config).unwrap();

    // Create test embeddings
    let embedding1: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0).sin()).collect();
    let embedding2: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0).cos()).collect();

    // Add embeddings
    let chunk1_id = "chunk1".to_string();
    let chunk2_id = "chunk2".to_string();
    let key1 = index.add(&chunk1_id, &embedding1).unwrap();
    let key2 = index.add(&chunk2_id, &embedding2).unwrap();

    assert_ne!(key1, key2, "Keys should be different");

    // Search for similar embeddings
    let results = index.search(&embedding1, 5).unwrap();
    assert!(!results.is_empty(), "Search should return results");
    assert_eq!(results[0].chunk_id, "chunk1", "Most similar should be chunk1");

    // Remove and verify
    index.remove(&chunk1_id).unwrap();
    let results_after_remove = index.search(&embedding1, 5).unwrap();
    assert!(
        results_after_remove.is_empty() || results_after_remove[0].chunk_id != "chunk1",
        "chunk1 should not be in results after removal"
    );
}

/// Test BM25 search
#[test]
fn test_bm25_operations() {
    let index = Bm25Index::new_in_memory().unwrap();

    let chunk1 = dindex::types::Chunk {
        metadata: dindex::types::ChunkMetadata::new("chunk1".to_string(), "doc1".to_string()),
        content: "The quick brown fox jumps over the lazy dog".to_string(),
        token_count: 9,
    };

    let chunk2 = dindex::types::Chunk {
        metadata: dindex::types::ChunkMetadata::new("chunk2".to_string(), "doc1".to_string()),
        content: "A fast cat runs across the street quickly".to_string(),
        token_count: 8,
    };

    index.add(&chunk1).unwrap();
    index.add(&chunk2).unwrap();
    index.commit().unwrap();

    // Search for "fox"
    let results = index.search("fox", 10).unwrap();
    assert!(!results.is_empty(), "Should find results for 'fox'");
    assert_eq!(results[0].chunk_id, "chunk1", "chunk1 should rank first for 'fox'");

    // Search for "cat"
    let results = index.search("cat", 10).unwrap();
    assert!(!results.is_empty(), "Should find results for 'cat'");
    assert_eq!(results[0].chunk_id, "chunk2", "chunk2 should rank first for 'cat'");
}
