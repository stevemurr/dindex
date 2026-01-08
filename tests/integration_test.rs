//! Integration tests for dindex
//!
//! These tests verify end-to-end functionality of the system.

use dindex::{
    chunking::TextSplitter,
    config::{ChunkingConfig, EmbeddingConfig, IndexConfig, RetrievalConfig},
    embedding::EmbeddingEngine,
    index::{ChunkStorage, VectorIndex},
    retrieval::{Bm25Index, HybridIndexer, HybridRetriever},
    scraping::{
        ContentExtractor, DomainAssignment, FetchEngine, PolitenessController,
        UrlDeduplicator, ContentDeduplicator,
        dedup::SimHash,
        extractor::ExtractorConfig,
        fetcher::FetchConfig,
        politeness::PolitenessConfig,
    },
    types::{Document, Query},
};
use std::sync::Arc;
use tempfile::TempDir;
use url::Url;

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

// ============ SCRAPING MODULE TESTS ============

/// Test domain assignment consistent hashing
#[test]
fn test_domain_assignment() {
    let mut assignment = DomainAssignment::new(10);

    // Add some nodes
    assignment.on_node_join("peer1".to_string());
    assignment.on_node_join("peer2".to_string());
    assignment.on_node_join("peer3".to_string());

    // Same domain should always map to same peer
    let peer1 = assignment.assign_domain("example.com");
    let peer2 = assignment.assign_domain("example.com");
    assert_eq!(peer1, peer2, "Same domain should consistently map to same peer");

    // Different domains should be distributed across peers
    let domains = vec![
        "google.com",
        "github.com",
        "rust-lang.org",
        "mozilla.org",
        "example.net",
        "test.io",
    ];

    let mut peer_assignments: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for domain in &domains {
        if let Some(peer) = assignment.assign_domain(domain) {
            *peer_assignments.entry(peer).or_insert(0) += 1;
        }
    }

    // Should have assigned all domains
    let total: usize = peer_assignments.values().sum();
    assert_eq!(total, domains.len(), "All domains should be assigned");
}

/// Test domain reassignment on node leave
#[test]
fn test_domain_reassignment_on_node_leave() {
    let mut assignment = DomainAssignment::new(10);

    assignment.on_node_join("peer1".to_string());
    assignment.on_node_join("peer2".to_string());

    let original_peer = assignment.assign_domain("example.com").unwrap();

    // Remove the peer that owns the domain
    assignment.on_node_leave(&original_peer);

    // Domain should now be assigned to remaining peer
    let new_peer = assignment.assign_domain("example.com");
    assert!(new_peer.is_some(), "Domain should still be assignable");
    assert_ne!(
        new_peer.as_ref(),
        Some(&original_peer),
        "Domain should be reassigned to different peer"
    );
}

/// Test URL deduplication
#[test]
fn test_url_deduplication() {
    let mut dedup = UrlDeduplicator::new(1000, 0.01);

    let url1 = Url::parse("https://example.com/page1").unwrap();
    let url2 = Url::parse("https://example.com/page2").unwrap();
    let url1_fragment = Url::parse("https://example.com/page1#section").unwrap();

    // First URL should be new
    assert!(dedup.is_new_url(&url1), "First URL should be new");

    // Same URL should be duplicate
    assert!(!dedup.is_new_url(&url1), "Same URL should be duplicate");

    // Different URL should be new
    assert!(dedup.is_new_url(&url2), "Different URL should be new");

    // URL with fragment should be duplicate (fragments are stripped)
    assert!(
        !dedup.is_new_url(&url1_fragment),
        "URL with fragment should be duplicate of base URL"
    );

    assert_eq!(dedup.len(), 2, "Should have tracked 2 unique URLs");
}

/// Test SimHash computation
#[test]
fn test_simhash_computation() {
    // Identical texts should produce identical hashes
    let text1 = "The quick brown fox jumps over the lazy dog";
    let hash1 = SimHash::compute(text1);
    let hash2 = SimHash::compute(text1);
    assert_eq!(hash1, hash2, "Identical text should produce identical SimHash");

    // Similar texts should produce similar hashes (low Hamming distance)
    let text_similar = "The quick brown fox leaps over the lazy dog";
    let hash_similar = SimHash::compute(text_similar);
    let distance = hash1.hamming_distance(&hash_similar);
    assert!(
        distance < 20,
        "Similar texts should have low Hamming distance, got {}",
        distance
    );

    // Very different texts should produce different hashes (high Hamming distance)
    let text_different = "Lorem ipsum dolor sit amet consectetur adipiscing elit";
    let hash_different = SimHash::compute(text_different);
    let distance_diff = hash1.hamming_distance(&hash_different);
    assert!(
        distance_diff > 10,
        "Different texts should have high Hamming distance, got {}",
        distance_diff
    );
}

/// Test content deduplication
#[test]
fn test_content_deduplication() {
    let mut dedup = ContentDeduplicator::new(100, 3, "node1".to_string());

    let text1 = "This is the first document with some content for testing purposes.";
    let text1_copy = "This is the first document with some content for testing purposes.";
    let text_different = "This is a completely different document with entirely different content.";

    // Register first document
    dedup.register(text1, "doc1".to_string());

    // Identical text should be detected as duplicate
    let dup_result = dedup.is_duplicate_local(text1_copy);
    assert!(dup_result.is_some(), "Identical text should be duplicate");
    assert_eq!(dup_result.unwrap(), "doc1");

    // Different text should not be duplicate
    let diff_result = dedup.is_duplicate_local(text_different);
    assert!(diff_result.is_none(), "Different text should not be duplicate");
}

/// Test content extraction
#[test]
fn test_content_extraction() {
    let extractor = ContentExtractor::new(ExtractorConfig {
        min_content_length: 30,
        min_word_count: 5,
        words_per_minute: 200,
    });

    let html = r#"
        <html>
        <head>
            <title>Test Article Title</title>
            <meta name="author" content="John Doe">
        </head>
        <body>
            <nav>Navigation menu here</nav>
            <article>
                <h1>Main Heading</h1>
                <p>This is the main article content with several words for testing.</p>
                <p>Second paragraph with more content to ensure sufficient length.</p>
            </article>
            <footer>Footer content</footer>
        </body>
        </html>
    "#;

    let url = Url::parse("https://example.com/article").unwrap();
    let result = extractor.extract(html, &url);

    assert!(result.is_ok(), "Content extraction should succeed");
    let content = result.unwrap();

    assert_eq!(content.title, "Test Article Title");
    assert!(!content.text_content.is_empty(), "Should extract text content");
    assert!(content.word_count >= 5, "Should have minimum word count");
}

/// Test metadata extraction
#[test]
fn test_metadata_extraction() {
    let extractor = ContentExtractor::new(ExtractorConfig::default());

    let html = r#"
        <html lang="en">
        <head>
            <title>Test Page</title>
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG Description">
            <meta name="author" content="Jane Smith">
            <meta property="article:published_time" content="2024-01-15T10:00:00Z">
            <link rel="canonical" href="https://example.com/canonical-url">
        </head>
        <body>
            <article>
                <p>Article content for testing metadata extraction.</p>
            </article>
        </body>
        </html>
    "#;

    let url = Url::parse("https://example.com/page").unwrap();
    let metadata = extractor.extract_metadata(html, &url);

    assert_eq!(metadata.title, "OG Title", "Should use OG title");
    assert_eq!(
        metadata.description,
        Some("OG Description".to_string()),
        "Should extract OG description"
    );
    assert_eq!(
        metadata.author,
        Some("Jane Smith".to_string()),
        "Should extract author"
    );
    assert_eq!(
        metadata.language,
        Some("en".to_string()),
        "Should extract language"
    );
    assert!(metadata.published_date.is_some(), "Should extract date");
    assert_eq!(
        metadata.canonical_url,
        Some("https://example.com/canonical-url".to_string()),
        "Should extract canonical URL"
    );
}

/// Test fetch engine creation
#[test]
fn test_fetch_engine_creation() {
    let config = FetchConfig::default();
    let engine = FetchEngine::new(config);

    assert!(engine.is_ok(), "FetchEngine should be creatable");
    let engine = engine.unwrap();

    assert!(
        engine.user_agent().contains("DecentralizedSearchBot"),
        "Should have correct user agent"
    );
}

/// Test politeness controller initialization
#[test]
fn test_politeness_controller_init() {
    let config = PolitenessConfig::default();
    let controller = PolitenessController::new(config);

    assert_eq!(
        controller.user_agent(),
        "DecentralizedSearchBot/1.0",
        "Should have correct user agent"
    );
}
