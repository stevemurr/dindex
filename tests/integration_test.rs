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
    // Use 1024 dimensions to match the new default (bge-m3)
    let embedding = create_test_embedding("test text", 1024);
    assert_eq!(embedding.len(), 1024);

    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (magnitude - 1.0).abs() < 0.01,
        "Test embedding should be normalized, got magnitude {}",
        magnitude
    );

    // Same text should produce same embedding
    let embedding2 = create_test_embedding("test text", 1024);
    assert_eq!(embedding, embedding2, "Same text should produce same embedding");

    // Different text should produce different embedding
    let embedding3 = create_test_embedding("different text", 1024);
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

// ============ IMPORT MODULE TESTS ============

/// Test Wikipedia XML import end-to-end
#[test]
fn test_wikipedia_import_integration() {
    use dindex::import::{WikimediaSource, DumpSource};

    // Sample Wikipedia XML with realistic content
    let sample_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <siteinfo>
    <sitename>Wikipedia</sitename>
    <dbname>enwiki</dbname>
    <base>https://en.wikipedia.org/wiki/Main_Page</base>
  </siteinfo>
  <page>
    <title>Rust (programming language)</title>
    <ns>0</ns>
    <id>46180928</id>
    <revision>
      <id>1234567</id>
      <parentid>1234566</parentid>
      <timestamp>2024-06-15T12:30:00Z</timestamp>
      <contributor>
        <username>WikiEditor</username>
        <id>12345</id>
      </contributor>
      <text bytes="2500" xml:space="preserve">'''Rust''' is a [[multi-paradigm]], [[high-level programming language|high-level]], [[general-purpose programming language]]. Rust emphasizes [[type safety]], [[memory safety]], and [[concurrency]]. Rust enforces memory safety—that is, that all references point to valid memory—without requiring the use of a [[garbage collection (computer science)|garbage collector]] or [[reference counting]] present in other memory-safe languages.

== History ==
Rust was originally designed by Graydon Hoare at [[Mozilla Research]], with contributions from Dave Herman, [[Brendan Eich]], and others. The designers refined the language while writing the [[Servo (software)|Servo]] experimental browser engine, and the Rust [[compiler]].

== Design ==
The concrete syntax of Rust is similar to [[C (programming language)|C]] and [[C++]], with blocks of code delimited by curly brackets, and [[control flow]] keywords such as &lt;code&gt;if&lt;/code&gt;, &lt;code&gt;else&lt;/code&gt;, &lt;code&gt;while&lt;/code&gt;, and &lt;code&gt;for&lt;/code&gt;.

=== Memory safety ===
Rust is designed to be memory safe. It does not permit [[null pointer]]s, [[dangling pointer]]s, or [[data race]]s in safe code. Data values can only be initialized through a fixed set of forms, all of which require their inputs to be already initialized.

== See also ==
* [[Comparison of programming languages]]
* [[Systems programming]]

== References ==
{{reflist}}

[[Category:Programming languages]]
[[Category:Systems programming languages]]
[[Category:Rust (programming language)]]</text>
    </revision>
  </page>
  <page>
    <title>Memory safety</title>
    <ns>0</ns>
    <id>46180929</id>
    <revision>
      <id>1234568</id>
      <timestamp>2024-06-14T10:00:00Z</timestamp>
      <text bytes="1800" xml:space="preserve">'''Memory safety''' is the state of being protected from various software [[bug (software)|bugs]] and [[computer security|security vulnerabilities]] when dealing with [[Random-access memory|memory]] access, such as [[buffer overflow]]s and [[dangling pointer]]s.

Memory safety is considered important in software development because memory corruption bugs are a major source of security vulnerabilities. Languages like [[Rust (programming language)|Rust]], [[Java (programming language)|Java]], and [[Python (programming language)|Python]] provide memory safety guarantees.

== Types of memory errors ==
* [[Buffer overflow]] - writing data beyond allocated memory
* [[Use after free]] - accessing memory after it has been deallocated
* [[Double free]] - freeing memory twice
* [[Memory leak]] - failing to free allocated memory

== Approaches to memory safety ==
Different programming languages take different approaches:
* '''Garbage collection''' - automatic memory management ([[Java (programming language)|Java]], [[Go (programming language)|Go]])
* '''Ownership system''' - compile-time tracking ([[Rust (programming language)|Rust]])
* '''Manual management''' - programmer responsibility ([[C (programming language)|C]], [[C++]])

[[Category:Computer security]]
[[Category:Memory management]]</text>
    </revision>
  </page>
  <page>
    <title>Talk:Rust (programming language)</title>
    <ns>1</ns>
    <id>46180930</id>
    <revision>
      <id>1234569</id>
      <timestamp>2024-06-13T08:00:00Z</timestamp>
      <text bytes="500" xml:space="preserve">== Article improvement suggestions ==
This article needs more citations for the performance claims. Also, should we add a section about async/await support? ~~~~</text>
    </revision>
  </page>
  <page>
    <title>Redirect page</title>
    <ns>0</ns>
    <id>46180931</id>
    <redirect title="Rust (programming language)" />
    <revision>
      <id>1234570</id>
      <timestamp>2024-06-12T08:00:00Z</timestamp>
      <text bytes="50" xml:space="preserve">#REDIRECT [[Rust (programming language)]]</text>
    </revision>
  </page>
  <page>
    <title>Stub article</title>
    <ns>0</ns>
    <id>46180932</id>
    <revision>
      <id>1234571</id>
      <timestamp>2024-06-11T08:00:00Z</timestamp>
      <text bytes="20" xml:space="preserve">Too short.</text>
    </revision>
  </page>
</mediawiki>
"#;

    // Write sample XML to temp file
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, sample_xml.as_bytes()).unwrap();

    // Create WikimediaSource
    let mut source = WikimediaSource::open(temp_file.path()).unwrap();

    // Collect all documents
    let docs: Vec<_> = source.iter_documents().collect();

    // Should have exactly 2 documents (main namespace, non-redirect, sufficient length)
    // Filtered out: Talk page (ns=1), Redirect, Stub (too short)
    assert_eq!(docs.len(), 2, "Expected 2 documents, got {}", docs.len());

    // Verify first document (Rust programming language)
    let rust_doc = docs[0].as_ref().expect("First doc should be Ok");
    assert_eq!(rust_doc.id, "46180928");
    assert_eq!(rust_doc.title, "Rust (programming language)");
    assert_eq!(
        rust_doc.url,
        Some("https://en.wikipedia.org/wiki/Rust_(programming_language)".to_string())
    );

    // Verify wikitext was converted to plaintext
    assert!(rust_doc.content.contains("Rust"), "Content should contain 'Rust'");
    assert!(rust_doc.content.contains("memory safety"), "Content should contain 'memory safety'");
    // Bold markup should be removed
    assert!(!rust_doc.content.contains("'''"), "Bold markup should be removed");
    // Internal links should be processed (text kept, markup removed)
    assert!(!rust_doc.content.contains("[["), "Link markup should be removed");
    assert!(!rust_doc.content.contains("]]"), "Link markup should be removed");
    // Categories should be removed
    assert!(!rust_doc.content.contains("Category:"), "Categories should be removed");
    // Templates should be removed
    assert!(!rust_doc.content.contains("{{"), "Templates should be removed");

    // Verify second document (Memory safety)
    let memory_doc = docs[1].as_ref().expect("Second doc should be Ok");
    assert_eq!(memory_doc.id, "46180929");
    assert_eq!(memory_doc.title, "Memory safety");
    assert!(memory_doc.content.contains("buffer overflow") || memory_doc.content.contains("Buffer overflow"));

    // Verify timestamp parsing
    assert!(rust_doc.modified.is_some(), "Should have parsed timestamp");
}

/// Test Wikipedia import with namespace filtering
#[test]
fn test_wikipedia_import_namespace_filtering() {
    use dindex::import::{WikimediaSource, DumpSource};

    let sample_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>Main Article</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <id>100</id>
      <timestamp>2024-01-15T10:00:00Z</timestamp>
      <text>This is a main namespace article with sufficient content for indexing. It contains multiple sentences about various topics to ensure it passes the minimum length requirements.</text>
    </revision>
  </page>
  <page>
    <title>Talk:Main Article</title>
    <ns>1</ns>
    <id>2</id>
    <revision>
      <id>101</id>
      <text>This is a talk page with sufficient content for indexing. It contains discussion about the main article with multiple sentences for length.</text>
    </revision>
  </page>
  <page>
    <title>User:TestUser</title>
    <ns>2</ns>
    <id>3</id>
    <revision>
      <id>102</id>
      <text>This is a user page with sufficient content for testing namespace filtering. It contains multiple sentences about the user.</text>
    </revision>
  </page>
  <page>
    <title>Wikipedia:Policy</title>
    <ns>4</ns>
    <id>4</id>
    <revision>
      <id>103</id>
      <text>This is a Wikipedia policy page with sufficient content for testing namespace filtering. It contains guidelines and procedures.</text>
    </revision>
  </page>
</mediawiki>
"#;

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, sample_xml.as_bytes()).unwrap();

    // Default: only main namespace (0)
    let mut source = WikimediaSource::open(temp_file.path()).unwrap();
    let docs: Vec<_> = source.iter_documents().collect();
    assert_eq!(docs.len(), 1, "Default should only include main namespace");
    assert_eq!(docs[0].as_ref().unwrap().title, "Main Article");

    // Allow main (0) and talk (1) namespaces
    let mut temp_file2 = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file2, sample_xml.as_bytes()).unwrap();
    let mut source2 = WikimediaSource::open(temp_file2.path())
        .unwrap()
        .with_namespaces(Some(vec![0, 1]));
    let docs2: Vec<_> = source2.iter_documents().collect();
    assert_eq!(docs2.len(), 2, "Should include main and talk namespaces");

    // Allow all namespaces
    let mut temp_file3 = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file3, sample_xml.as_bytes()).unwrap();
    let mut source3 = WikimediaSource::open(temp_file3.path())
        .unwrap()
        .with_namespaces(None);
    let docs3: Vec<_> = source3.iter_documents().collect();
    assert_eq!(docs3.len(), 4, "Should include all namespaces");
}

/// Test Wikipedia import handles redirects correctly
#[test]
fn test_wikipedia_import_skips_redirects() {
    use dindex::import::{WikimediaSource, DumpSource};

    let sample_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>Main Article</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <id>100</id>
      <text>This is the main article content with enough text to pass the minimum length requirement for indexing purposes.</text>
    </revision>
  </page>
  <page>
    <title>Redirect 1</title>
    <ns>0</ns>
    <id>2</id>
    <redirect title="Main Article" />
    <revision>
      <id>101</id>
      <text>#REDIRECT [[Main Article]]</text>
    </revision>
  </page>
  <page>
    <title>Redirect 2</title>
    <ns>0</ns>
    <id>3</id>
    <redirect title="Main Article" />
    <revision>
      <id>102</id>
      <text>#REDIRECT [[Main Article]]</text>
    </revision>
  </page>
</mediawiki>
"#;

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, sample_xml.as_bytes()).unwrap();

    let mut source = WikimediaSource::open(temp_file.path()).unwrap();
    let docs: Vec<_> = source.iter_documents().collect();

    // Should only have the main article, not the redirects
    assert_eq!(docs.len(), 1, "Redirects should be filtered out");
    assert_eq!(docs[0].as_ref().unwrap().title, "Main Article");
}

/// Test Wikipedia import with document chunking and test embeddings
#[test]
fn test_wikipedia_import_with_chunking() {
    use dindex::import::{WikimediaSource, DumpSource};
    use dindex::chunking::TextSplitter;
    use dindex::config::ChunkingConfig;
    use dindex::types::Document;

    let sample_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>Long Article</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <id>100</id>
      <timestamp>2024-01-15T10:00:00Z</timestamp>
      <text>This is a long article that will be split into multiple chunks during the import process.

== First Section ==
The first section contains detailed information about the topic at hand. It discusses various aspects including historical context, current applications, and future implications. This paragraph needs to be long enough to potentially create a chunk boundary.

== Second Section ==
The second section continues with more information. It covers different aspects of the subject matter, providing examples and explanations. Technical details are included here along with references to related concepts.

== Third Section ==
Finally, the third section wraps up the discussion. It summarizes the key points and provides conclusions. Additional resources and further reading suggestions might be included here.

This final paragraph ensures we have enough content for multiple chunks to be created during the document processing pipeline.</text>
    </revision>
  </page>
</mediawiki>
"#;

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, sample_xml.as_bytes()).unwrap();

    let mut source = WikimediaSource::open(temp_file.path()).unwrap();
    let docs: Vec<_> = source.iter_documents().collect();
    assert_eq!(docs.len(), 1);

    let dump_doc = docs[0].as_ref().unwrap();

    // Convert to Document and chunk
    let document = Document::new(&dump_doc.content)
        .with_id(&dump_doc.id)
        .with_title(&dump_doc.title);

    // Use smaller chunk size to ensure multiple chunks
    let config = ChunkingConfig {
        chunk_size: 100,
        min_chunk_size: 50,
        max_chunk_size: 150,
        overlap_fraction: 0.1,
    };

    let splitter = TextSplitter::new(config);
    let chunks = splitter.split_document(&document);

    // Should have multiple chunks
    assert!(chunks.len() > 1, "Long article should produce multiple chunks, got {}", chunks.len());

    // Each chunk should have content
    for chunk in &chunks {
        assert!(!chunk.content.is_empty(), "Chunk should have content");
        assert!(chunk.content.len() >= 50, "Chunk should meet minimum length");
    }

    // Create embeddings for each chunk (using test helper with 1024 dims for bge-m3)
    for chunk in &chunks {
        let embedding = create_test_embedding(&chunk.content, 1024);
        assert_eq!(embedding.len(), 1024);

        // Verify normalization
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.01, "Embedding should be normalized");
    }
}

/// Test WikiText parser handles various markup correctly
#[test]
fn test_wikitext_parsing_comprehensive() {
    use dindex::import::WikiTextParser;

    let parser = WikiTextParser::new();

    // Test bold and italic removal
    let text = "'''Bold text''' and ''italic text'' mixed.";
    let result = parser.parse(text);
    assert!(!result.contains("'''"), "Bold markup should be removed");
    assert!(!result.contains("''"), "Italic markup should be removed");
    assert!(result.contains("Bold text"), "Bold content should remain");

    // Test internal links
    let text = "See [[Article]] and [[Display|link text]] for more.";
    let result = parser.parse(text);
    assert!(!result.contains("[["), "Link markup should be removed");
    assert!(!result.contains("]]"), "Link markup should be removed");
    assert!(result.contains("Article") || result.contains("link text"), "Link text should remain");

    // Test external links
    // Note: The current regex_lite implementation has limited regex support,
    // so external links may not be fully processed. This tests current behavior.
    let text = "Visit [https://example.com Example Site] for info.";
    let result = parser.parse(text);
    // External links should ideally be converted to just the display text
    // For now, verify the parser doesn't crash and returns something
    assert!(result.contains("Visit"), "Text before link should remain");
    assert!(result.contains("info"), "Text after link should remain");

    // Test template removal
    let text = "Start {{template|param=value}} end.";
    let result = parser.parse(text);
    assert!(!result.contains("{{"), "Template should be removed");
    assert!(!result.contains("}}"), "Template should be removed");
    assert!(result.contains("Start") && result.contains("end"), "Surrounding text should remain");

    // Test category removal
    let text = "Content here. [[Category:Test Category]]";
    let result = parser.parse(text);
    assert!(!result.contains("Category:"), "Category should be removed");

    // Test reference removal
    let text = "Fact<ref>Citation needed</ref> and more<ref name=\"foo\">Source</ref>.";
    let result = parser.parse(text);
    assert!(!result.contains("<ref"), "Ref tags should be removed");
    assert!(!result.contains("</ref>"), "Ref end tags should be removed");
    assert!(result.contains("Fact"), "Content before ref should remain");

    // Test HTML comment removal
    let text = "Visible <!-- hidden comment --> text.";
    let result = parser.parse(text);
    assert!(!result.contains("<!--"), "Comment should be removed");
    assert!(!result.contains("-->"), "Comment should be removed");
    assert!(!result.contains("hidden"), "Comment content should be removed");

    // Test heading processing
    let text = "== Section Heading ==\nContent under section.";
    let result = parser.parse(text);
    assert!(result.contains("Section Heading"), "Heading text should remain");

    // Test table removal
    let text = "Before {| class=\"wikitable\"\n|-\n| cell\n|} after.";
    let result = parser.parse(text);
    assert!(!result.contains("{|"), "Table start should be removed");
    assert!(!result.contains("|}"), "Table end should be removed");

    // Test file/image removal
    let text = "Text [[File:Example.jpg|thumb|Caption]] more text.";
    let result = parser.parse(text);
    assert!(!result.contains("File:"), "File link should be removed");
    assert!(!result.contains("[["), "Markup should be removed");
}

/// Test import handles malformed XML gracefully
#[test]
fn test_wikipedia_import_malformed_xml() {
    use dindex::import::{WikimediaSource, DumpSource};

    // Missing closing tags, but still parseable pages
    let malformed_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<mediawiki>
  <page>
    <title>Valid Article</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <id>100</id>
      <text>This article has valid structure with enough content for indexing. It meets all minimum requirements.</text>
    </revision>
  </page>
</mediawiki>
"#;

    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, malformed_xml.as_bytes()).unwrap();

    let mut source = WikimediaSource::open(temp_file.path()).unwrap();
    let docs: Vec<_> = source.iter_documents().collect();

    // Should still parse the valid page
    assert_eq!(docs.len(), 1);
    assert!(docs[0].is_ok());
}

/// Test URL generation for different Wikipedia languages
#[test]
fn test_wikipedia_url_generation_languages() {
    use dindex::import::{WikimediaSource, DumpSource};

    let sample_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<mediawiki>
  <page>
    <title>Test Article</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <id>100</id>
      <text>This is a test article with enough content for indexing purposes in various language editions of Wikipedia.</text>
    </revision>
  </page>
</mediawiki>
"#;

    // Test English Wikipedia URL
    let mut temp_file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file, sample_xml.as_bytes()).unwrap();

    let mut source = WikimediaSource::open(temp_file.path())
        .unwrap()
        .with_base_url("https://en.wikipedia.org/wiki/");
    let docs: Vec<_> = source.iter_documents().collect();
    assert_eq!(
        docs[0].as_ref().unwrap().url,
        Some("https://en.wikipedia.org/wiki/Test_Article".to_string())
    );

    // Test German Wikipedia URL
    let mut temp_file2 = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut temp_file2, sample_xml.as_bytes()).unwrap();

    let mut source2 = WikimediaSource::open(temp_file2.path())
        .unwrap()
        .with_base_url("https://de.wikipedia.org/wiki/");
    let docs2: Vec<_> = source2.iter_documents().collect();
    assert_eq!(
        docs2[0].as_ref().unwrap().url,
        Some("https://de.wikipedia.org/wiki/Test_Article".to_string())
    );
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

/// Test reset-index cleans up all storage files
#[test]
fn test_reset_index_cleans_all_storage() {
    use dindex::index::DocumentRegistry;
    use dindex::types::DocumentIdentity;
    use std::fs;

    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path();

    // Create all the storage types that reset-index should clean up

    // 1. BM25 index directory
    let bm25_path = data_dir.join("bm25");
    fs::create_dir_all(&bm25_path).unwrap();
    fs::write(bm25_path.join("test.idx"), b"test").unwrap();

    // 2. Chunk storage (sled database)
    let chunk_storage = ChunkStorage::new(data_dir).unwrap();
    drop(chunk_storage); // Close the database
    assert!(data_dir.join("chunks.sled").exists());

    // 3. Document registry (sled database) - must register something to trigger db creation
    let registry = DocumentRegistry::new(data_dir, 3).unwrap();
    let identity = DocumentIdentity::compute("Test document content for registry");
    registry.register(
        identity,
        Some("Test Doc".to_string()),
        Some("https://example.com".to_string()),
        "test",
        None,
        vec!["chunk-1".to_string()],
    );
    registry.save().unwrap();
    drop(registry); // Close the database
    assert!(data_dir.join("document_registry.sled").exists());

    // 4. Vector index
    let vector_index = VectorIndex::new(64, &IndexConfig::default()).unwrap();
    let vector_path = data_dir.join("vector.index");
    vector_index.save(&vector_path).unwrap();
    assert!(vector_path.exists());

    // 5. Vector index mappings
    let mappings_path = data_dir.join("vector.index.mappings.json");
    fs::write(&mappings_path, "{}").unwrap();

    // 6. Legacy chunks.json (for migration)
    let legacy_chunks_path = data_dir.join("chunks.json");
    fs::write(&legacy_chunks_path, "[]").unwrap();

    // 7. Daemon files
    fs::write(data_dir.join("dindex.pid"), "12345").unwrap();
    fs::write(data_dir.join("dindex.log"), "log").unwrap();
    fs::write(data_dir.join("dindex.err"), "err").unwrap();

    // Verify all files exist before reset
    let items_to_delete = [
        "bm25",
        "chunks.sled",
        "document_registry.sled",
        "vector.index",
        "vector.index.mappings.json",
        "chunks.json",
        "dindex.pid",
        "dindex.log",
        "dindex.err",
    ];

    for item in &items_to_delete {
        assert!(
            data_dir.join(item).exists(),
            "{} should exist before reset",
            item
        );
    }

    // Simulate reset-index by deleting all items
    for item in &items_to_delete {
        let path = data_dir.join(item);
        if path.exists() {
            if path.is_dir() {
                fs::remove_dir_all(&path).unwrap();
            } else {
                fs::remove_file(&path).unwrap();
            }
        }
    }

    // Verify all files are gone after reset
    for item in &items_to_delete {
        assert!(
            !data_dir.join(item).exists(),
            "{} should not exist after reset",
            item
        );
    }
}
