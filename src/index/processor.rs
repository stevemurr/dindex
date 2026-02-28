//! Unified Document Processor
//!
//! Single entry point for all document ingestion, handling deduplication
//! and index updates uniformly across all sources.

use super::registry::{DocumentRegistry, DuplicateCheckResult};
use crate::chunking::TextSplitter;
use crate::retrieval::HybridIndexer;
use crate::types::{Chunk, DocumentIdentity, Embedding};
use anyhow::Result;
use std::sync::Arc;
use tracing::{debug, info};

/// Result of processing a document
#[derive(Debug, Clone)]
pub enum ProcessingResult {
    /// New document indexed
    Indexed {
        content_id: String,
        chunks_created: usize,
    },
    /// Existing document - only metadata updated (exact match)
    MetadataUpdated {
        content_id: String,
    },
    /// Content updated (near-duplicate with changes)
    ContentUpdated {
        content_id: String,
        chunks_created: usize,
        chunks_removed: usize,
    },
    /// Document skipped
    Skipped {
        content_id: String,
        reason: String,
    },
}

impl ProcessingResult {
    /// Get the content ID if available
    pub fn content_id(&self) -> &str {
        match self {
            ProcessingResult::Indexed { content_id, .. } => content_id,
            ProcessingResult::MetadataUpdated { content_id } => content_id,
            ProcessingResult::ContentUpdated { content_id, .. } => content_id,
            ProcessingResult::Skipped { content_id, .. } => content_id,
        }
    }

    /// Check if the document was indexed (new or updated)
    pub fn was_indexed(&self) -> bool {
        matches!(
            self,
            ProcessingResult::Indexed { .. } | ProcessingResult::ContentUpdated { .. }
        )
    }

    /// Get number of chunks created
    pub fn chunks_created(&self) -> usize {
        match self {
            ProcessingResult::Indexed { chunks_created, .. } => *chunks_created,
            ProcessingResult::ContentUpdated { chunks_created, .. } => *chunks_created,
            _ => 0,
        }
    }
}

/// Configuration for document processing
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// Minimum content length to process
    pub min_content_length: usize,
    /// Enable deduplication
    pub dedup_enabled: bool,
    /// Maximum Hamming distance for near-duplicates
    pub simhash_threshold: u32,
    /// Whether to update content on near-duplicates
    pub update_near_duplicates: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            min_content_length: 100,
            dedup_enabled: true,
            simhash_threshold: 3,
            update_near_duplicates: true,
        }
    }
}

/// Unified document processor for all ingestion sources
pub struct DocumentProcessor {
    /// Document registry for deduplication
    registry: Arc<DocumentRegistry>,
    /// Hybrid indexer for storing documents
    indexer: Arc<HybridIndexer>,
    /// Text splitter for chunking
    splitter: TextSplitter,
    /// Processing configuration
    config: ProcessorConfig,
}

impl DocumentProcessor {
    /// Create a new document processor
    pub fn new(
        registry: Arc<DocumentRegistry>,
        indexer: Arc<HybridIndexer>,
        splitter: TextSplitter,
        config: ProcessorConfig,
    ) -> Self {
        Self {
            registry,
            indexer,
            splitter,
            config,
        }
    }

    /// Process a document
    ///
    /// # Arguments
    /// * `content` - The document content
    /// * `url` - Optional URL for the document
    /// * `title` - Optional document title
    /// * `source_type` - Type of source (e.g., "wikipedia", "web", "local")
    /// * `source_id` - Optional source-specific ID (key, value)
    /// * `embedding_fn` - Function to generate embeddings for text
    pub fn process<F>(
        &self,
        content: &str,
        url: Option<String>,
        title: Option<String>,
        source_type: &str,
        source_id: Option<(&str, &str)>,
        embedding_fn: F,
    ) -> Result<ProcessingResult>
    where
        F: Fn(&str) -> Embedding,
    {
        // Check content length
        if content.len() < self.config.min_content_length {
            let identity = DocumentIdentity::compute(content);
            return Ok(ProcessingResult::Skipped {
                content_id: identity.content_id.to_string(),
                reason: format!(
                    "Content too short ({} < {} chars)",
                    content.len(),
                    self.config.min_content_length
                ),
            });
        }

        // Compute document identity
        let identity = DocumentIdentity::compute(content);
        let content_id_str = identity.content_id.to_string();

        // Check for duplicates if dedup is enabled
        if self.config.dedup_enabled {
            let check_result = self.registry.check_duplicate(&identity);

            match check_result {
                DuplicateCheckResult::New => {
                    // New document - proceed with indexing
                    self.index_new_document(
                        content,
                        identity,
                        url,
                        title,
                        source_type,
                        source_id,
                        embedding_fn,
                    )
                }
                DuplicateCheckResult::ExactMatch { entry } => {
                    // Exact match - just update metadata
                    debug!("Exact match found for content_id: {}", content_id_str);

                    self.registry.update_metadata(
                        &entry.content_id,
                        url,
                        source_id,
                    );

                    Ok(ProcessingResult::MetadataUpdated {
                        content_id: content_id_str,
                    })
                }
                DuplicateCheckResult::NearDuplicate {
                    entry,
                    hamming_distance,
                } => {
                    if self.config.update_near_duplicates
                        && hamming_distance <= self.config.simhash_threshold
                    {
                        // Near-duplicate - update content
                        debug!(
                            "Near-duplicate found (distance: {}) for content_id: {}",
                            hamming_distance, content_id_str
                        );

                        self.update_document_content(
                            content,
                            identity,
                            entry,
                            url,
                            source_id,
                            embedding_fn,
                        )
                    } else {
                        // Too different, skip
                        Ok(ProcessingResult::Skipped {
                            content_id: content_id_str,
                            reason: format!(
                                "Near-duplicate (distance: {}) but too different to update",
                                hamming_distance
                            ),
                        })
                    }
                }
            }
        } else {
            // Dedup disabled - always index
            self.index_new_document(
                content,
                identity,
                url,
                title,
                source_type,
                source_id,
                embedding_fn,
            )
        }
    }

    /// Index a new document
    fn index_new_document<F>(
        &self,
        content: &str,
        identity: DocumentIdentity,
        url: Option<String>,
        title: Option<String>,
        source_type: &str,
        source_id: Option<(&str, &str)>,
        embedding_fn: F,
    ) -> Result<ProcessingResult>
    where
        F: Fn(&str) -> Embedding,
    {
        let content_id_str = identity.content_id.to_string();

        // Create chunks
        let doc_id = content_id_str.clone();
        let chunks = self.create_chunks(content, &doc_id, title.as_deref(), url.as_deref());

        if chunks.is_empty() {
            return Ok(ProcessingResult::Skipped {
                content_id: content_id_str,
                reason: "No chunks created".to_string(),
            });
        }

        // Generate embeddings and index
        let mut chunk_ids = Vec::with_capacity(chunks.len());
        let chunks_with_embeddings: Vec<(Chunk, Embedding)> = chunks
            .into_iter()
            .map(|chunk| {
                chunk_ids.push(chunk.metadata.chunk_id.clone());
                let embedding = embedding_fn(&chunk.content);
                (chunk, embedding)
            })
            .collect();

        let chunks_created = chunks_with_embeddings.len();

        // Index chunks
        self.indexer.index_batch(&chunks_with_embeddings)?;

        // Register in registry
        self.registry.register(
            identity,
            title,
            url,
            source_type,
            source_id,
            chunk_ids,
        );

        info!(
            "Indexed new document {} with {} chunks",
            content_id_str, chunks_created
        );

        Ok(ProcessingResult::Indexed {
            content_id: content_id_str,
            chunks_created,
        })
    }

    /// Update an existing document's content
    fn update_document_content<F>(
        &self,
        content: &str,
        new_identity: DocumentIdentity,
        existing_entry: crate::index::registry::DocumentEntry,
        url: Option<String>,
        source_id: Option<(&str, &str)>,
        embedding_fn: F,
    ) -> Result<ProcessingResult>
    where
        F: Fn(&str) -> Embedding,
    {
        let content_id_str = existing_entry.content_id.to_string();
        let old_chunk_ids = existing_entry.chunk_ids.clone();

        // Create new chunks
        let doc_id = content_id_str.clone();
        let chunks = self.create_chunks(
            content,
            &doc_id,
            existing_entry.title.as_deref(),
            url.as_deref(),
        );

        if chunks.is_empty() {
            return Ok(ProcessingResult::Skipped {
                content_id: content_id_str,
                reason: "No chunks created from updated content".to_string(),
            });
        }

        // Remove old chunks
        let chunks_removed = old_chunk_ids.len();
        for chunk_id in &old_chunk_ids {
            self.indexer.remove_chunk(chunk_id)?;
        }

        // Generate embeddings and index new chunks
        let mut new_chunk_ids = Vec::with_capacity(chunks.len());
        let chunks_with_embeddings: Vec<(Chunk, Embedding)> = chunks
            .into_iter()
            .map(|chunk| {
                new_chunk_ids.push(chunk.metadata.chunk_id.clone());
                let embedding = embedding_fn(&chunk.content);
                (chunk, embedding)
            })
            .collect();

        let chunks_created = chunks_with_embeddings.len();

        // Index new chunks
        self.indexer.index_batch(&chunks_with_embeddings)?;

        // Update registry
        self.registry.update_content(
            &existing_entry.content_id,
            new_identity,
            new_chunk_ids,
        );

        // Update metadata if provided
        self.registry.update_metadata(
            &existing_entry.content_id,
            url,
            source_id,
        );

        info!(
            "Updated document {}: removed {} chunks, created {} chunks",
            content_id_str, chunks_removed, chunks_created
        );

        Ok(ProcessingResult::ContentUpdated {
            content_id: content_id_str,
            chunks_created,
            chunks_removed,
        })
    }

    /// Create chunks from content
    fn create_chunks(
        &self,
        content: &str,
        doc_id: &str,
        title: Option<&str>,
        url: Option<&str>,
    ) -> Vec<Chunk> {
        // Create a temporary document for splitting
        let mut doc = crate::types::Document::new(content).with_id(doc_id);
        if let Some(t) = title {
            doc = doc.with_title(t);
        }
        if let Some(u) = url {
            doc = doc.with_url(u);
        }

        self.splitter.split_document(&doc)
    }

    /// Save the registry to disk
    pub fn save(&self) -> Result<()> {
        self.registry.save()?;
        self.indexer.save()?;
        Ok(())
    }

    /// Get the underlying registry
    pub fn registry(&self) -> &Arc<DocumentRegistry> {
        &self.registry
    }

    /// Get the underlying indexer
    pub fn indexer(&self) -> &Arc<HybridIndexer> {
        &self.indexer
    }

    /// Get processing statistics
    pub fn stats(&self) -> ProcessorStats {
        let registry_stats = self.registry.stats();
        ProcessorStats {
            total_documents: registry_stats.total_documents,
            total_chunks: registry_stats.total_chunks,
            total_urls: registry_stats.total_urls,
            source_counts: registry_stats.source_counts,
        }
    }
}

/// Processor statistics
#[derive(Debug, Clone)]
pub struct ProcessorStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub total_urls: usize,
    pub source_counts: std::collections::HashMap<String, usize>,
}

/// Builder for DocumentProcessor
pub struct DocumentProcessorBuilder {
    registry: Option<Arc<DocumentRegistry>>,
    indexer: Option<Arc<HybridIndexer>>,
    splitter: Option<TextSplitter>,
    config: ProcessorConfig,
}

impl DocumentProcessorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            registry: None,
            indexer: None,
            splitter: None,
            config: ProcessorConfig::default(),
        }
    }

    /// Set the document registry
    pub fn with_registry(mut self, registry: Arc<DocumentRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Set the hybrid indexer
    pub fn with_indexer(mut self, indexer: Arc<HybridIndexer>) -> Self {
        self.indexer = Some(indexer);
        self
    }

    /// Set the text splitter
    pub fn with_splitter(mut self, splitter: TextSplitter) -> Self {
        self.splitter = Some(splitter);
        self
    }

    /// Set minimum content length
    pub fn with_min_content_length(mut self, min_length: usize) -> Self {
        self.config.min_content_length = min_length;
        self
    }

    /// Enable/disable deduplication
    pub fn with_dedup(mut self, enabled: bool) -> Self {
        self.config.dedup_enabled = enabled;
        self
    }

    /// Set SimHash threshold for near-duplicates
    pub fn with_simhash_threshold(mut self, threshold: u32) -> Self {
        self.config.simhash_threshold = threshold;
        self
    }

    /// Build the processor
    pub fn build(self) -> Result<DocumentProcessor> {
        let registry = self
            .registry
            .ok_or_else(|| anyhow::anyhow!("Registry is required"))?;
        let indexer = self
            .indexer
            .ok_or_else(|| anyhow::anyhow!("Indexer is required"))?;
        let splitter = self
            .splitter
            .ok_or_else(|| anyhow::anyhow!("Splitter is required"))?;

        Ok(DocumentProcessor::new(registry, indexer, splitter, self.config))
    }
}

impl Default for DocumentProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ChunkingConfig, IndexConfig};
    use crate::index::{ChunkStorage, VectorIndex};
    use crate::retrieval::Bm25Index;
    use tempfile::TempDir;

    fn create_test_embedding(text: &str, dims: usize) -> Embedding {
        let hash = xxhash_rust::xxh3::xxh3_64(text.as_bytes());
        (0..dims)
            .map(|i| {
                let h = hash.wrapping_add(i as u64);
                ((h % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    fn create_test_processor(temp_dir: &TempDir) -> DocumentProcessor {
        let registry = Arc::new(
            DocumentRegistry::new(temp_dir.path(), 3).unwrap(),
        );

        let vector_index = Arc::new(
            VectorIndex::new(64, &IndexConfig::default()).unwrap(),
        );
        let bm25_index = Arc::new(Bm25Index::new_in_memory().unwrap());
        let chunk_storage = Arc::new(
            ChunkStorage::new(temp_dir.path().join("chunks")).unwrap(),
        );

        let indexer = Arc::new(HybridIndexer::new(
            vector_index,
            bm25_index,
            chunk_storage,
        ));

        let splitter = TextSplitter::new(ChunkingConfig {
            chunk_size: 100,
            overlap_fraction: 0.1,
            min_chunk_size: 10,
            max_chunk_size: 200,
        });

        let config = ProcessorConfig {
            min_content_length: 20,
            dedup_enabled: true,
            simhash_threshold: 3,
            update_near_duplicates: true,
        };

        DocumentProcessor::new(registry, indexer, splitter, config)
    }

    #[test]
    fn test_process_new_document() {
        let temp_dir = TempDir::new().unwrap();
        let processor = create_test_processor(&temp_dir);

        let content = "This is a test document with enough content to pass the minimum length requirement. It contains multiple sentences for testing purposes.";

        let result = processor
            .process(
                content,
                Some("https://example.com/test".to_string()),
                Some("Test Document".to_string()),
                "test",
                Some(("test_id", "123")),
                |text| create_test_embedding(text, 64),
            )
            .unwrap();

        match result {
            ProcessingResult::Indexed { chunks_created, .. } => {
                assert!(chunks_created > 0);
            }
            _ => panic!("Expected Indexed result"),
        }

        // Check registry
        assert_eq!(processor.registry().len(), 1);
    }

    #[test]
    fn test_process_exact_duplicate() {
        let temp_dir = TempDir::new().unwrap();
        let processor = create_test_processor(&temp_dir);

        let content = "This is a test document that will be indexed twice to test exact duplicate detection behavior.";

        // First indexing
        let result1 = processor
            .process(
                content,
                Some("https://example.com/first".to_string()),
                Some("First".to_string()),
                "test",
                None,
                |text| create_test_embedding(text, 64),
            )
            .unwrap();

        assert!(matches!(result1, ProcessingResult::Indexed { .. }));

        // Second indexing with same content
        let result2 = processor
            .process(
                content,
                Some("https://example.com/second".to_string()),
                Some("Second".to_string()),
                "test",
                None,
                |text| create_test_embedding(text, 64),
            )
            .unwrap();

        assert!(matches!(result2, ProcessingResult::MetadataUpdated { .. }));

        // Still only one document in registry
        assert_eq!(processor.registry().len(), 1);
    }

    #[test]
    fn test_process_content_too_short() {
        let temp_dir = TempDir::new().unwrap();
        let processor = create_test_processor(&temp_dir);

        let content = "Too short";

        let result = processor
            .process(
                content,
                None,
                None,
                "test",
                None,
                |text| create_test_embedding(text, 64),
            )
            .unwrap();

        match result {
            ProcessingResult::Skipped { reason, .. } => {
                assert!(reason.contains("too short"));
            }
            _ => panic!("Expected Skipped result"),
        }
    }

    #[test]
    fn test_process_with_dedup_disabled() {
        let temp_dir = TempDir::new().unwrap();

        let registry = Arc::new(
            DocumentRegistry::new(temp_dir.path(), 3).unwrap(),
        );
        let vector_index = Arc::new(
            VectorIndex::new(64, &IndexConfig::default()).unwrap(),
        );
        let bm25_index = Arc::new(Bm25Index::new_in_memory().unwrap());
        let chunk_storage = Arc::new(
            ChunkStorage::new(temp_dir.path().join("chunks")).unwrap(),
        );
        let indexer = Arc::new(HybridIndexer::new(
            vector_index,
            bm25_index,
            chunk_storage,
        ));
        let splitter = TextSplitter::new(ChunkingConfig::default());

        let config = ProcessorConfig {
            min_content_length: 20,
            dedup_enabled: false, // Disabled
            simhash_threshold: 3,
            update_near_duplicates: true,
        };

        let processor = DocumentProcessor::new(registry, indexer, splitter, config);

        let content = "This is a test document with enough content for the test to work properly.";

        // Index twice with dedup disabled
        let result1 = processor
            .process(content, None, None, "test", None, |text| {
                create_test_embedding(text, 64)
            })
            .unwrap();

        let result2 = processor
            .process(content, None, None, "test", None, |text| {
                create_test_embedding(text, 64)
            })
            .unwrap();

        // Both should be indexed (dedup disabled)
        assert!(matches!(result1, ProcessingResult::Indexed { .. }));
        assert!(matches!(result2, ProcessingResult::Indexed { .. }));
    }

    #[test]
    fn test_builder() {
        let temp_dir = TempDir::new().unwrap();

        let registry = Arc::new(
            DocumentRegistry::new(temp_dir.path(), 3).unwrap(),
        );
        let vector_index = Arc::new(
            VectorIndex::new(64, &IndexConfig::default()).unwrap(),
        );
        let bm25_index = Arc::new(Bm25Index::new_in_memory().unwrap());
        let chunk_storage = Arc::new(
            ChunkStorage::new(temp_dir.path().join("chunks")).unwrap(),
        );
        let indexer = Arc::new(HybridIndexer::new(
            vector_index,
            bm25_index,
            chunk_storage,
        ));
        let splitter = TextSplitter::new(ChunkingConfig::default());

        let processor = DocumentProcessorBuilder::new()
            .with_registry(registry)
            .with_indexer(indexer)
            .with_splitter(splitter)
            .with_min_content_length(50)
            .with_dedup(true)
            .with_simhash_threshold(5)
            .build()
            .unwrap();

        assert_eq!(processor.config.min_content_length, 50);
        assert_eq!(processor.config.simhash_threshold, 5);
    }

    #[test]
    fn test_process_near_duplicate() {
        let temp_dir = TempDir::new().unwrap();
        let processor = create_test_processor(&temp_dir);

        // Index original document
        let content1 = "This is a test document with enough content to pass the minimum length requirement for processing.";
        let result1 = processor
            .process(
                content1,
                Some("https://example.com/v1".to_string()),
                Some("Original Doc".to_string()),
                "test",
                None,
                |text| create_test_embedding(text, 64),
            )
            .unwrap();

        assert!(matches!(result1, ProcessingResult::Indexed { .. }));

        // Index near-duplicate: slightly modified content that produces
        // the same SimHash content_id but a different SHA256 content_hash.
        // Since DocumentIdentity normalizes (lowercases, collapses whitespace),
        // we need a change that affects the SHA256 but keeps the SimHash close.
        let content2 = "This is a test document with enough content to pass the minimum length requirement for processing today.";

        let id1 = crate::types::DocumentIdentity::compute(content1);
        let id2 = crate::types::DocumentIdentity::compute(content2);

        // If they happen to share the same content_id (same simhash), the registry
        // will detect a NearDuplicate (same content_id key, different content_hash).
        // If the simhash differs, the near-dup detection uses LSH buckets.
        // Either way, the processor should handle it gracefully.
        if id1.content_id == id2.content_id {
            // Same content_id, different hash -> NearDuplicate with distance 0
            let result2 = processor
                .process(
                    content2,
                    Some("https://example.com/v2".to_string()),
                    Some("Updated Doc".to_string()),
                    "test",
                    None,
                    |text| create_test_embedding(text, 64),
                )
                .unwrap();

            assert!(
                matches!(result2, ProcessingResult::ContentUpdated { .. }),
                "Expected ContentUpdated for near-duplicate with same content_id, got {:?}",
                result2
            );
        } else {
            // Different content_id but possibly within hamming distance threshold
            let distance = id1.hamming_distance(&id2);
            let result2 = processor
                .process(
                    content2,
                    Some("https://example.com/v2".to_string()),
                    Some("Updated Doc".to_string()),
                    "test",
                    None,
                    |text| create_test_embedding(text, 64),
                )
                .unwrap();

            if distance <= 3 {
                // Within threshold -> ContentUpdated
                assert!(
                    matches!(result2, ProcessingResult::ContentUpdated { .. }),
                    "Expected ContentUpdated for near-duplicate within threshold, got {:?}",
                    result2
                );
            } else {
                // Could be Indexed (new) or Skipped (too different)
                assert!(
                    matches!(
                        result2,
                        ProcessingResult::Indexed { .. } | ProcessingResult::Skipped { .. }
                    ),
                    "Expected Indexed or Skipped for content beyond threshold, got {:?}",
                    result2
                );
            }
        }
    }

    #[test]
    fn test_save_persists_state() {
        let temp_dir = TempDir::new().unwrap();
        let processor = create_test_processor(&temp_dir);

        let content = "This is a document that should be persisted across save and reload cycles.";
        processor
            .process(
                content,
                Some("https://example.com/persist".to_string()),
                Some("Persist Test".to_string()),
                "test",
                None,
                |text| create_test_embedding(text, 64),
            )
            .unwrap();

        assert_eq!(processor.registry().len(), 1);

        // Save should succeed without error
        processor.save().unwrap();

        // Verify registry was saved by reloading
        let reloaded_registry =
            DocumentRegistry::load(temp_dir.path(), 3).unwrap();
        assert_eq!(reloaded_registry.len(), 1);
    }

    #[test]
    fn test_stats_returns_correct_counts() {
        let temp_dir = TempDir::new().unwrap();
        let processor = create_test_processor(&temp_dir);

        // Initially empty
        let stats = processor.stats();
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.total_chunks, 0);
        assert_eq!(stats.total_urls, 0);
        assert!(stats.source_counts.is_empty());

        // Add a document
        let content = "This is a test document with enough content for the processor to create chunks from it.";
        processor
            .process(
                content,
                Some("https://example.com/stats".to_string()),
                Some("Stats Test".to_string()),
                "web",
                None,
                |text| create_test_embedding(text, 64),
            )
            .unwrap();

        let stats = processor.stats();
        assert_eq!(stats.total_documents, 1);
        assert!(stats.total_chunks > 0);
        assert_eq!(stats.total_urls, 1);
        assert_eq!(stats.source_counts.get("web"), Some(&1));
    }

    #[test]
    fn test_processing_result_content_id() {
        let indexed = ProcessingResult::Indexed {
            content_id: "abc123".to_string(),
            chunks_created: 5,
        };
        assert_eq!(indexed.content_id(), "abc123");

        let updated = ProcessingResult::MetadataUpdated {
            content_id: "def456".to_string(),
        };
        assert_eq!(updated.content_id(), "def456");

        let content_updated = ProcessingResult::ContentUpdated {
            content_id: "ghi789".to_string(),
            chunks_created: 3,
            chunks_removed: 2,
        };
        assert_eq!(content_updated.content_id(), "ghi789");

        let skipped = ProcessingResult::Skipped {
            content_id: "jkl012".to_string(),
            reason: "too short".to_string(),
        };
        assert_eq!(skipped.content_id(), "jkl012");
    }

    #[test]
    fn test_processing_result_was_indexed() {
        let indexed = ProcessingResult::Indexed {
            content_id: "a".to_string(),
            chunks_created: 1,
        };
        assert!(indexed.was_indexed());

        let content_updated = ProcessingResult::ContentUpdated {
            content_id: "b".to_string(),
            chunks_created: 2,
            chunks_removed: 1,
        };
        assert!(content_updated.was_indexed());

        let metadata_updated = ProcessingResult::MetadataUpdated {
            content_id: "c".to_string(),
        };
        assert!(!metadata_updated.was_indexed());

        let skipped = ProcessingResult::Skipped {
            content_id: "d".to_string(),
            reason: "test".to_string(),
        };
        assert!(!skipped.was_indexed());
    }

    #[test]
    fn test_processing_result_chunks_created() {
        let indexed = ProcessingResult::Indexed {
            content_id: "a".to_string(),
            chunks_created: 5,
        };
        assert_eq!(indexed.chunks_created(), 5);

        let content_updated = ProcessingResult::ContentUpdated {
            content_id: "b".to_string(),
            chunks_created: 3,
            chunks_removed: 2,
        };
        assert_eq!(content_updated.chunks_created(), 3);

        let metadata_updated = ProcessingResult::MetadataUpdated {
            content_id: "c".to_string(),
        };
        assert_eq!(metadata_updated.chunks_created(), 0);

        let skipped = ProcessingResult::Skipped {
            content_id: "d".to_string(),
            reason: "test".to_string(),
        };
        assert_eq!(skipped.chunks_created(), 0);
    }
}
