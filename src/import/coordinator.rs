//! Import coordinator that orchestrates the bulk import process

use super::progress::ImportProgress;
use super::source::{DumpDocument, DumpSource, ImportCheckpoint, ImportConfig, ImportError, ImportStats};
use crate::chunking::TextSplitter;
use crate::config::{ChunkingConfig, DedupConfig, IndexConfig};
use crate::index::{
    ChunkStorage, DocumentProcessor, DocumentRegistry, ProcessingResult, ProcessorConfig,
    VectorIndex,
};
use crate::retrieval::{Bm25Index, HybridIndexer};
use crate::types::Embedding;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

/// Import coordinator for bulk imports
pub struct ImportCoordinator {
    /// Import configuration
    config: ImportConfig,
    /// Document processor for unified ingestion
    processor: DocumentProcessor,
    /// Vector index (for saving)
    vector_index: Arc<VectorIndex>,
    /// Embedding dimensions
    embedding_dims: usize,
    /// Data directory for saving
    data_dir: PathBuf,
    /// Quiet mode
    quiet: bool,
}

impl ImportCoordinator {
    /// Create a new import coordinator
    pub fn new(
        config: ImportConfig,
        data_dir: impl AsRef<Path>,
        chunking_config: ChunkingConfig,
        index_config: IndexConfig,
        dedup_config: DedupConfig,
        embedding_dims: usize,
    ) -> Result<Self, ImportError> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir).map_err(ImportError::Io)?;

        // Initialize document registry
        let registry = Arc::new(
            DocumentRegistry::load(&data_dir, dedup_config.simhash_distance_threshold)
                .map_err(|e| ImportError::Index(e.to_string()))?,
        );

        // Initialize index components
        let vector_index = Arc::new(
            VectorIndex::new(embedding_dims, &index_config)
                .map_err(|e| ImportError::Index(e.to_string()))?,
        );

        let bm25_path = data_dir.join("bm25");
        let bm25_index = Arc::new(
            Bm25Index::new(&bm25_path).map_err(|e| ImportError::Index(e.to_string()))?,
        );

        let chunk_storage = Arc::new(
            ChunkStorage::new(&data_dir).map_err(|e| ImportError::Index(e.to_string()))?,
        );

        let indexer = Arc::new(HybridIndexer::new(
            vector_index.clone(),
            bm25_index,
            chunk_storage,
        ));

        let splitter = TextSplitter::new(chunking_config);

        // Create processor config from dedup config
        let processor_config = ProcessorConfig {
            min_content_length: config.min_content_length,
            dedup_enabled: config.deduplicate && dedup_config.enabled,
            simhash_threshold: dedup_config.simhash_distance_threshold,
            update_near_duplicates: dedup_config.update_near_duplicates,
        };

        let processor = DocumentProcessor::new(registry, indexer, splitter, processor_config);

        Ok(Self {
            config,
            processor,
            vector_index,
            embedding_dims,
            data_dir,
            quiet: false,
        })
    }

    /// Set quiet mode (no progress output)
    pub fn with_quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    /// Run import from a dump source
    pub fn import<S: DumpSource>(&mut self, mut source: S) -> Result<ImportStats, ImportError> {
        let source_name = source.source_name().to_string();
        info!("Starting import from: {}", source_name);

        // Create progress tracker
        let progress = ImportProgress::new(
            PathBuf::from(&source_name),
            source.document_count_hint(),
            self.config.checkpoint_interval,
            self.config.checkpoint_path.clone(),
            self.quiet,
        );

        let mut docs_count = 0;

        for doc_result in source.iter_documents() {
            // Check for cancellation
            if progress.is_cancelled() {
                info!("Import cancelled");
                break;
            }

            // Check max documents limit
            if let Some(max) = self.config.max_documents {
                if docs_count >= max {
                    info!("Reached max documents limit: {}", max);
                    break;
                }
            }

            match doc_result {
                Ok(doc) => {
                    let title = doc.title.clone();
                    let doc_size = doc.content.len() as u64;

                    // Process document using the unified processor
                    let embedding_dims = self.embedding_dims;
                    let result = self.processor.process(
                        &doc.content,
                        doc.url.clone(),
                        Some(title.clone()),
                        "wikipedia",
                        Some(("wikipedia_id", doc.id.as_str())),
                        |text| Self::generate_embedding(text, embedding_dims),
                    );

                    match result {
                        Ok(ProcessingResult::Indexed { chunks_created, .. }) => {
                            progress.document_processed(&title, true, chunks_created, doc_size);
                            docs_count += 1;
                        }
                        Ok(ProcessingResult::MetadataUpdated { .. }) => {
                            // Document already exists, metadata updated
                            progress.document_processed(&title, false, 0, doc_size);
                        }
                        Ok(ProcessingResult::ContentUpdated { chunks_created, .. }) => {
                            // Document content updated
                            progress.document_processed(&title, true, chunks_created, doc_size);
                            docs_count += 1;
                        }
                        Ok(ProcessingResult::Skipped { reason, .. }) => {
                            tracing::debug!("Skipped document '{}': {}", title, reason);
                            progress.document_processed(&title, false, 0, doc_size);
                        }
                        Err(e) => {
                            warn!("Error processing document '{}': {}", title, e);
                            progress.document_error(&e.to_string());
                        }
                    }
                }
                Err(e) => {
                    warn!("Error reading document: {}", e);
                    progress.document_error(&e.to_string());
                }
            }
        }

        // Save all data
        self.save()?;

        progress.finish();

        if !self.quiet {
            progress.print_summary();
        }

        Ok(progress.get_stats())
    }

    /// Resume import from a checkpoint
    pub fn resume<S: DumpSource>(
        &mut self,
        mut source: S,
        checkpoint: &ImportCheckpoint,
    ) -> Result<ImportStats, ImportError> {
        info!(
            "Resuming import from checkpoint: {} documents, {} bytes",
            checkpoint.documents_processed, checkpoint.byte_position
        );

        // Seek source to checkpoint position
        source.seek_to(checkpoint.byte_position)?;

        // Run normal import (progress will start from checkpoint position)
        self.import(source)
    }

    /// Generate embedding for text (placeholder - uses hash-based pseudo-embedding)
    fn generate_embedding(text: &str, dims: usize) -> Embedding {
        // In production, this would use the real embedding engine
        // For now, create deterministic pseudo-embeddings based on content hash
        let hash = xxhash_rust::xxh3::xxh3_64(text.as_bytes());

        (0..dims)
            .map(|i| {
                let h = hash.wrapping_add(i as u64);
                // Use hash to generate pseudo-random values in [-1, 1]
                ((h % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    /// Save all indices to disk
    pub fn save(&self) -> Result<(), ImportError> {
        let index_path = self.data_dir.join("vector.index");
        self.vector_index
            .save(&index_path)
            .map_err(|e| ImportError::Index(e.to_string()))?;

        self.processor
            .save()
            .map_err(|e| ImportError::Index(e.to_string()))?;

        info!("Saved indices to {}", self.data_dir.display());
        Ok(())
    }

    /// Get access to the document processor
    pub fn processor(&self) -> &DocumentProcessor {
        &self.processor
    }
}

/// Builder for ImportCoordinator with sensible defaults
pub struct ImportCoordinatorBuilder {
    config: ImportConfig,
    data_dir: PathBuf,
    chunking_config: ChunkingConfig,
    index_config: IndexConfig,
    dedup_config: DedupConfig,
    embedding_dims: usize,
    quiet: bool,
}

impl ImportCoordinatorBuilder {
    /// Create a new builder with default settings
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self {
            config: ImportConfig::default(),
            data_dir: data_dir.as_ref().to_path_buf(),
            chunking_config: ChunkingConfig::default(),
            index_config: IndexConfig::default(),
            dedup_config: DedupConfig::default(),
            embedding_dims: 768,
            quiet: false,
        }
    }

    /// Set import configuration
    pub fn with_config(mut self, config: ImportConfig) -> Self {
        self.config = config;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Enable/disable deduplication
    pub fn with_dedup(mut self, deduplicate: bool) -> Self {
        self.config.deduplicate = deduplicate;
        self
    }

    /// Set minimum content length
    pub fn with_min_content_length(mut self, min_length: usize) -> Self {
        self.config.min_content_length = min_length;
        self
    }

    /// Set maximum documents to import
    pub fn with_max_documents(mut self, max_docs: Option<usize>) -> Self {
        self.config.max_documents = max_docs;
        self
    }

    /// Set checkpoint path
    pub fn with_checkpoint(mut self, path: impl AsRef<Path>) -> Self {
        self.config.checkpoint_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set embedding dimensions
    pub fn with_embedding_dims(mut self, dims: usize) -> Self {
        self.embedding_dims = dims;
        self
    }

    /// Set chunking configuration
    pub fn with_chunking_config(mut self, config: ChunkingConfig) -> Self {
        self.chunking_config = config;
        self
    }

    /// Set index configuration
    pub fn with_index_config(mut self, config: IndexConfig) -> Self {
        self.index_config = config;
        self
    }

    /// Set dedup configuration
    pub fn with_dedup_config(mut self, config: DedupConfig) -> Self {
        self.dedup_config = config;
        self
    }

    /// Set quiet mode
    pub fn with_quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    /// Build the coordinator
    pub fn build(self) -> Result<ImportCoordinator, ImportError> {
        ImportCoordinator::new(
            self.config,
            self.data_dir,
            self.chunking_config,
            self.index_config,
            self.dedup_config,
            self.embedding_dims,
        )
        .map(|c| c.with_quiet(self.quiet))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_coordinator_builder() {
        let temp_dir = TempDir::new().unwrap();

        let coordinator = ImportCoordinatorBuilder::new(temp_dir.path())
            .with_batch_size(50)
            .with_dedup(true)
            .with_min_content_length(100)
            .with_max_documents(Some(1000))
            .with_embedding_dims(256)
            .with_quiet(true)
            .build()
            .unwrap();

        // Coordinator was created successfully
        assert!(coordinator.config.deduplicate);
        assert_eq!(coordinator.config.batch_size, 50);
        assert_eq!(coordinator.config.min_content_length, 100);
        assert_eq!(coordinator.config.max_documents, Some(1000));
    }

    #[test]
    fn test_generate_embedding() {
        let embedding1 = ImportCoordinator::generate_embedding("test content", 64);
        let embedding2 = ImportCoordinator::generate_embedding("test content", 64);
        let embedding3 = ImportCoordinator::generate_embedding("different content", 64);

        // Same content should produce same embedding
        assert_eq!(embedding1, embedding2);

        // Different content should produce different embedding
        assert_ne!(embedding1, embedding3);

        // Correct dimensions
        assert_eq!(embedding1.len(), 64);
    }
}
