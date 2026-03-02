//! IndexStack — unified factory for the three index components
//!
//! Reduces boilerplate by encapsulating the common pattern of creating/loading
//! VectorIndex + Bm25Index + ChunkStorage together.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;

use crate::config::{Config, IndexConfig, RetrievalConfig};
use crate::embedding::EmbeddingEngine;
use crate::retrieval::{Bm25Index, HybridIndexer, HybridRetriever};

use super::{ChunkStorage, VectorIndex};

/// Unified container for the three index components.
pub struct IndexStack {
    pub vector_index: Arc<VectorIndex>,
    pub bm25_index: Arc<Bm25Index>,
    pub chunk_storage: Arc<ChunkStorage>,
}

impl IndexStack {
    /// Open existing indexes (load vector index if present, otherwise create new).
    /// This is the read-oriented path used by search and the daemon.
    pub fn open(config: &Config) -> Result<Self> {
        let data_dir = &config.node.data_dir;
        std::fs::create_dir_all(data_dir)?;

        let index_path = data_dir.join("vector.index");
        let vector_index = if index_path.exists() {
            info!("Loading existing vector index");
            Arc::new(VectorIndex::load(&index_path, &config.index)?)
        } else {
            info!("Creating new vector index");
            Arc::new(VectorIndex::new(config.embedding.dimensions, &config.index)?)
        };

        let bm25_path = data_dir.join("bm25");
        let bm25_index = Arc::new(Bm25Index::new(&bm25_path)?);
        info!("BM25 index loaded");

        let chunk_storage = Arc::new(ChunkStorage::load(data_dir)?);
        info!("Chunk storage loaded with {} chunks", chunk_storage.len());

        Ok(Self {
            vector_index,
            bm25_index,
            chunk_storage,
        })
    }

    /// Open existing indexes in read-only mode (no BM25 write lock acquired).
    ///
    /// Use this for the CLI search fallback when the daemon may already hold the
    /// BM25 write lock. If the BM25 directory doesn't exist, falls back to an
    /// empty in-memory index (no lock conflict, returns empty BM25 results).
    pub fn open_read_only(config: &Config) -> Result<Self> {
        let data_dir = &config.node.data_dir;
        std::fs::create_dir_all(data_dir)?;

        let index_path = data_dir.join("vector.index");
        let vector_index = if index_path.exists() {
            info!("Loading existing vector index");
            Arc::new(VectorIndex::load(&index_path, &config.index)?)
        } else {
            info!("Creating new vector index");
            Arc::new(VectorIndex::new(config.embedding.dimensions, &config.index)?)
        };

        let bm25_path = data_dir.join("bm25");
        let bm25_index = if bm25_path.exists() {
            Arc::new(
                Bm25Index::open_read_only(&bm25_path)
                    .context("Failed to open BM25 index in read-only mode")?,
            )
        } else {
            info!("BM25 directory not found, using empty in-memory index");
            Arc::new(Bm25Index::new_in_memory()?)
        };
        info!("BM25 index loaded (read-only)");

        let chunk_storage = match ChunkStorage::load(data_dir) {
            Ok(cs) => {
                info!("Chunk storage loaded with {} chunks", cs.len());
                Arc::new(cs)
            }
            Err(e) => {
                info!("Chunk storage unavailable ({}), using empty in-memory store", e);
                Arc::new(ChunkStorage::new_in_memory()?)
            }
        };

        Ok(Self {
            vector_index,
            bm25_index,
            chunk_storage,
        })
    }

    /// Create fresh indexes (always new, never loads existing vector index).
    /// This is the write-oriented path used by index, scrape, and import commands.
    pub fn create(data_dir: &Path, dimensions: usize, index_config: &IndexConfig) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;

        let vector_index = Arc::new(
            VectorIndex::new(dimensions, index_config)
                .context("Failed to create vector index")?,
        );

        let bm25_path = data_dir.join("bm25");
        let bm25_index = Arc::new(
            Bm25Index::new(&bm25_path)
                .context("Failed to create BM25 index")?,
        );

        let chunk_storage = Arc::new(
            ChunkStorage::new(data_dir)
                .context("Failed to create chunk storage")?,
        );

        Ok(Self {
            vector_index,
            bm25_index,
            chunk_storage,
        })
    }

    /// Build a HybridIndexer for write operations.
    pub fn indexer(&self) -> HybridIndexer {
        HybridIndexer::new(
            self.vector_index.clone(),
            self.bm25_index.clone(),
            self.chunk_storage.clone(),
        )
    }

    /// Build a HybridRetriever for search operations.
    pub fn retriever(
        &self,
        embedding_engine: Option<Arc<EmbeddingEngine>>,
        config: RetrievalConfig,
    ) -> HybridRetriever {
        HybridRetriever::new(
            self.vector_index.clone(),
            self.bm25_index.clone(),
            self.chunk_storage.clone(),
            embedding_engine,
            config,
        )
    }

    /// Save the vector index to disk.
    pub fn save_vector_index(&self, data_dir: &Path) -> Result<()> {
        let index_path = data_dir.join("vector.index");
        self.vector_index.save(&index_path)
    }
}
