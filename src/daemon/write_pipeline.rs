//! Write Pipeline
//!
//! Handles batched writes and periodic commits for the daemon.
//! Provides a single writer that serializes all index modifications.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::embedding::{generate_with_fallback, EmbeddingEngine};
use crate::types::Chunk;

use super::index_manager::IndexManager;

/// Items that can be ingested by the write pipeline
pub enum IngestItem {
    /// A chunk to be indexed
    Chunk {
        stream_id: Uuid,
        chunk: Chunk,
        embedding: Option<Vec<f32>>,
    },
    /// Request to commit and flush
    Commit {
        stream_id: Uuid,
        respond_to: oneshot::Sender<Result<()>>,
    },
}

/// Write pipeline for batched indexing and periodic commits
pub struct WritePipeline {
    ingest_tx: mpsc::Sender<IngestItem>,
    batch_size: usize,
}

impl WritePipeline {
    /// Create and start the write pipeline with a shutdown receiver
    pub fn start(
        index_manager: Arc<IndexManager>,
        embedding_engine: Option<Arc<EmbeddingEngine>>,
        batch_size: usize,
        commit_interval: Duration,
        shutdown: broadcast::Receiver<()>,
    ) -> Self {
        let (ingest_tx, ingest_rx) = mpsc::channel(1000);

        // Spawn the background worker
        let worker = WritePipelineWorker::new(
            index_manager,
            embedding_engine,
            ingest_rx,
            batch_size,
            commit_interval,
        );

        tokio::spawn(async move {
            worker.run(shutdown).await;
        });

        Self {
            ingest_tx,
            batch_size,
        }
    }

    /// Ingest an item into the write pipeline
    pub async fn ingest(&self, item: IngestItem) -> Result<()> {
        self.ingest_tx
            .send(item)
            .await
            .map_err(|_| anyhow::anyhow!("Write pipeline closed"))?;
        Ok(())
    }

    /// Get the configured batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Background worker that processes the write queue
struct WritePipelineWorker {
    index_manager: Arc<IndexManager>,
    embedding_engine: Option<Arc<EmbeddingEngine>>,
    ingest_rx: mpsc::Receiver<IngestItem>,
    batch_size: usize,
    commit_interval: Duration,
}

impl WritePipelineWorker {
    fn new(
        index_manager: Arc<IndexManager>,
        embedding_engine: Option<Arc<EmbeddingEngine>>,
        ingest_rx: mpsc::Receiver<IngestItem>,
        batch_size: usize,
        commit_interval: Duration,
    ) -> Self {
        Self {
            index_manager,
            embedding_engine,
            ingest_rx,
            batch_size,
            commit_interval,
        }
    }

    async fn run(mut self, mut shutdown: broadcast::Receiver<()>) {
        info!(
            "Write pipeline started (batch_size={}, commit_interval={:?})",
            self.batch_size, self.commit_interval
        );

        let mut batch: Vec<(Chunk, Vec<f32>)> = Vec::with_capacity(self.batch_size);
        let mut interval = tokio::time::interval(self.commit_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                // Handle incoming items
                Some(item) = self.ingest_rx.recv() => {
                    match item {
                        IngestItem::Chunk { chunk, embedding, stream_id } => {
                            debug!("Received chunk for stream {}", stream_id);

                            // Generate embedding if not provided
                            let embedding = embedding.unwrap_or_else(|| {
                                self.generate_embedding(&chunk.content)
                            });

                            batch.push((chunk, embedding));

                            // Flush if batch is full
                            if batch.len() >= self.batch_size {
                                self.flush_batch(&mut batch).await;
                            }
                        }
                        IngestItem::Commit { respond_to, stream_id } => {
                            debug!("Commit requested for stream {}", stream_id);

                            // Flush any pending chunks first
                            if !batch.is_empty() {
                                self.flush_batch(&mut batch).await;
                            }

                            // Commit to disk
                            let result = self.index_manager.commit();
                            let _ = respond_to.send(result);
                        }
                    }
                }

                // Periodic commit
                _ = interval.tick() => {
                    if !batch.is_empty() {
                        debug!("Periodic flush of {} chunks", batch.len());
                        self.flush_batch(&mut batch).await;
                        if let Err(e) = self.index_manager.commit() {
                            warn!("Periodic commit failed: {}", e);
                        }
                    }
                }

                // Shutdown signal
                _ = shutdown.recv() => {
                    info!("Write pipeline shutting down");

                    // Final flush
                    if !batch.is_empty() {
                        self.flush_batch(&mut batch).await;
                    }

                    // Final commit
                    if let Err(e) = self.index_manager.commit() {
                        error!("Final commit failed: {}", e);
                    }

                    break;
                }
            }
        }

        info!("Write pipeline stopped");
    }

    async fn flush_batch(&self, batch: &mut Vec<(Chunk, Vec<f32>)>) {
        if batch.is_empty() {
            return;
        }

        debug!("Flushing batch of {} chunks", batch.len());

        match self.index_manager.index_batch(batch) {
            Ok(keys) => {
                debug!("Indexed {} chunks with keys {:?}", keys.len(), &keys[..keys.len().min(3)]);
            }
            Err(e) => {
                error!("Failed to index batch: {}", e);
            }
        }

        batch.clear();
    }

    /// Generate embedding for content using real embedding engine
    fn generate_embedding(&self, content: &str) -> Vec<f32> {
        generate_with_fallback(
            self.embedding_engine.as_deref(),
            content,
            self.index_manager.dimensions(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::types::ChunkMetadata;
    use tempfile::TempDir;

    fn test_config(data_dir: &std::path::Path) -> Config {
        let mut config = Config::default();
        config.node.data_dir = data_dir.to_path_buf();
        config
    }

    #[tokio::test]
    async fn test_write_pipeline_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(temp_dir.path());
        let index_manager = Arc::new(IndexManager::load(&config).unwrap());

        let (_shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let pipeline = WritePipeline::start(
            index_manager,
            None,
            100,
            Duration::from_secs(60),
            shutdown_rx,
        );
        assert_eq!(pipeline.batch_size(), 100);
    }

    #[tokio::test]
    async fn test_write_pipeline_ingest_and_commit() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(temp_dir.path());
        let index_manager = Arc::new(IndexManager::load(&config).unwrap());

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let pipeline = WritePipeline::start(
            index_manager.clone(),
            None, // No embedding engine in test
            10,
            Duration::from_secs(60),
            shutdown_rx,
        );

        // Ingest a chunk
        let chunk = Chunk {
            metadata: ChunkMetadata::new("test-chunk".to_string(), "test-doc".to_string()),
            content: "Test content".to_string(),
            token_count: 2,
        };

        pipeline
            .ingest(IngestItem::Chunk {
                stream_id: Uuid::new_v4(),
                chunk,
                embedding: None,
            })
            .await
            .unwrap();

        // Request commit
        let (tx, rx) = oneshot::channel();
        pipeline
            .ingest(IngestItem::Commit {
                stream_id: Uuid::new_v4(),
                respond_to: tx,
            })
            .await
            .unwrap();

        // Wait for commit
        let result = rx.await.unwrap();
        assert!(result.is_ok());

        // Shutdown
        let _ = shutdown_tx.send(());
    }
}
