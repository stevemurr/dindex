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

use crate::embedding::{generate_embedding, EmbeddingEngine};
use crate::types::Chunk;

use super::index_manager::IndexManager;
use super::metrics::DaemonMetrics;

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
        Self::start_with_metrics(index_manager, embedding_engine, batch_size, commit_interval, shutdown, None)
    }

    /// Create and start the write pipeline with metrics
    pub fn start_with_metrics(
        index_manager: Arc<IndexManager>,
        embedding_engine: Option<Arc<EmbeddingEngine>>,
        batch_size: usize,
        commit_interval: Duration,
        shutdown: broadcast::Receiver<()>,
        metrics: Option<Arc<DaemonMetrics>>,
    ) -> Self {
        let (ingest_tx, ingest_rx) = mpsc::channel(1000);

        // Spawn the background worker
        let worker = WritePipelineWorker::new(
            index_manager,
            embedding_engine,
            ingest_rx,
            batch_size,
            commit_interval,
            metrics,
        );

        tokio::spawn(async move {
            worker.run(shutdown).await;
        });

        Self {
            ingest_tx,
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

}

/// Background worker that processes the write queue
struct WritePipelineWorker {
    index_manager: Arc<IndexManager>,
    embedding_engine: Option<Arc<EmbeddingEngine>>,
    ingest_rx: mpsc::Receiver<IngestItem>,
    batch_size: usize,
    commit_interval: Duration,
    metrics: Option<Arc<DaemonMetrics>>,
    /// Consecutive flush failures for the current batch
    consecutive_failures: u32,
    /// Total chunks skipped due to embedding failures
    skipped_chunks: u64,
}

impl WritePipelineWorker {
    fn new(
        index_manager: Arc<IndexManager>,
        embedding_engine: Option<Arc<EmbeddingEngine>>,
        ingest_rx: mpsc::Receiver<IngestItem>,
        batch_size: usize,
        commit_interval: Duration,
        metrics: Option<Arc<DaemonMetrics>>,
    ) -> Self {
        Self {
            index_manager,
            embedding_engine,
            ingest_rx,
            batch_size,
            commit_interval,
            metrics,
            consecutive_failures: 0,
            skipped_chunks: 0,
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
                            let embedding = match embedding {
                                Some(e) => e,
                                None => match self.generate_embedding(&chunk.content) {
                                    Ok(e) => e,
                                    Err(e) => {
                                        self.skipped_chunks += 1;
                                        warn!(
                                            "Skipping chunk {} (total skipped: {}): embedding failed: {}",
                                            chunk.metadata.chunk_id, self.skipped_chunks, e
                                        );
                                        continue;
                                    }
                                },
                            };

                            batch.push((chunk, embedding));

                            // Flush if batch is full
                            if batch.len() >= self.batch_size {
                                if let Err(e) = self.flush_batch(&mut batch).await {
                                    warn!("Batch flush failed: {}", e);
                                }
                            }
                        }
                        IngestItem::Commit { respond_to, stream_id } => {
                            debug!("Commit requested for stream {}", stream_id);

                            // Flush any pending chunks first
                            let flush_result = if !batch.is_empty() {
                                self.flush_batch(&mut batch).await
                            } else {
                                Ok(0)
                            };

                            // Propagate flush error to commit requester
                            let result = match flush_result {
                                Ok(_) => self.index_manager.commit(),
                                Err(e) => Err(e),
                            };
                            let _ = respond_to.send(result);
                        }
                    }
                }

                // Periodic commit
                _ = interval.tick() => {
                    if !batch.is_empty() {
                        debug!("Periodic flush of {} chunks", batch.len());
                        match self.flush_batch(&mut batch).await {
                            Ok(_) => {
                                if let Err(e) = self.index_manager.commit() {
                                    warn!("Periodic commit failed: {}", e);
                                }
                            }
                            Err(e) => {
                                warn!("Periodic flush failed: {}", e);
                            }
                        }
                    }
                }

                // Shutdown signal
                _ = shutdown.recv() => {
                    info!("Write pipeline shutting down");

                    // Final flush
                    if !batch.is_empty() {
                        if let Err(e) = self.flush_batch(&mut batch).await {
                            error!("Final flush failed, {} chunks may be lost: {}", batch.len(), e);
                            batch.clear();
                        }
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

    /// Flush a batch of chunks to the index.
    ///
    /// On success, clears the batch and resets the failure counter.
    /// On failure, keeps the batch for retry. After 3 consecutive failures,
    /// drops the batch to prevent unbounded memory growth.
    async fn flush_batch(&mut self, batch: &mut Vec<(Chunk, Vec<f32>)>) -> Result<usize> {
        if batch.is_empty() {
            return Ok(0);
        }

        debug!("Flushing batch of {} chunks", batch.len());

        match self.index_manager.index_batch(batch) {
            Ok(keys) => {
                let count = keys.len();
                debug!("Indexed {} chunks with keys {:?}", count, &keys[..count.min(3)]);
                self.consecutive_failures = 0;
                batch.clear();
                Ok(count)
            }
            Err(e) => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= 3 {
                    error!(
                        "Failed to index batch after {} attempts, dropping {} chunks: {}",
                        self.consecutive_failures,
                        batch.len(),
                        e
                    );
                    batch.clear();
                    self.consecutive_failures = 0;
                } else {
                    error!(
                        "Failed to index batch (attempt {}), will retry: {}",
                        self.consecutive_failures, e
                    );
                }
                Err(anyhow::anyhow!("Failed to index batch: {}", e))
            }
        }
    }

    /// Generate embedding for content using the real embedding engine.
    ///
    /// Returns an error if no engine is available or embedding fails.
    fn generate_embedding(&self, content: &str) -> anyhow::Result<Vec<f32>> {
        if let Some(ref metrics) = self.metrics {
            metrics.embedding_requests_total.inc();
            let start = std::time::Instant::now();
            let result = generate_embedding(
                self.embedding_engine.as_deref(),
                content,
            );
            metrics.embedding_latency.observe(start.elapsed());
            if result.is_err() {
                metrics.embedding_errors_total.inc();
            }
            result
        } else {
            generate_embedding(
                self.embedding_engine.as_deref(),
                content,
            )
        }
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
