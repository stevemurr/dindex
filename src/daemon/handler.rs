//! Request Handler
//!
//! Dispatches incoming requests to the appropriate service and returns responses.

use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::{broadcast, oneshot};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::config::Config;
use crate::query::QueryCoordinator;
use crate::types::Chunk;

use super::index_manager::IndexManager;
use super::jobs::JobManager;
use super::protocol::*;
use super::write_pipeline::{IngestItem, WritePipeline};

/// Active indexing stream state
struct IndexStream {
    chunks: Vec<(Chunk, Option<Vec<f32>>)>,
    created_at: Instant,
}

/// Request handler that processes incoming IPC requests
pub struct RequestHandler {
    index_manager: Arc<IndexManager>,
    write_pipeline: Arc<WritePipeline>,
    job_manager: Arc<JobManager>,
    start_time: Instant,
    shutdown_tx: broadcast::Sender<()>,

    /// Active indexing streams (stream_id -> stream state)
    active_streams: DashMap<Uuid, IndexStream>,

    /// Optional query coordinator for distributed search
    query_coordinator: RwLock<Option<Arc<QueryCoordinator>>>,
}

impl RequestHandler {
    /// Create a new request handler
    pub fn new(
        index_manager: Arc<IndexManager>,
        write_pipeline: Arc<WritePipeline>,
        config: Config,
        shutdown_tx: broadcast::Sender<()>,
    ) -> Self {
        let job_manager = Arc::new(JobManager::new(
            index_manager.clone(),
            write_pipeline.clone(),
            config,
            shutdown_tx.clone(),
        ));

        Self {
            index_manager,
            write_pipeline,
            job_manager,
            start_time: Instant::now(),
            shutdown_tx,
            active_streams: DashMap::new(),
            query_coordinator: RwLock::new(None),
        }
    }

    /// Set the query coordinator for distributed search
    pub fn set_query_coordinator(&self, coordinator: Arc<QueryCoordinator>) {
        *self.query_coordinator.write() = Some(coordinator);
    }

    /// Handle an incoming request and return a response
    pub async fn handle(&self, request: Request) -> Response {
        debug!("Handling request: {:?}", std::mem::discriminant(&request));

        match request {
            // Query operations
            Request::Search {
                query,
                top_k,
                format: _,
            } => self.handle_search(query, top_k).await,

            // Write operations
            Request::IndexDocuments { stream_id } => self.handle_index_documents(stream_id).await,
            Request::IndexChunk { stream_id, chunk } => {
                self.handle_index_chunk(stream_id, chunk).await
            }
            Request::IndexComplete { stream_id } => self.handle_index_complete(stream_id).await,

            // Import operations
            Request::ImportStart { source, options } => {
                self.handle_import_start(source, options).await
            }
            Request::ImportCancel { job_id } => self.handle_job_cancel(job_id).await,
            Request::JobProgress { job_id } => self.handle_job_progress(job_id).await,

            // Scrape operations
            Request::ScrapeStart { urls, options } => {
                self.handle_scrape_start(urls, options).await
            }
            Request::ScrapeCancel { job_id } => self.handle_job_cancel(job_id).await,

            // Management operations
            Request::Ping => Response::Pong,
            Request::Status => self.handle_status().await,
            Request::Stats => self.handle_stats().await,
            Request::ForceCommit => self.handle_force_commit().await,
            Request::Shutdown => self.handle_shutdown().await,
        }
    }

    // ============ Query Handlers ============

    async fn handle_search(&self, query: String, top_k: usize) -> Response {
        // Check if we have a query coordinator for distributed search
        let coordinator = self.query_coordinator.read().clone();

        if let Some(coordinator) = coordinator {
            // Distributed search via QueryCoordinator
            let query_obj = crate::types::Query::new(query, top_k);
            match coordinator.execute(&query_obj).await {
                Ok(aggregated) => {
                    info!(
                        "Distributed search: {} results from {} nodes (quality: {:.2})",
                        aggregated.results.len(),
                        aggregated.responding_nodes.len(),
                        aggregated.quality_estimate
                    );
                    Response::SearchResults {
                        results: aggregated.results,
                        query_time_ms: aggregated.total_time_ms,
                    }
                }
                Err(e) => {
                    error!("Distributed search failed: {}", e);
                    Response::error(ErrorCode::SearchFailed, e.to_string())
                }
            }
        } else {
            // Local-only search
            match self.index_manager.search(&query, top_k) {
                Ok((results, query_time_ms)) => Response::SearchResults {
                    results,
                    query_time_ms,
                },
                Err(e) => {
                    error!("Search failed: {}", e);
                    Response::error(ErrorCode::SearchFailed, e.to_string())
                }
            }
        }
    }

    // ============ Write Handlers ============

    async fn handle_index_documents(&self, stream_id: Uuid) -> Response {
        // Create a new indexing stream
        self.active_streams.insert(
            stream_id,
            IndexStream {
                chunks: Vec::new(),
                created_at: Instant::now(),
            },
        );

        debug!("Created indexing stream: {}", stream_id);
        Response::StreamReady { stream_id }
    }

    async fn handle_index_chunk(&self, stream_id: Uuid, chunk: ChunkPayload) -> Response {
        let mut stream = match self.active_streams.get_mut(&stream_id) {
            Some(s) => s,
            None => {
                return Response::error(
                    ErrorCode::StreamNotFound,
                    format!("Stream {} not found", stream_id),
                )
            }
        };

        // Convert payload to chunk
        let chunk_obj = Chunk {
            metadata: chunk.metadata,
            content: chunk.content,
            token_count: 0, // Will be recalculated if needed
        };

        stream.chunks.push((chunk_obj, chunk.embedding));
        let count = stream.chunks.len();

        Response::ChunkAck { stream_id, count }
    }

    async fn handle_index_complete(&self, stream_id: Uuid) -> Response {
        // Remove the stream
        let stream = match self.active_streams.remove(&stream_id) {
            Some((_, s)) => s,
            None => {
                return Response::error(
                    ErrorCode::StreamNotFound,
                    format!("Stream {} not found", stream_id),
                )
            }
        };

        let chunk_count = stream.chunks.len();
        let duration_ms = stream.created_at.elapsed().as_millis() as u64;

        // Send chunks to write pipeline
        for (chunk, embedding) in stream.chunks {
            if let Err(e) = self
                .write_pipeline
                .ingest(IngestItem::Chunk {
                    stream_id,
                    chunk,
                    embedding,
                })
                .await
            {
                error!("Failed to ingest chunk: {}", e);
                return Response::error(ErrorCode::IndexFailed, e.to_string());
            }
        }

        // Request commit
        let (tx, rx) = oneshot::channel();
        if let Err(e) = self
            .write_pipeline
            .ingest(IngestItem::Commit {
                stream_id,
                respond_to: tx,
            })
            .await
        {
            error!("Failed to request commit: {}", e);
            return Response::error(ErrorCode::IndexFailed, e.to_string());
        }

        // Wait for commit to complete
        match rx.await {
            Ok(Ok(())) => Response::JobComplete {
                job_id: stream_id,
                stats: JobStats {
                    documents_processed: 1,
                    chunks_indexed: chunk_count,
                    duration_ms,
                    errors: 0,
                },
            },
            Ok(Err(e)) => {
                error!("Commit failed: {}", e);
                Response::error(ErrorCode::IndexFailed, e.to_string())
            }
            Err(_) => Response::error(ErrorCode::InternalError, "Commit channel closed"),
        }
    }

    // ============ Import Handlers ============

    async fn handle_import_start(&self, source: ImportSource, options: ImportOptions) -> Response {
        info!(
            "Starting import job: {:?} with options {:?}",
            source, options
        );

        let job_id = self.job_manager.start_import(source, options);
        Response::JobStarted { job_id }
    }

    async fn handle_job_cancel(&self, job_id: Uuid) -> Response {
        if self.job_manager.cancel(job_id) {
            info!("Cancelled job: {}", job_id);
            Response::Ok
        } else {
            Response::error(
                ErrorCode::JobNotFound,
                format!("Job {} not found or already completed", job_id),
            )
        }
    }

    async fn handle_job_progress(&self, job_id: Uuid) -> Response {
        match self.job_manager.get_progress(job_id) {
            Some(progress) => Response::JobProgress { job_id, progress },
            None => Response::error(
                ErrorCode::JobNotFound,
                format!("Job {} not found", job_id),
            ),
        }
    }

    // ============ Scrape Handlers ============

    async fn handle_scrape_start(&self, urls: Vec<String>, options: ScrapeOptions) -> Response {
        info!(
            "Starting scrape job: {} URLs with options {:?}",
            urls.len(),
            options
        );

        let job_id = self.job_manager.start_scrape(urls, options);
        Response::JobStarted { job_id }
    }

    // ============ Management Handlers ============

    async fn handle_status(&self) -> Response {
        let uptime_seconds = self.start_time.elapsed().as_secs();
        let memory_mb = self.get_memory_usage();
        let active_jobs = self.job_manager.active_count();
        let pending_writes = self.index_manager.pending_count();

        Response::Status(DaemonStatus {
            running: true,
            uptime_seconds,
            memory_mb,
            active_jobs,
            pending_writes,
        })
    }

    async fn handle_stats(&self) -> Response {
        match self.index_manager.stats() {
            Ok(stats) => Response::Stats(stats),
            Err(e) => {
                error!("Failed to get stats: {}", e);
                Response::error(ErrorCode::InternalError, e.to_string())
            }
        }
    }

    async fn handle_force_commit(&self) -> Response {
        match self.index_manager.commit() {
            Ok(()) => Response::Ok,
            Err(e) => {
                error!("Force commit failed: {}", e);
                Response::error(ErrorCode::InternalError, e.to_string())
            }
        }
    }

    async fn handle_shutdown(&self) -> Response {
        info!("Shutdown requested");

        // Commit any pending changes
        if let Err(e) = self.index_manager.commit() {
            warn!("Failed to commit during shutdown: {}", e);
        }

        // Signal shutdown
        let _ = self.shutdown_tx.send(());

        Response::Ok
    }

    /// Get approximate memory usage in MB
    fn get_memory_usage(&self) -> u64 {
        // Try to read from /proc/self/statm on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/self/statm") {
                if let Some(rss) = content.split_whitespace().nth(1) {
                    if let Ok(pages) = rss.parse::<u64>() {
                        // Page size is typically 4KB
                        return pages * 4 / 1024;
                    }
                }
            }
        }

        // Fallback: return 0
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_handler() -> (RequestHandler, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = crate::config::Config::default();
        config.node.data_dir = temp_dir.path().to_path_buf();

        let index_manager = Arc::new(IndexManager::load(&config).unwrap());
        let write_pipeline = Arc::new(WritePipeline::new(index_manager.clone(), 100));
        let (shutdown_tx, _) = broadcast::channel(1);

        let handler = RequestHandler::new(index_manager, write_pipeline, config, shutdown_tx);
        (handler, temp_dir)
    }

    #[tokio::test]
    async fn test_ping() {
        let (handler, _temp) = create_test_handler().await;
        let response = handler.handle(Request::Ping).await;
        assert!(matches!(response, Response::Pong));
    }

    #[tokio::test]
    async fn test_status() {
        let (handler, _temp) = create_test_handler().await;
        let response = handler.handle(Request::Status).await;

        match response {
            Response::Status(status) => {
                assert!(status.running);
            }
            _ => panic!("Expected Status response"),
        }
    }

    #[tokio::test]
    async fn test_stats() {
        let (handler, _temp) = create_test_handler().await;
        let response = handler.handle(Request::Stats).await;

        match response {
            Response::Stats(stats) => {
                assert_eq!(stats.total_chunks, 0);
            }
            _ => panic!("Expected Stats response"),
        }
    }
}
