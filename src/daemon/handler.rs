//! Request Handler
//!
//! Dispatches incoming requests to the appropriate service and returns responses.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use super::metrics::{DaemonMetrics, Timer};
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
    metrics: Arc<DaemonMetrics>,
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
        metrics: Arc<DaemonMetrics>,
    ) -> Self {
        let job_manager = Arc::new(JobManager::new(
            index_manager.clone(),
            write_pipeline.clone(),
            config,
            shutdown_tx.clone(),
            metrics.clone(),
        ));

        Self {
            index_manager,
            write_pipeline,
            job_manager,
            metrics,
            start_time: Instant::now(),
            shutdown_tx,
            active_streams: DashMap::new(),
            query_coordinator: RwLock::new(None),
        }
    }

    /// Get a reference to the daemon metrics
    pub fn metrics(&self) -> &Arc<DaemonMetrics> {
        &self.metrics
    }

    /// Set the query coordinator for distributed search
    pub fn set_query_coordinator(&self, coordinator: Arc<QueryCoordinator>) {
        *self.query_coordinator.write() = Some(coordinator);
    }

    /// Subscribe to SSE events for a scrape job
    pub fn subscribe_job_events(
        &self,
        job_id: uuid::Uuid,
    ) -> Option<broadcast::Receiver<super::scrape_events::ScrapeEvent>> {
        self.job_manager.subscribe_events(job_id)
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
                filters,
            } => self.handle_search(query, top_k, filters).await,

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

            // Cluster operations
            Request::ClusterDocuments {
                document_urls,
                max_clusters,
            } => self.handle_cluster_documents(document_urls, max_clusters).await,

            // Delete operations
            Request::DeleteDocuments { document_ids } => {
                self.handle_delete_documents(document_ids).await
            }
            Request::ClearIndex => self.handle_clear_index().await,

            // Management operations
            Request::Ping => Response::Pong,
            Request::Status => self.handle_status().await,
            Request::Stats => self.handle_stats().await,
            Request::Metrics => self.handle_metrics().await,
            Request::ForceCommit => self.handle_force_commit().await,
            Request::Shutdown => self.handle_shutdown().await,
        }
    }

    // ============ Query Handlers ============

    async fn handle_search(
        &self,
        query: String,
        top_k: usize,
        filters: Option<crate::types::QueryFilters>,
    ) -> Response {
        self.metrics.queries_total.inc();
        let timer = Timer::start();

        // Check if we have a query coordinator for distributed search
        let coordinator = self.query_coordinator.read().clone();

        let response = if let Some(coordinator) = coordinator {
            // Distributed search via QueryCoordinator
            let mut query_obj = crate::types::Query::new(query, top_k);
            query_obj.filters = filters;
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
                    self.metrics.queries_failed.inc();
                    Response::error(ErrorCode::SearchFailed, e.to_string())
                }
            }
        } else {
            // Local-only search
            match self.index_manager.search_with_filters(&query, top_k, filters.as_ref()) {
                Ok((results, query_time_ms)) => Response::SearchResults {
                    results,
                    query_time_ms,
                },
                Err(e) => {
                    error!("Search failed: {}", e);
                    self.metrics.queries_failed.inc();
                    Response::error(ErrorCode::SearchFailed, e.to_string())
                }
            }
        };

        timer.record(&self.metrics.query_latency);
        response
    }

    // ============ Write Handlers ============

    /// Remove indexing streams that have been idle for too long
    fn cleanup_stale_streams(&self) {
        const STREAM_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes
        self.active_streams.retain(|id, stream| {
            let stale = stream.created_at.elapsed() > STREAM_TIMEOUT;
            if stale {
                warn!("Removing stale indexing stream {}", id);
            }
            !stale
        });
    }

    async fn handle_index_documents(&self, stream_id: Uuid) -> Response {
        // Clean up any stale streams before creating new ones
        self.cleanup_stale_streams();

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
                self.metrics.writes_failed.inc();
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
            self.metrics.writes_failed.inc();
            return Response::error(ErrorCode::IndexFailed, e.to_string());
        }

        // Wait for commit to complete
        match rx.await {
            Ok(Ok(())) => {
                self.metrics.chunks_indexed.add(chunk_count as u64);
                self.metrics.documents_indexed.inc();
                Response::JobComplete {
                    job_id: stream_id,
                    stats: JobStats {
                        documents_processed: 1,
                        chunks_indexed: chunk_count,
                        duration_ms,
                        errors: 0,
                    },
                }
            }
            Ok(Err(e)) => {
                error!("Commit failed: {}", e);
                self.metrics.writes_failed.inc();
                Response::error(ErrorCode::IndexFailed, e.to_string())
            }
            Err(_) => {
                self.metrics.writes_failed.inc();
                Response::error(ErrorCode::InternalError, "Commit channel closed")
            }
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

    // ============ Delete Handlers ============

    async fn handle_delete_documents(&self, document_ids: Vec<String>) -> Response {
        info!("Deleting {} documents", document_ids.len());

        match self.index_manager.delete_documents(&document_ids) {
            Ok((docs_deleted, chunks_deleted)) => Response::DeleteComplete {
                documents_deleted: docs_deleted,
                chunks_deleted,
            },
            Err(e) => {
                error!("Delete failed: {}", e);
                Response::error(ErrorCode::DeleteFailed, e.to_string())
            }
        }
    }

    async fn handle_clear_index(&self) -> Response {
        info!("Clearing all index entries");

        match self.index_manager.clear_all() {
            Ok(chunks_deleted) => Response::ClearComplete { chunks_deleted },
            Err(e) => {
                error!("Clear failed: {}", e);
                Response::error(ErrorCode::DeleteFailed, e.to_string())
            }
        }
    }

    // ============ Cluster Handlers ============

    async fn handle_cluster_documents(
        &self,
        document_urls: Vec<String>,
        max_clusters: usize,
    ) -> Response {
        let start = Instant::now();

        match self.index_manager.cluster_documents(&document_urls, max_clusters) {
            Ok(result) => {
                // Group matched docs by cluster assignment
                let num_clusters = result
                    .assignments
                    .iter()
                    .copied()
                    .max()
                    .map(|m| m + 1)
                    .unwrap_or(0)
                    .max(if result.matched_docs.is_empty() { 0 } else { 1 });

                let mut cluster_urls: Vec<Vec<String>> = vec![Vec::new(); num_clusters];
                let mut documents = HashMap::new();

                for (i, doc) in result.matched_docs.iter().enumerate() {
                    let cluster_idx = result.assignments[i];
                    if cluster_idx < num_clusters {
                        cluster_urls[cluster_idx].push(doc.url.clone());
                    }
                    documents.insert(
                        doc.url.clone(),
                        DocumentInfo {
                            title: doc.title.clone(),
                            snippet: doc.snippet.clone(),
                        },
                    );
                }

                let clusters: Vec<ClusterInfo> = cluster_urls
                    .into_iter()
                    .enumerate()
                    .filter(|(_, urls)| !urls.is_empty())
                    .map(|(i, urls)| {
                        let label = generate_cluster_label(&urls, &documents);
                        ClusterInfo {
                            cluster_id: format!("cluster_{}", i),
                            label,
                            document_urls: urls,
                        }
                    })
                    .collect();

                let cluster_time_ms = start.elapsed().as_millis() as u64;

                Response::ClusterResults {
                    clusters,
                    documents,
                    unmatched_urls: result.unmatched_urls,
                    cluster_time_ms,
                }
            }
            Err(e) => {
                error!("Cluster failed: {}", e);
                Response::error(ErrorCode::ClusterFailed, e.to_string())
            }
        }
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

    async fn handle_metrics(&self) -> Response {
        self.metrics.update_memory_usage();
        let snapshot = self.metrics.snapshot();
        Response::Metrics { snapshot }
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
        let timer = Timer::start();
        match self.index_manager.commit() {
            Ok(()) => {
                timer.record(&self.metrics.commit_latency);
                self.metrics.commits_total.inc();
                Response::Ok
            }
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
        super::metrics::get_memory_usage()
            .map(|bytes| bytes / (1024 * 1024))
            .unwrap_or(0)
    }
}

/// Stop words to skip when extracting title keywords
const LABEL_STOP_WORDS: &[&str] = &[
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "been", "being", "have", "has", "had", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "but", "yet", "also", "just",
    "than", "then", "into", "over", "such", "its", "our", "their", "your", "his",
    "her", "about", "between", "through", "during", "before", "after",
];

/// Generate a heuristic label for a cluster from its document URLs and metadata
fn generate_cluster_label(urls: &[String], documents: &HashMap<String, DocumentInfo>) -> String {
    // Collect titles and domains
    let mut domain_counts: HashMap<String, usize> = HashMap::new();
    let mut title_words: HashMap<String, usize> = HashMap::new();

    for url in urls {
        // Extract domain
        if let Ok(parsed) = url::Url::parse(url) {
            if let Some(host) = parsed.host_str() {
                *domain_counts.entry(host.to_string()).or_default() += 1;
            }
        }

        // Extract title keywords
        if let Some(doc) = documents.get(url) {
            if let Some(title) = &doc.title {
                for word in title.split_whitespace() {
                    let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
                    let lower = clean.to_lowercase();
                    if lower.len() >= 4 && !LABEL_STOP_WORDS.contains(&lower.as_str()) {
                        *title_words.entry(lower).or_default() += 1;
                    }
                }
            }
        }
    }

    // Top 2 keywords from titles, sorted by frequency then alphabetically
    let mut word_vec: Vec<_> = title_words.into_iter().collect();
    word_vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let keywords: Vec<String> = word_vec
        .into_iter()
        .take(2)
        .map(|(w, _)| title_case(&w))
        .collect();

    if !keywords.is_empty() {
        return keywords.join(" ");
    }

    // Fall back to most common domain
    if let Some((domain, _)) = domain_counts.into_iter().max_by_key(|(_, c)| *c) {
        return domain;
    }

    "Uncategorized".to_string()
}

/// Title-case a word (first letter uppercase, rest lowercase)
fn title_case(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + &chars.as_str().to_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    async fn create_test_handler() -> (RequestHandler, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = crate::config::Config::default();
        config.node.data_dir = temp_dir.path().to_path_buf();

        let index_manager = Arc::new(IndexManager::load(&config).unwrap());
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let write_pipeline = Arc::new(WritePipeline::start(
            index_manager.clone(),
            None,
            100,
            Duration::from_secs(60),
            shutdown_rx,
        ));

        let metrics = DaemonMetrics::shared();
        let handler = RequestHandler::new(index_manager, write_pipeline, config, shutdown_tx, metrics);
        (handler, temp_dir)
    }

    // ============ Label Generation Tests ============

    #[test]
    fn test_label_from_title_keywords() {
        let urls = vec![
            "https://arxiv.org/abs/2301.001".to_string(),
            "https://arxiv.org/abs/2301.002".to_string(),
            "https://arxiv.org/abs/2301.003".to_string(),
        ];
        let mut documents = HashMap::new();
        documents.insert(
            urls[0].clone(),
            DocumentInfo {
                title: Some("Scaling Laws for Neural Language Models".to_string()),
                snippet: "...".to_string(),
            },
        );
        documents.insert(
            urls[1].clone(),
            DocumentInfo {
                title: Some("Neural Network Pruning Techniques".to_string()),
                snippet: "...".to_string(),
            },
        );
        documents.insert(
            urls[2].clone(),
            DocumentInfo {
                title: Some("Deep Neural Architecture Search".to_string()),
                snippet: "...".to_string(),
            },
        );

        let label = generate_cluster_label(&urls, &documents);
        // "neural" appears 3 times — should be top keyword
        assert!(
            label.contains("Neural"),
            "Expected 'Neural' in label, got: {}",
            label
        );
    }

    #[test]
    fn test_label_domain_fallback_when_no_titles() {
        let urls = vec![
            "https://github.com/rust-lang/rust".to_string(),
            "https://github.com/tokio-rs/tokio".to_string(),
            "https://docs.rs/serde".to_string(),
        ];
        let mut documents = HashMap::new();
        // No titles — just snippets
        for url in &urls {
            documents.insert(
                url.clone(),
                DocumentInfo {
                    title: None,
                    snippet: "Some content".to_string(),
                },
            );
        }

        let label = generate_cluster_label(&urls, &documents);
        // github.com has 2 occurrences, docs.rs has 1 — should pick github.com
        assert_eq!(label, "github.com");
    }

    #[test]
    fn test_label_uncategorized_fallback() {
        let urls: Vec<String> = Vec::new();
        let documents = HashMap::new();

        let label = generate_cluster_label(&urls, &documents);
        assert_eq!(label, "Uncategorized");
    }

    #[test]
    fn test_label_filters_stop_words_and_short_words() {
        let urls = vec!["https://example.com/1".to_string()];
        let mut documents = HashMap::new();
        documents.insert(
            urls[0].clone(),
            DocumentInfo {
                // "the", "and", "for" are stop words; "AI" is < 4 chars
                title: Some("The AI and Machine Learning for Research".to_string()),
                snippet: "...".to_string(),
            },
        );

        let label = generate_cluster_label(&urls, &documents);
        // Should pick "machine" and "learning" (both 7+ chars, not stop words)
        assert!(
            label.contains("Machine") || label.contains("Learning") || label.contains("Research"),
            "Expected meaningful keywords, got: {}",
            label
        );
        assert!(
            !label.to_lowercase().contains("the "),
            "Should not contain stop words, got: {}",
            label
        );
    }

    #[test]
    fn test_label_title_case() {
        assert_eq!(title_case("machine"), "Machine");
        assert_eq!(title_case("LEARNING"), "Learning");
        assert_eq!(title_case(""), "");
    }

    #[test]
    fn test_label_at_most_two_keywords() {
        let urls = vec!["https://example.com/1".to_string()];
        let mut documents = HashMap::new();
        documents.insert(
            urls[0].clone(),
            DocumentInfo {
                title: Some(
                    "Advanced Machine Learning Algorithms Research".to_string(),
                ),
                snippet: "...".to_string(),
            },
        );

        let label = generate_cluster_label(&urls, &documents);
        let word_count = label.split_whitespace().count();
        assert!(
            word_count <= 2,
            "Label should have at most 2 words, got {}: '{}'",
            word_count,
            label
        );
    }

    // ============ Handler Tests ============

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

    // ============ Cluster Handler Integration Tests ============

    /// Seed a document into the temp directory's ChunkStorage for handler tests
    fn seed_test_document(
        data_dir: &std::path::Path,
        url: &str,
        title: &str,
        content: &str,
        embedding: Vec<f32>,
    ) {
        let storage = crate::index::ChunkStorage::new(data_dir).unwrap();
        let doc_id = format!("doc-{}", crate::util::fast_hash(url));
        let chunk_id = format!("chunk-{}-0", doc_id);

        let mut metadata = crate::types::ChunkMetadata::new(chunk_id, doc_id);
        metadata.source_url = Some(url.to_string());
        metadata.source_title = Some(title.to_string());

        let chunk = crate::types::Chunk {
            metadata,
            content: content.to_string(),
            token_count: 10,
        };

        let indexed = crate::types::IndexedChunk {
            chunk,
            embedding,
            lsh_signature: None,
            index_key: 0,
        };

        storage.store(&indexed).unwrap();
        storage.save().unwrap();
    }

    async fn create_seeded_handler() -> (RequestHandler, TempDir) {
        let temp_dir = TempDir::new().unwrap();

        // Seed documents before loading the IndexManager
        seed_test_document(
            temp_dir.path(),
            "https://arxiv.org/abs/2301.001",
            "Scaling Laws for Neural Language Models",
            "We study empirical scaling laws for language model performance.",
            vec![0.9, 0.1, 0.0],
        );
        seed_test_document(
            temp_dir.path(),
            "https://arxiv.org/abs/2301.002",
            "Neural Network Pruning Techniques",
            "This paper presents structured pruning methods for transformers.",
            vec![0.85, 0.15, 0.0],
        );
        seed_test_document(
            temp_dir.path(),
            "https://github.com/rust-lang/rust",
            "The Rust Programming Language",
            "Rust is a systems programming language focused on safety.",
            vec![0.0, 0.0, 1.0],
        );

        let mut config = crate::config::Config::default();
        config.node.data_dir = temp_dir.path().to_path_buf();

        let index_manager = Arc::new(IndexManager::load(&config).unwrap());
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let write_pipeline = Arc::new(WritePipeline::start(
            index_manager.clone(),
            None,
            100,
            Duration::from_secs(60),
            shutdown_rx,
        ));

        let metrics = DaemonMetrics::shared();
        let handler =
            RequestHandler::new(index_manager, write_pipeline, config, shutdown_tx, metrics);
        (handler, temp_dir)
    }

    #[tokio::test]
    async fn test_cluster_documents_returns_clusters_and_labels() {
        let (handler, _temp) = create_seeded_handler().await;

        let response = handler
            .handle(Request::ClusterDocuments {
                document_urls: vec![
                    "https://arxiv.org/abs/2301.001".to_string(),
                    "https://arxiv.org/abs/2301.002".to_string(),
                    "https://github.com/rust-lang/rust".to_string(),
                ],
                max_clusters: 2,
            })
            .await;

        match response {
            Response::ClusterResults {
                clusters,
                documents,
                unmatched_urls,
                cluster_time_ms,
            } => {
                // Should have clusters (at least 1, at most 2)
                assert!(!clusters.is_empty(), "Expected at least one cluster");
                assert!(clusters.len() <= 2, "Expected at most 2 clusters");

                // All cluster IDs should be well-formed
                for cluster in &clusters {
                    assert!(cluster.cluster_id.starts_with("cluster_"));
                    assert!(!cluster.label.is_empty(), "Label should not be empty");
                    assert!(
                        !cluster.document_urls.is_empty(),
                        "Each cluster should have documents"
                    );
                }

                // All 3 docs should appear across clusters
                let total_docs: usize = clusters.iter().map(|c| c.document_urls.len()).sum();
                assert_eq!(total_docs, 3);

                // Documents map should have entries for all matched URLs
                assert_eq!(documents.len(), 3);
                assert!(documents.contains_key("https://arxiv.org/abs/2301.001"));
                assert!(documents.contains_key("https://github.com/rust-lang/rust"));

                // Title should be preserved
                let arxiv_doc = &documents["https://arxiv.org/abs/2301.001"];
                assert_eq!(
                    arxiv_doc.title.as_deref(),
                    Some("Scaling Laws for Neural Language Models")
                );

                // No unmatched URLs
                assert!(unmatched_urls.is_empty());

                // Timing should be populated
                assert!(cluster_time_ms < 10_000, "Clustering should be fast");
            }
            other => panic!("Expected ClusterResults, got: {:?}", std::mem::discriminant(&other)),
        }
    }

    #[tokio::test]
    async fn test_cluster_documents_with_unmatched_urls() {
        let (handler, _temp) = create_seeded_handler().await;

        let response = handler
            .handle(Request::ClusterDocuments {
                document_urls: vec![
                    "https://arxiv.org/abs/2301.001".to_string(),
                    "https://not-indexed.com/missing".to_string(),
                    "https://also-missing.org/page".to_string(),
                ],
                max_clusters: 5,
            })
            .await;

        match response {
            Response::ClusterResults {
                clusters,
                documents,
                unmatched_urls,
                ..
            } => {
                // One matched doc → one cluster
                assert_eq!(clusters.len(), 1);
                assert_eq!(clusters[0].document_urls.len(), 1);

                // Documents map has only the matched one
                assert_eq!(documents.len(), 1);

                // Two unmatched
                assert_eq!(unmatched_urls.len(), 2);
                assert!(unmatched_urls.contains(&"https://not-indexed.com/missing".to_string()));
                assert!(unmatched_urls.contains(&"https://also-missing.org/page".to_string()));
            }
            other => panic!("Expected ClusterResults, got: {:?}", std::mem::discriminant(&other)),
        }
    }

    #[tokio::test]
    async fn test_cluster_documents_all_unmatched() {
        let (handler, _temp) = create_test_handler().await;

        let response = handler
            .handle(Request::ClusterDocuments {
                document_urls: vec![
                    "https://nothing.com/a".to_string(),
                    "https://nothing.com/b".to_string(),
                ],
                max_clusters: 3,
            })
            .await;

        match response {
            Response::ClusterResults {
                clusters,
                documents,
                unmatched_urls,
                ..
            } => {
                assert!(clusters.is_empty());
                assert!(documents.is_empty());
                assert_eq!(unmatched_urls.len(), 2);
            }
            other => panic!("Expected ClusterResults, got: {:?}", std::mem::discriminant(&other)),
        }
    }
}
