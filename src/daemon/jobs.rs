//! Background Job Management
//!
//! Handles long-running import and scrape jobs with progress tracking.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::{broadcast, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::chunking::TextSplitter;
use crate::config::Config;
use crate::import::{DumpSource, WikimediaSource};
use crate::scraping::coordinator::{
    ProcessOutcome, ProcessResult, ProcessedContent,
    ScrapingConfig as CoordinatorScrapingConfig, ScrapingCoordinator,
};
use crate::types::Document;
use url::Url;

use super::index_manager::IndexManager;
use super::metrics::DaemonMetrics;
use super::protocol::{ImportOptions, ImportSource, JobStats, Progress, ProgressStage, ScrapeOptions};
use super::scrape_events::{JobCompletionStatus, ScrapeEvent, UrlInfo, UrlSource, UrlStatus};
use super::write_pipeline::{IngestItem, WritePipeline};

/// Job state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Job information
pub struct JobInfo {
    pub id: Uuid,
    pub job_type: JobType,
    pub state: JobState,
    pub progress: Progress,
    pub stats: Option<JobStats>,
    pub error: Option<String>,
    pub started_at: Instant,
    pub completed_at: Option<Instant>,
    /// Cancel channel - wrapped in Mutex since oneshot::Sender is not Clone
    cancel_tx: std::sync::Mutex<Option<oneshot::Sender<()>>>,
    /// SSE event broadcaster (scrape jobs only)
    pub event_tx: Option<broadcast::Sender<ScrapeEvent>>,
    /// Per-URL status tracking (scrape jobs only)
    pub url_statuses: Option<Arc<DashMap<String, UrlInfo>>>,
}

/// Type of job
#[derive(Debug, Clone)]
pub enum JobType {
    Import { source: ImportSource },
    Scrape { urls: Vec<String> },
}

/// How long to retain completed/failed/cancelled jobs before cleanup
const JOB_RETENTION: Duration = Duration::from_secs(3600); // 1 hour

/// Channel capacity for scrape SSE events
const SCRAPE_EVENT_CHANNEL_CAPACITY: usize = 256;

/// Maximum consecutive empty frontier iterations before declaring exhaustion
const MAX_EMPTY_ITERATIONS: u32 = 10;

/// Job manager for tracking and controlling background jobs
pub struct JobManager {
    jobs: Arc<DashMap<Uuid, JobInfo>>,
    index_manager: Arc<IndexManager>,
    write_pipeline: Arc<WritePipeline>,
    config: Config,
    shutdown_tx: broadcast::Sender<()>,
    metrics: Arc<DaemonMetrics>,
}

impl JobManager {
    /// Create a new job manager
    pub fn new(
        index_manager: Arc<IndexManager>,
        write_pipeline: Arc<WritePipeline>,
        config: Config,
        shutdown_tx: broadcast::Sender<()>,
        metrics: Arc<DaemonMetrics>,
    ) -> Self {
        Self {
            jobs: Arc::new(DashMap::new()),
            index_manager,
            write_pipeline,
            config,
            shutdown_tx,
            metrics,
        }
    }

    /// Start an import job
    pub fn start_import(
        &self,
        source: ImportSource,
        options: ImportOptions,
    ) -> Uuid {
        self.cleanup_old_jobs();
        let job_id = Uuid::new_v4();
        let (cancel_tx, cancel_rx) = oneshot::channel();

        let job_info = JobInfo {
            id: job_id,
            job_type: JobType::Import { source: source.clone() },
            state: JobState::Running,
            progress: Progress {
                job_id,
                stage: ProgressStage::Starting,
                current: 0,
                total: None,
                rate: None,
                eta_seconds: None,
            },
            stats: None,
            error: None,
            started_at: Instant::now(),
            completed_at: None,
            cancel_tx: std::sync::Mutex::new(Some(cancel_tx)),
            event_tx: None,
            url_statuses: None,
        };

        self.jobs.insert(job_id, job_info);
        self.metrics.jobs_started.inc();

        // Spawn import task
        let jobs = Arc::clone(&self.jobs);
        let config = self.config.clone();
        let index_manager = self.index_manager.clone();
        let write_pipeline = self.write_pipeline.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            let result = run_import_job(
                job_id,
                source,
                options,
                config,
                index_manager,
                write_pipeline,
                Arc::clone(&jobs),
                cancel_rx,
                &mut shutdown_rx,
            )
            .await;

            // Update job state
            if let Some(mut job) = jobs.get_mut(&job_id) {
                job.completed_at = Some(Instant::now());
                match result {
                    Ok(stats) => {
                        job.state = JobState::Completed;
                        job.stats = Some(stats);
                        job.progress.stage = ProgressStage::Completed;
                        metrics.jobs_completed.inc();
                    }
                    Err(e) => {
                        if job.state != JobState::Cancelled {
                            job.state = JobState::Failed;
                            job.error = Some(e.to_string());
                            job.progress.stage = ProgressStage::Failed;
                            metrics.jobs_failed.inc();
                        }
                    }
                }
            }
        });

        job_id
    }

    /// Start a scrape job
    pub fn start_scrape(
        &self,
        urls: Vec<String>,
        options: ScrapeOptions,
    ) -> Uuid {
        self.cleanup_old_jobs();
        let job_id = Uuid::new_v4();
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let (event_tx, _) = broadcast::channel::<ScrapeEvent>(SCRAPE_EVENT_CHANNEL_CAPACITY);
        let url_statuses: Arc<DashMap<String, UrlInfo>> = Arc::new(DashMap::new());

        let job_info = JobInfo {
            id: job_id,
            job_type: JobType::Scrape { urls: urls.clone() },
            state: JobState::Running,
            progress: Progress {
                job_id,
                stage: ProgressStage::Starting,
                current: 0,
                total: Some(urls.len() as u64),
                rate: None,
                eta_seconds: None,
            },
            stats: None,
            error: None,
            started_at: Instant::now(),
            completed_at: None,
            cancel_tx: std::sync::Mutex::new(Some(cancel_tx)),
            event_tx: Some(event_tx.clone()),
            url_statuses: Some(Arc::clone(&url_statuses)),
        };

        self.jobs.insert(job_id, job_info);
        self.metrics.jobs_started.inc();

        // Spawn scrape task
        let jobs = Arc::clone(&self.jobs);
        let config = self.config.clone();
        let index_manager = self.index_manager.clone();
        let write_pipeline = self.write_pipeline.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let task_event_tx = event_tx.clone();
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            let result = run_scrape_job(
                job_id,
                urls,
                options,
                config,
                index_manager,
                write_pipeline,
                Arc::clone(&jobs),
                cancel_rx,
                &mut shutdown_rx,
                task_event_tx.clone(),
                Arc::clone(&url_statuses),
                metrics.clone(),
            )
            .await;

            // Update job state and emit completion event
            if let Some(mut job) = jobs.get_mut(&job_id) {
                job.completed_at = Some(Instant::now());
                match &result {
                    Ok(stats) => {
                        job.state = JobState::Completed;
                        job.stats = Some(stats.clone());
                        job.progress.stage = ProgressStage::Completed;
                        metrics.jobs_completed.inc();
                        let _ = task_event_tx.send(ScrapeEvent::JobCompleted {
                            job_id,
                            status: JobCompletionStatus::Completed,
                            stats: Some(stats.clone()),
                            error: None,
                        });
                    }
                    Err(e) => {
                        if job.state == JobState::Cancelled {
                            let _ = task_event_tx.send(ScrapeEvent::JobCompleted {
                                job_id,
                                status: JobCompletionStatus::Cancelled,
                                stats: None,
                                error: None,
                            });
                        } else {
                            job.state = JobState::Failed;
                            let err_msg = e.to_string();
                            job.error = Some(err_msg.clone());
                            job.progress.stage = ProgressStage::Failed;
                            metrics.jobs_failed.inc();
                            let _ = task_event_tx.send(ScrapeEvent::JobCompleted {
                                job_id,
                                status: JobCompletionStatus::Failed,
                                stats: None,
                                error: Some(err_msg),
                            });
                        }
                    }
                }
            }
        });

        job_id
    }

    /// Subscribe to SSE events for a scrape job
    pub fn subscribe_events(&self, job_id: Uuid) -> Option<broadcast::Receiver<ScrapeEvent>> {
        self.jobs
            .get(&job_id)
            .and_then(|job| job.event_tx.as_ref().map(|tx| tx.subscribe()))
    }

    /// Cancel a job
    pub fn cancel(&self, job_id: Uuid) -> bool {
        if let Some(mut job) = self.jobs.get_mut(&job_id) {
            if job.state == JobState::Running {
                job.state = JobState::Cancelled;
                job.progress.stage = ProgressStage::Cancelled;
                if let Ok(mut guard) = job.cancel_tx.lock() {
                    if let Some(cancel_tx) = guard.take() {
                        let _ = cancel_tx.send(());
                    }
                }
                self.metrics.jobs_cancelled.inc();
                return true;
            }
        }
        false
    }

    /// Get job progress
    pub fn get_progress(&self, job_id: Uuid) -> Option<Progress> {
        self.jobs.get(&job_id).map(|job| job.progress.clone())
    }

    /// Get count of active jobs
    pub fn active_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|r| r.state == JobState::Running)
            .count()
    }

    /// Remove completed/failed/cancelled jobs older than JOB_RETENTION
    fn cleanup_old_jobs(&self) {
        self.jobs.retain(|_, job| {
            job.state == JobState::Running
                || job
                    .completed_at
                    .map(|t| t.elapsed() < JOB_RETENTION)
                    .unwrap_or(true)
        });
    }
}

/// Run an import job
async fn run_import_job(
    job_id: Uuid,
    source: ImportSource,
    _options: ImportOptions,
    config: Config,
    index_manager: Arc<IndexManager>,
    write_pipeline: Arc<WritePipeline>,
    jobs: Arc<DashMap<Uuid, JobInfo>>,
    mut cancel_rx: oneshot::Receiver<()>,
    shutdown_rx: &mut broadcast::Receiver<()>,
) -> anyhow::Result<JobStats> {
    info!("Starting import job {}: {:?}", job_id, source);

    let start_time = Instant::now();
    let mut documents_processed = 0usize;
    let mut chunks_indexed = 0usize;
    let mut errors = 0usize;

    // Update progress helper
    fn update_progress(jobs: &DashMap<Uuid, JobInfo>, job_id: Uuid, stage: ProgressStage, current: u64, total: Option<u64>) {
        if let Some(mut job) = jobs.get_mut(&job_id) {
            job.progress.stage = stage;
            job.progress.current = current;
            job.progress.total = total;

            // Calculate rate and ETA
            let elapsed = job.started_at.elapsed().as_secs_f64();
            if elapsed > 0.0 && current > 0 {
                let rate = current as f64 / elapsed;
                job.progress.rate = Some(rate);
                if let Some(t) = total {
                    let remaining = t.saturating_sub(current);
                    // ETA = remaining items / rate (items per second)
                    let eta = (remaining as f64 / rate) as u64;
                    job.progress.eta_seconds = Some(eta);
                }
            } else if elapsed > 0.0 {
                job.progress.rate = Some(0.0);
            }
        }
    }

    let splitter = TextSplitter::new(config.chunking.clone());

    match source {
        ImportSource::WikimediaXml { path } => {
            let path = PathBuf::from(path);

            // Create wikimedia source - needs to be mutable for iter_documents
            let mut wiki_source = WikimediaSource::open(&path)?;

            update_progress(&jobs, job_id, ProgressStage::Parsing, 0, None);

            // Process documents in batches to avoid holding iterator across await points
            let min_content_length = config.bulk_import.min_content_length;
            let batch_size = 100;
            let mut docs_parsed = 0usize;
            let mut last_progress_update = Instant::now();

            loop {
                // Check for cancellation
                if cancel_rx.try_recv().is_ok() {
                    info!("Import job {} cancelled", job_id);
                    return Err(anyhow::anyhow!("Job cancelled"));
                }

                // Check for shutdown
                match shutdown_rx.try_recv() {
                    Ok(_) | Err(broadcast::error::TryRecvError::Closed) => {
                        info!("Import job {} stopped due to shutdown", job_id);
                        return Err(anyhow::anyhow!("Shutdown"));
                    }
                    Err(broadcast::error::TryRecvError::Empty) | Err(broadcast::error::TryRecvError::Lagged(_)) => {}
                }

                // Collect a batch of documents synchronously (no await while iterator is live)
                let batch: Vec<_> = {
                    let mut doc_iter = wiki_source.iter_documents();
                    let mut batch = Vec::with_capacity(batch_size);
                    for _ in 0..batch_size {
                        match doc_iter.next() {
                            Some(Ok(doc)) => {
                                docs_parsed += 1;
                                if doc.content.len() >= min_content_length {
                                    batch.push(doc);
                                }
                            }
                            Some(Err(e)) => {
                                warn!("Error reading document: {}", e);
                                errors += 1;
                                docs_parsed += 1;
                            }
                            None => break,
                        }
                    }
                    batch
                };

                // Update parsing progress if we haven't started processing yet
                if documents_processed == 0 && docs_parsed > 0 {
                    update_progress(&jobs, job_id, ProgressStage::Parsing, docs_parsed as u64, None);
                }

                // If no documents in batch, we're done
                if batch.is_empty() {
                    break;
                }

                // Process the batch (now safe to await)
                for doc in batch {
                    // Create document and chunk it
                    let document = Document::new(doc.content)
                        .with_title(doc.title);

                    let chunks = splitter.split_document(&document);

                    // Send chunks to write pipeline
                    for chunk in chunks {
                        let stream_id = job_id;
                        write_pipeline
                            .ingest(IngestItem::Chunk {
                                stream_id,
                                chunk,
                                embedding: None,
                            })
                            .await?;
                        chunks_indexed += 1;
                    }

                    documents_processed += 1;

                    // Update progress more frequently (every 500ms or every 10 docs)
                    if documents_processed % 10 == 0 || last_progress_update.elapsed() > Duration::from_millis(500) {
                        update_progress(&jobs, job_id, ProgressStage::Importing, documents_processed as u64, None);
                        last_progress_update = Instant::now();
                    }
                }

                // Yield to allow other tasks to run
                tokio::task::yield_now().await;
            }

            // Commit changes
            update_progress(&jobs, job_id, ProgressStage::Committing, documents_processed as u64, None);
            index_manager.commit()?;
        }
        ImportSource::Zim { path: _ } => {
            // TODO: Implement ZIM import
            return Err(anyhow::anyhow!("ZIM import not yet implemented"));
        }
        ImportSource::Warc { path: _ } => {
            // TODO: Implement WARC import
            return Err(anyhow::anyhow!("WARC import not yet implemented"));
        }
    }

    let duration_ms = start_time.elapsed().as_millis() as u64;

    info!(
        "Import job {} completed: {} docs, {} chunks, {} errors in {}ms",
        job_id, documents_processed, chunks_indexed, errors, duration_ms
    );

    Ok(JobStats {
        documents_processed,
        chunks_indexed,
        duration_ms,
        errors,
    })
}

/// Build a coordinator ScrapingConfig from the TOML config + per-job ScrapeOptions.
fn build_coordinator_config(
    config: &crate::config::ScrapingConfig,
    options: &ScrapeOptions,
) -> CoordinatorScrapingConfig {
    CoordinatorScrapingConfig::from_config(
        config,
        options.max_depth,
        options.stay_on_domain,
        options.delay_ms,
        options.max_pages,
    )
}

/// Emit an SSE event, logging whether any subscribers received it.
fn emit(tx: &broadcast::Sender<ScrapeEvent>, event: ScrapeEvent) {
    let event_name = event.event_name();
    match tx.send(event) {
        Ok(n) => debug!("SSE emit {}: {} subscriber(s)", event_name, n),
        Err(_) => debug!("SSE emit {}: no subscribers connected", event_name),
    }
}

/// Update per-URL status tracking.
fn track_url(
    statuses: &DashMap<String, UrlInfo>,
    url: String,
    status: UrlStatus,
    depth: u8,
    title: Option<String>,
    error: Option<String>,
    chunks_created: usize,
    duration_ms: Option<u64>,
) {
    let key = url.clone();
    statuses.insert(
        key,
        UrlInfo {
            url,
            status,
            depth,
            title,
            error,
            chunks_created,
            duration_ms,
            updated_at: Instant::now(),
        },
    );
}

/// Run a scrape job using the full ScrapingCoordinator.
async fn run_scrape_job(
    job_id: Uuid,
    urls: Vec<String>,
    options: ScrapeOptions,
    config: Config,
    index_manager: Arc<IndexManager>,
    write_pipeline: Arc<WritePipeline>,
    jobs: Arc<DashMap<Uuid, JobInfo>>,
    mut cancel_rx: oneshot::Receiver<()>,
    shutdown_rx: &mut broadcast::Receiver<()>,
    event_tx: broadcast::Sender<ScrapeEvent>,
    url_statuses: Arc<DashMap<String, UrlInfo>>,
    metrics: Arc<DaemonMetrics>,
) -> anyhow::Result<JobStats> {
    info!("Starting scrape job {}: {} URLs", job_id, urls.len());

    let start_time = Instant::now();
    let mut documents_processed = 0usize;
    let mut chunks_indexed = 0usize;
    let mut urls_succeeded = 0u64;
    let mut urls_failed = 0u64;
    let mut urls_skipped = 0u64;
    let max_pages = options.max_pages;

    // Build coordinator config and create coordinator
    let coord_config = build_coordinator_config(&config.scraping, &options);
    let peer_id = format!("daemon-{}", job_id);
    let mut coordinator = ScrapingCoordinator::new(coord_config, peer_id)
        .map_err(|e| anyhow::anyhow!("Failed to create scraping coordinator: {}", e))?;

    // Parse and add seed URLs
    let mut seed_urls = Vec::new();
    for url_str in &urls {
        match Url::parse(url_str) {
            Ok(url) => seed_urls.push(url),
            Err(e) => warn!("Invalid seed URL {}: {}", url_str, e),
        }
    }

    if seed_urls.is_empty() {
        return Err(anyhow::anyhow!("No valid seed URLs"));
    }

    coordinator.add_seeds(seed_urls.clone()).await;

    // Emit job_started and url_queued for seeds
    emit(
        &event_tx,
        ScrapeEvent::JobStarted {
            job_id,
            seed_urls: urls.clone(),
            max_depth: options.max_depth,
            max_pages,
        },
    );

    for seed in &seed_urls {
        let url_str = seed.as_str().to_string();
        emit(
            &event_tx,
            ScrapeEvent::UrlQueued {
                job_id,
                url: url_str.clone(),
                depth: 0,
                source: UrlSource::Seed,
            },
        );
        track_url(&url_statuses, url_str, UrlStatus::Queued, 0, None, None, 0, None);
    }

    let splitter = TextSplitter::new(config.chunking.clone());
    let mut empty_iterations = 0u32;

    // Main scraping loop
    loop {
        // Check for cancellation (non-blocking)
        if cancel_rx.try_recv().is_ok() {
            info!("Scrape job {} cancelled", job_id);
            return Err(anyhow::anyhow!("Job cancelled"));
        }
        if let Ok(_) | Err(broadcast::error::TryRecvError::Closed) = shutdown_rx.try_recv() {
            info!("Scrape job {} stopped due to shutdown", job_id);
            return Err(anyhow::anyhow!("Shutdown"));
        }

        // Check max_pages limit
        let total_processed = urls_succeeded + urls_failed + urls_skipped;
        if total_processed >= max_pages as u64 {
            info!("Scrape job {} reached max_pages limit ({})", job_id, max_pages);
            break;
        }

        // Get next URL from frontier
        let next = coordinator.get_next_url().await;
        let scored_url = match next {
            Some(su) => {
                empty_iterations = 0;
                su
            }
            None => {
                empty_iterations += 1;
                if empty_iterations >= MAX_EMPTY_ITERATIONS {
                    info!("Scrape job {} frontier exhausted", job_id);
                    break;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        };

        let url_str = scored_url.url.as_str().to_string();

        // Emit url_fetching
        emit(
            &event_tx,
            ScrapeEvent::UrlFetching {
                job_id,
                url: url_str.clone(),
            },
        );
        track_url(
            &url_statuses,
            url_str.clone(),
            UrlStatus::Fetching,
            scored_url.depth,
            None,
            None,
            0,
            None,
        );

        // Process URL through the full coordinator pipeline
        metrics.scrape_fetch_total.inc();
        let result = coordinator.process_url(&scored_url.url).await;
        let fetch_duration = result.duration;
        let duration_ms = fetch_duration.as_millis() as u64;
        metrics.scrape_fetch_latency.observe(fetch_duration);

        // Destructure to avoid partial-move issues
        let ProcessResult { outcome, discovered_urls, .. } = result;

        match outcome {
            ProcessOutcome::Success(ProcessedContent { content, metadata, .. }) => {
                // Remove existing document for upsert behavior
                if let Err(e) = index_manager.replace_by_url(&url_str) {
                    warn!("Failed to check/replace existing document for URL {}: {}", url_str, e);
                }

                // Convert to Document and chunk
                let document = ScrapingCoordinator::to_document(&scored_url.url, &content, &metadata);
                let chunks = splitter.split_document(&document);
                let chunk_count = chunks.len();

                // Send chunks to write pipeline
                for chunk in chunks {
                    write_pipeline
                        .ingest(IngestItem::Chunk {
                            stream_id: job_id,
                            chunk,
                            embedding: None,
                        })
                        .await?;
                }

                chunks_indexed += chunk_count;
                documents_processed += 1;
                urls_succeeded += 1;
                metrics.scrape_pages_indexed.inc();

                // Add discovered URLs to frontier and emit events
                let discovered_count = discovered_urls.len();
                for disc_url in &discovered_urls {
                    emit(
                        &event_tx,
                        ScrapeEvent::UrlQueued {
                            job_id,
                            url: disc_url.as_str().to_string(),
                            depth: scored_url.depth + 1,
                            source: UrlSource::Discovered,
                        },
                    );
                    track_url(
                        &url_statuses,
                        disc_url.as_str().to_string(),
                        UrlStatus::Queued,
                        scored_url.depth + 1,
                        None,
                        None,
                        0,
                        None,
                    );
                }
                coordinator
                    .add_discovered_urls(discovered_urls, scored_url.depth)
                    .await;

                emit(
                    &event_tx,
                    ScrapeEvent::UrlIndexed {
                        job_id,
                        url: url_str.clone(),
                        title: Some(content.title.clone()),
                        word_count: content.word_count,
                        chunks_created: chunk_count,
                        duration_ms,
                        discovered_urls: discovered_count,
                    },
                );
                track_url(
                    &url_statuses,
                    url_str.clone(),
                    UrlStatus::Indexed,
                    scored_url.depth,
                    Some(content.title.clone()),
                    None,
                    chunk_count,
                    Some(duration_ms),
                );
            }
            ProcessOutcome::Failure { error: error_msg } => {
                // Distinguish skips from failures
                let is_skip = error_msg.contains("already seen")
                    || error_msg.contains("Duplicate")
                    || error_msg.contains("Near-duplicate")
                    || error_msg.contains("Disallowed by robots.txt");

                if is_skip {
                    urls_skipped += 1;
                    // Not counted as a fetch error (skip is intentional)
                    emit(
                        &event_tx,
                        ScrapeEvent::UrlSkipped {
                            job_id,
                            url: url_str.clone(),
                            reason: error_msg.clone(),
                        },
                    );
                    track_url(
                        &url_statuses,
                        url_str.clone(),
                        UrlStatus::Skipped,
                        scored_url.depth,
                        None,
                        Some(error_msg),
                        0,
                        Some(duration_ms),
                    );
                } else {
                    urls_failed += 1;
                    metrics.scrape_fetch_errors_total.inc();
                    emit(
                        &event_tx,
                        ScrapeEvent::UrlFailed {
                            job_id,
                            url: url_str.clone(),
                            error: error_msg.clone(),
                            duration_ms,
                        },
                    );
                    track_url(
                        &url_statuses,
                        url_str.clone(),
                        UrlStatus::Failed,
                        scored_url.depth,
                        None,
                        Some(error_msg),
                        0,
                        Some(duration_ms),
                    );
                }

                // Still add discovered URLs even on extraction failure
                if !discovered_urls.is_empty() {
                    coordinator
                        .add_discovered_urls(discovered_urls, scored_url.depth)
                        .await;
                }
            }
        }

        // Emit aggregate progress and update polling progress
        let coord_stats = coordinator.stats().await;
        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        let total_processed = urls_succeeded + urls_failed + urls_skipped;
        let rate = if elapsed_ms > 0 {
            Some(total_processed as f64 / (elapsed_ms as f64 / 1000.0))
        } else {
            None
        };

        emit(
            &event_tx,
            ScrapeEvent::Progress {
                job_id,
                urls_processed: total_processed,
                urls_succeeded,
                urls_failed,
                urls_skipped,
                urls_queued: coord_stats.queue_size,
                chunks_indexed,
                elapsed_ms,
                rate,
                eta_seconds: None,
            },
        );

        // Update polling-compatible progress
        if let Some(mut job) = jobs.get_mut(&job_id) {
            job.progress.stage = ProgressStage::Scraping;
            job.progress.current = total_processed;
            job.progress.total = Some((total_processed + coord_stats.queue_size as u64).max(total_processed));
            job.progress.rate = rate;
        }
    }

    // Commit changes
    if let Some(mut job) = jobs.get_mut(&job_id) {
        job.progress.stage = ProgressStage::Committing;
    }
    index_manager.commit()?;

    let duration_ms = start_time.elapsed().as_millis() as u64;

    info!(
        "Scrape job {} completed: {} docs, {} chunks, {} errors in {}ms",
        job_id, documents_processed, chunks_indexed, urls_failed, duration_ms
    );

    Ok(JobStats {
        documents_processed,
        chunks_indexed,
        duration_ms,
        errors: urls_failed as usize,
    })
}

