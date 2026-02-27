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
use crate::types::Document;

use super::index_manager::IndexManager;
use super::protocol::{ImportOptions, ImportSource, JobStats, Progress, ScrapeOptions};
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
}

/// Type of job
#[derive(Debug, Clone)]
pub enum JobType {
    Import { source: ImportSource },
    Scrape { urls: Vec<String> },
}

/// Job manager for tracking and controlling background jobs
pub struct JobManager {
    jobs: Arc<DashMap<Uuid, JobInfo>>,
    index_manager: Arc<IndexManager>,
    write_pipeline: Arc<WritePipeline>,
    config: Config,
    shutdown_tx: broadcast::Sender<()>,
}

impl JobManager {
    /// Create a new job manager
    pub fn new(
        index_manager: Arc<IndexManager>,
        write_pipeline: Arc<WritePipeline>,
        config: Config,
        shutdown_tx: broadcast::Sender<()>,
    ) -> Self {
        Self {
            jobs: Arc::new(DashMap::new()),
            index_manager,
            write_pipeline,
            config,
            shutdown_tx,
        }
    }

    /// Start an import job
    pub fn start_import(
        &self,
        source: ImportSource,
        options: ImportOptions,
    ) -> Uuid {
        let job_id = Uuid::new_v4();
        let (cancel_tx, cancel_rx) = oneshot::channel();

        let job_info = JobInfo {
            id: job_id,
            job_type: JobType::Import { source: source.clone() },
            state: JobState::Running,
            progress: Progress {
                job_id,
                stage: "starting".to_string(),
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
        };

        self.jobs.insert(job_id, job_info);

        // Spawn import task
        let jobs = Arc::clone(&self.jobs);
        let config = self.config.clone();
        let index_manager = self.index_manager.clone();
        let write_pipeline = self.write_pipeline.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

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
                        job.progress.stage = "completed".to_string();
                    }
                    Err(e) => {
                        if job.state != JobState::Cancelled {
                            job.state = JobState::Failed;
                            job.error = Some(e.to_string());
                            job.progress.stage = "failed".to_string();
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
        let job_id = Uuid::new_v4();
        let (cancel_tx, cancel_rx) = oneshot::channel();

        let job_info = JobInfo {
            id: job_id,
            job_type: JobType::Scrape { urls: urls.clone() },
            state: JobState::Running,
            progress: Progress {
                job_id,
                stage: "starting".to_string(),
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
        };

        self.jobs.insert(job_id, job_info);

        // Spawn scrape task
        let jobs = Arc::clone(&self.jobs);
        let config = self.config.clone();
        let index_manager = self.index_manager.clone();
        let write_pipeline = self.write_pipeline.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

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
            )
            .await;

            // Update job state
            if let Some(mut job) = jobs.get_mut(&job_id) {
                job.completed_at = Some(Instant::now());
                match result {
                    Ok(stats) => {
                        job.state = JobState::Completed;
                        job.stats = Some(stats);
                        job.progress.stage = "completed".to_string();
                    }
                    Err(e) => {
                        if job.state != JobState::Cancelled {
                            job.state = JobState::Failed;
                            job.error = Some(e.to_string());
                            job.progress.stage = "failed".to_string();
                        }
                    }
                }
            }
        });

        job_id
    }

    /// Cancel a job
    pub fn cancel(&self, job_id: Uuid) -> bool {
        if let Some(mut job) = self.jobs.get_mut(&job_id) {
            if job.state == JobState::Running {
                job.state = JobState::Cancelled;
                job.progress.stage = "cancelled".to_string();
                if let Ok(mut guard) = job.cancel_tx.lock() {
                    if let Some(cancel_tx) = guard.take() {
                        let _ = cancel_tx.send(());
                    }
                }
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
    fn update_progress(jobs: &DashMap<Uuid, JobInfo>, job_id: Uuid, stage: &str, current: u64, total: Option<u64>) {
        if let Some(mut job) = jobs.get_mut(&job_id) {
            job.progress.stage = stage.to_string();
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

            update_progress(&jobs, job_id, "parsing", 0, None);

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
                    update_progress(&jobs, job_id, "parsing", docs_parsed as u64, None);
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
                        update_progress(&jobs, job_id, "importing", documents_processed as u64, None);
                        last_progress_update = Instant::now();
                    }
                }

                // Yield to allow other tasks to run
                tokio::task::yield_now().await;
            }

            // Commit changes
            update_progress(&jobs, job_id, "committing", documents_processed as u64, None);
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

/// Run a scrape job
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
) -> anyhow::Result<JobStats> {
    info!("Starting scrape job {}: {} URLs", job_id, urls.len());

    let start_time = Instant::now();
    let mut documents_processed = 0usize;
    let mut chunks_indexed = 0usize;
    let mut errors = 0usize;
    let total_urls = urls.len();

    // Update progress helper
    fn update_progress(jobs: &DashMap<Uuid, JobInfo>, job_id: Uuid, total_urls: usize, stage: &str, current: u64) {
        if let Some(mut job) = jobs.get_mut(&job_id) {
            job.progress.stage = stage.to_string();
            job.progress.current = current;

            let elapsed = job.started_at.elapsed().as_secs_f64();
            if elapsed > 0.0 && current > 0 {
                let rate = current as f64 / elapsed;
                job.progress.rate = Some(rate);
                let remaining = total_urls.saturating_sub(current as usize);
                // ETA = remaining items / rate (items per second)
                let eta = (remaining as f64 / rate) as u64;
                job.progress.eta_seconds = Some(eta);
            } else if elapsed > 0.0 {
                job.progress.rate = Some(0.0);
            }
        }
    }

    let splitter = TextSplitter::new(config.chunking.clone());
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent("dindex/0.1 (scraper)")
        .build()?;

    for url in urls {
        // Check for cancellation or shutdown
        tokio::select! {
            biased;
            _ = &mut cancel_rx => {
                info!("Scrape job {} cancelled", job_id);
                return Err(anyhow::anyhow!("Job cancelled"));
            }
            _ = shutdown_rx.recv() => {
                info!("Scrape job {} stopped due to shutdown", job_id);
                return Err(anyhow::anyhow!("Shutdown"));
            }
            result = fetch_and_index_url(&client, &url, &splitter, &write_pipeline, job_id) => {
                match result {
                    Ok(chunk_count) => {
                        chunks_indexed += chunk_count;
                        documents_processed += 1;
                    }
                    Err(e) => {
                        warn!("Error scraping {}: {}", url, e);
                        errors += 1;
                    }
                }
                update_progress(&jobs, job_id, total_urls, "scraping", documents_processed as u64);
            }
        }

        // Respect politeness delay
        if options.delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(options.delay_ms)).await;
        }
    }

    // Commit changes
    if let Some(mut job) = jobs.get_mut(&job_id) {
        job.progress.stage = "committing".to_string();
    }
    index_manager.commit()?;

    let duration_ms = start_time.elapsed().as_millis() as u64;

    info!(
        "Scrape job {} completed: {} docs, {} chunks, {} errors in {}ms",
        job_id, documents_processed, chunks_indexed, errors, duration_ms
    );

    Ok(JobStats {
        documents_processed,
        chunks_indexed,
        duration_ms,
        errors,
    })
}

/// Fetch a URL and index its content
async fn fetch_and_index_url(
    client: &reqwest::Client,
    url: &str,
    splitter: &TextSplitter,
    write_pipeline: &WritePipeline,
    stream_id: Uuid,
) -> anyhow::Result<usize> {
    debug!("Fetching: {}", url);

    let response = client.get(url).send().await?;
    let html = response.text().await?;

    // Extract text from HTML (simple extraction)
    let text = extract_text_from_html(&html);

    if text.len() < 100 {
        return Ok(0);
    }

    // Create document and chunk
    let document = Document::new(text).with_url(url.to_string());
    let chunks = splitter.split_document(&document);
    let chunk_count = chunks.len();

    // Send chunks to write pipeline
    for chunk in chunks {
        write_pipeline
            .ingest(IngestItem::Chunk {
                stream_id,
                chunk,
                embedding: None,
            })
            .await?;
    }

    Ok(chunk_count)
}

/// Simple HTML text extraction
fn extract_text_from_html(html: &str) -> String {
    // Use scraper crate for proper HTML parsing
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);

    // Remove script and style elements
    let body_selector = Selector::parse("body").unwrap();
    let text_selector = Selector::parse("p, h1, h2, h3, h4, h5, h6, li, td, th, span, div").unwrap();

    let mut text = String::new();

    if let Some(body) = document.select(&body_selector).next() {
        for element in body.select(&text_selector) {
            let element_text: String = element.text().collect();
            let trimmed = element_text.trim();
            if !trimmed.is_empty() {
                text.push_str(trimmed);
                text.push('\n');
            }
        }
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_from_html() {
        let html = r#"
            <html>
            <body>
                <h1>Title</h1>
                <p>This is a paragraph.</p>
                <script>var x = 1;</script>
                <p>Another paragraph.</p>
            </body>
            </html>
        "#;

        let text = extract_text_from_html(html);
        assert!(text.contains("Title"));
        assert!(text.contains("This is a paragraph"));
        assert!(text.contains("Another paragraph"));
        assert!(!text.contains("var x"));
    }
}
