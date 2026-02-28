//! Background Job Management
//!
//! Handles long-running import and scrape jobs with progress tracking.

mod import;
mod scrape;

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::{broadcast, oneshot};
use uuid::Uuid;

use crate::config::Config;

use super::index_manager::IndexManager;
use super::metrics::DaemonMetrics;
use super::protocol::{ImportOptions, ImportSource, Progress, ProgressStage, ScrapeOptions};
use super::scrape_events::{JobCompletionStatus, ScrapeEvent, UrlInfo};
use super::write_pipeline::WritePipeline;

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
    pub stats: Option<super::protocol::JobStats>,
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
            let result = import::run_import_job(
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
            let result = scrape::run_scrape_job(
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
