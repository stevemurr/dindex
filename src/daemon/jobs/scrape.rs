//! Scrape job execution

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};
use url::Url;
use uuid::Uuid;

use crate::chunking::TextSplitter;
use crate::config::Config;
use crate::scraping::coordinator::{
    ProcessOutcome, ProcessResult, ProcessedContent,
    ScrapingConfig as CoordinatorScrapingConfig, ScrapingCoordinator,
};

use crate::daemon::index_manager::IndexManager;
use crate::daemon::metrics::DaemonMetrics;
use crate::daemon::protocol::{JobStats, ProgressStage, ScrapeOptions};
use crate::daemon::scrape_events::{ScrapeEvent, UrlInfo, UrlSource, UrlStatus};
use crate::daemon::write_pipeline::{IngestItem, WritePipeline};
use super::JobInfo;

/// Maximum consecutive empty frontier iterations before declaring exhaustion
const MAX_EMPTY_ITERATIONS: u32 = 10;

/// Build a coordinator ScrapingConfig from the TOML config + per-job ScrapeOptions.
pub(super) fn build_coordinator_config(
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
pub(super) async fn run_scrape_job(
    job_id: Uuid,
    urls: Vec<String>,
    options: ScrapeOptions,
    config: Config,
    index_manager: Arc<IndexManager>,
    write_pipeline: Arc<WritePipeline>,
    jobs: Arc<DashMap<Uuid, JobInfo>>,
    mut cancel_rx: tokio::sync::oneshot::Receiver<()>,
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
