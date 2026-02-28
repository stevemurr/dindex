//! Import job execution

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::broadcast;
use tracing::{info, warn};
use uuid::Uuid;

use crate::chunking::TextSplitter;
use crate::config::Config;
use crate::import::{DumpSource, WikimediaSource};
use crate::types::Document;

use super::super::index_manager::IndexManager;
use super::super::protocol::{ImportSource, JobStats, ProgressStage};
use super::super::write_pipeline::{IngestItem, WritePipeline};
use super::JobInfo;

/// Run an import job
pub(super) async fn run_import_job(
    job_id: Uuid,
    source: ImportSource,
    _options: super::super::protocol::ImportOptions,
    config: Config,
    index_manager: Arc<IndexManager>,
    write_pipeline: Arc<WritePipeline>,
    jobs: Arc<DashMap<Uuid, JobInfo>>,
    mut cancel_rx: tokio::sync::oneshot::Receiver<()>,
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
