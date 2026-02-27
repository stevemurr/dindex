use anyhow::{Context, Result};
use dindex::{
    client::{self, ClientError},
    config::Config,
    daemon::protocol::{ImportOptions, ImportSource},
    embedding::init_embedding_engine,
    import::{DumpFormat, ImportCheckpoint, ImportCoordinatorBuilder, WikimediaSource},
};
use std::path::PathBuf;
use tracing::{info, warn};

pub async fn import_dump(
    config: Config,
    path: PathBuf,
    format: Option<DumpFormat>,
    batch_size: usize,
    resume: bool,
    checkpoint: Option<PathBuf>,
    no_dedup: bool,
    max_docs: Option<usize>,
    min_length: usize,
    quiet: bool,
) -> Result<()> {
    // Check file exists
    if !path.exists() {
        anyhow::bail!("Dump file not found: {}", path.display());
    }

    // Canonicalize to absolute path (daemon may run from different directory)
    let path = path.canonicalize()
        .with_context(|| format!("Failed to resolve path: {}", path.display()))?;

    // Detect or use specified format
    let detected_format = format
        .or_else(|| DumpFormat::detect(&path));

    let Some(dump_format) = detected_format else {
        anyhow::bail!(
            "Could not detect dump format for: {}. Specify format with --format",
            path.display()
        );
    };

    info!("Importing from: {} (format: {:?})", path.display(), dump_format);

    // Try daemon first for Wikimedia XML format
    if matches!(dump_format, DumpFormat::WikimediaXml) {
        let source = ImportSource::WikimediaXml {
            path: path.to_string_lossy().to_string(),
        };
        let options = ImportOptions {
            batch_size,
            deduplicate: !no_dedup,
            min_content_length: min_length,
            max_documents: max_docs,
        };

        match client::start_import(source, options).await {
            Ok(job_id) => {
                info!("Import job started via daemon: {}", job_id);
                if !quiet {
                    println!("Import job started: {}", job_id);
                    println!("Monitoring progress...");
                }

                // Poll for progress
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                    match client::job_progress(job_id).await {
                        Ok(progress) => {
                            if !quiet {
                                let rate_str = progress.rate
                                    .filter(|&r| r > 0.0)
                                    .map(|r| format!(" ({:.1} docs/s)", r))
                                    .unwrap_or_default();
                                let eta_str = progress.eta_seconds
                                    .filter(|&e| e > 0)
                                    .map(|e| {
                                        if e > 3600 {
                                            format!(" ETA: {}h {}m", e / 3600, (e % 3600) / 60)
                                        } else if e > 60 {
                                            format!(" ETA: {}m {}s", e / 60, e % 60)
                                        } else {
                                            format!(" ETA: {}s", e)
                                        }
                                    })
                                    .unwrap_or_default();

                                // Clear line and show progress
                                print!("\r\x1b[K{}: {} docs{}{}",
                                    progress.stage,
                                    progress.current,
                                    rate_str,
                                    eta_str
                                );
                                use std::io::Write;
                                std::io::stdout().flush().ok();
                            }

                            if progress.stage == "completed" || progress.stage == "failed" || progress.stage == "cancelled" {
                                if !quiet {
                                    println!("\nImport {}", progress.stage);
                                }
                                break;
                            }
                        }
                        Err(ClientError::JobNotFound(_)) => {
                            // Job completed and was cleaned up
                            if !quiet {
                                println!("\nImport completed");
                            }
                            break;
                        }
                        Err(e) => {
                            warn!("Error getting job progress: {}", e);
                        }
                    }
                }

                return Ok(());
            }
            Err(ClientError::DaemonNotRunning) => {
                info!("Daemon not running, using direct import");
            }
            Err(e) => {
                warn!("Daemon import failed: {}, falling back to direct import", e);
            }
        }
    }

    // Direct import fallback
    info!("Using direct import (daemon not available)");

    // Determine checkpoint path
    let checkpoint_path = checkpoint.or_else(|| {
        if config.bulk_import.enable_checkpoints {
            let filename = path.file_name()?.to_str()?;
            Some(config.bulk_import.checkpoint_dir.join(format!("{}.checkpoint", filename)))
        } else {
            None
        }
    });

    // Check for existing checkpoint if resuming
    let existing_checkpoint = if resume {
        checkpoint_path.as_ref().and_then(|p| {
            if p.exists() {
                ImportCheckpoint::load(p).ok()
            } else {
                None
            }
        })
    } else {
        None
    };

    if let Some(ref cp) = existing_checkpoint {
        info!(
            "Resuming from checkpoint: {} documents, {} bytes",
            cp.documents_processed, cp.byte_position
        );
    }

    // Initialize embedding engine
    let engine = init_embedding_engine(&config)
        .context("Failed to initialize embedding engine")?;

    // Create import coordinator
    let mut coordinator = ImportCoordinatorBuilder::new(&config.node.data_dir)
        .with_batch_size(batch_size)
        .with_dedup(!no_dedup)
        .with_min_content_length(min_length)
        .with_max_documents(max_docs)
        .with_embedding_engine(engine)
        .with_chunking_config(config.chunking.clone())
        .with_index_config(config.index.clone())
        .with_quiet(quiet)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create import coordinator: {}", e))?;

    if let Some(ref cp_path) = checkpoint_path {
        // Ensure checkpoint directory exists
        if let Some(parent) = cp_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // Run import based on format
    let stats = match dump_format {
        DumpFormat::WikimediaXml => {
            let mut source = WikimediaSource::open(&path)
                .map_err(|e| anyhow::anyhow!("Failed to open dump: {}", e))?;

            // Configure namespace filter from config
            source = source.with_namespaces(Some(config.bulk_import.wikipedia_namespaces.clone()));

            if let Some(cp) = existing_checkpoint {
                coordinator
                    .resume(source, &cp)
                    .map_err(|e| anyhow::anyhow!("Import failed: {}", e))?
            } else {
                coordinator
                    .import(source)
                    .map_err(|e| anyhow::anyhow!("Import failed: {}", e))?
            }
        }
        DumpFormat::Zim => {
            anyhow::bail!("ZIM format is not yet supported. Coming soon!");
        }
        DumpFormat::Warc => {
            anyhow::bail!("WARC format is not yet supported. Coming soon!");
        }
        DumpFormat::PlainText => {
            anyhow::bail!("Plain text import is not supported via this command. Use 'dindex index' instead.");
        }
    };

    // Print summary
    if !quiet {
        println!("\nImport Complete!");
        println!("================");
        println!("Documents imported: {}", stats.documents_imported);
        println!("Documents skipped:  {}", stats.documents_skipped);
        println!("Documents errored:  {}", stats.documents_errored);
        println!("Chunks created:     {}", stats.chunks_created);
        println!("Processing rate:    {:.1} docs/sec", stats.docs_per_second);
        println!("Elapsed time:       {:.1}s", stats.elapsed_seconds);
        println!("\nIndex saved to: {}", config.node.data_dir.display());
    }

    Ok(())
}

pub async fn show_import_status(checkpoint_path: PathBuf) -> Result<()> {
    if !checkpoint_path.exists() {
        anyhow::bail!("Checkpoint file not found: {}", checkpoint_path.display());
    }

    let checkpoint = ImportCheckpoint::load(&checkpoint_path)
        .map_err(|e| anyhow::anyhow!("Failed to load checkpoint: {}", e))?;

    println!("\nImport Checkpoint Status");
    println!("========================");
    println!("Source file:         {}", checkpoint.source_path.display());
    println!("Byte position:       {} MB", checkpoint.byte_position / 1_000_000);
    println!("Documents processed: {}", checkpoint.documents_processed);
    println!("Documents imported:  {}", checkpoint.documents_imported);
    println!("Timestamp:           {}", checkpoint.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
    println!("\nTo resume import, run:");
    println!(
        "  dindex import {} --resume --checkpoint {}",
        checkpoint.source_path.display(),
        checkpoint_path.display()
    );

    Ok(())
}
