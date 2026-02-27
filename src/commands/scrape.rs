use anyhow::{Context, Result};
use dindex::{
    chunking::TextSplitter,
    client::{self, ClientError},
    config::Config,
    daemon::protocol::ScrapeOptions,
    embedding::init_embedding_engine,
    index::{ChunkStorage, VectorIndex},
    retrieval::{Bm25Index, HybridIndexer},
    util::truncate_for_display,
    scraping::{
        coordinator::{ScrapingConfig as ScrapingCoordConfig, ScrapingCoordinator},
        extractor::ExtractorConfig,
        fetcher::FetchConfig,
        politeness::PolitenessConfig,
    },
};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};
use url::Url;

pub async fn scrape_urls(
    config: Config,
    url_strings: Vec<String>,
    max_depth: u8,
    stay_on_domain: bool,
    max_pages: usize,
    delay_ms: u64,
    should_index: bool,
) -> Result<()> {
    info!("Starting web scraper...");

    // Parse URLs
    let seeds: Vec<Url> = url_strings
        .iter()
        .filter_map(|s| {
            Url::parse(s)
                .or_else(|_| Url::parse(&format!("https://{}", s)))
                .ok()
        })
        .collect();

    if seeds.is_empty() {
        anyhow::bail!("No valid URLs provided");
    }

    info!("Seed URLs: {:?}", seeds.iter().map(|u| u.as_str()).collect::<Vec<_>>());

    // Try daemon first if indexing is requested
    if should_index {
        let options = ScrapeOptions {
            max_depth,
            stay_on_domain,
            delay_ms,
            max_pages,
        };

        match client::start_scrape(url_strings.clone(), options).await {
            Ok(job_id) => {
                info!("Scrape job started via daemon: {}", job_id);
                println!("Scrape job started: {}", job_id);
                println!("Monitoring progress...");

                // Poll for progress
                loop {
                    tokio::time::sleep(Duration::from_secs(2)).await;

                    match client::job_progress(job_id).await {
                        Ok(progress) => {
                            let rate_str = progress.rate.map(|r| format!("{:.1} pages/s", r)).unwrap_or_default();
                            let eta_str = progress.eta_seconds.map(|e| format!("ETA: {}s", e)).unwrap_or_default();
                            print!("\r{}: {}/{} {} {}",
                                progress.stage,
                                progress.current,
                                progress.total.unwrap_or(0),
                                rate_str,
                                eta_str
                            );
                            use std::io::Write;
                            std::io::stdout().flush().ok();

                            if progress.stage == "completed" || progress.stage == "failed" || progress.stage == "cancelled" {
                                println!("\nScrape {}", progress.stage);
                                break;
                            }
                        }
                        Err(ClientError::JobNotFound(_)) => {
                            println!("\nScrape completed");
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
                info!("Daemon not running, using direct scraping");
            }
            Err(e) => {
                warn!("Daemon scrape failed: {}, falling back to direct scraping", e);
            }
        }
    }

    // Direct scraping fallback
    info!("Using direct scraping (daemon not available or not indexing)");

    println!("\nScrape Configuration:");
    println!("  Seeds: {} URLs", seeds.len());
    println!("  Max depth: {}", max_depth);
    println!("  Max pages: {}", max_pages);
    println!("  Indexing: {}", if should_index { "enabled" } else { "disabled" });

    // Initialize embedding engine if indexing
    let embedding_engine = if should_index {
        println!("  Model: {}", config.embedding.model_name);
        Some(
            init_embedding_engine(&config)
                .context("Failed to initialize embedding engine")?,
        )
    } else {
        None
    };
    println!();

    // Create scraping config
    let scraping_config = ScrapingCoordConfig {
        enabled: true,
        max_concurrent_fetches: config.scraping.max_concurrent_fetches,
        max_depth,
        stay_on_domain,
        include_patterns: config.scraping.include_patterns.clone(),
        exclude_patterns: config.scraping.exclude_patterns.clone(),
        max_pages_per_domain: max_pages,
        scrape_interval: Duration::from_millis(100),
        politeness: PolitenessConfig {
            user_agent: config.scraping.user_agent.clone(),
            default_delay: Duration::from_millis(delay_ms),
            min_delay: Duration::from_millis(delay_ms / 2),
            max_delay: Duration::from_secs(30),
            cache_size: 10000,
            request_timeout: Duration::from_secs(config.scraping.request_timeout_secs),
        },
        fetch: FetchConfig {
            user_agent: config.scraping.user_agent.clone(),
            timeout: Duration::from_secs(config.scraping.request_timeout_secs),
            connect_timeout: Duration::from_secs(10),
            max_content_size: 10 * 1024 * 1024,
            max_redirects: 10,
            min_text_ratio: 0.1,
            enable_js_rendering: config.scraping.enable_js_rendering,
            connections_per_host: 10,
        },
        extractor: ExtractorConfig::default(),
    };

    // Create coordinator
    let peer_id = format!("scraper_{}", uuid::Uuid::new_v4());
    let mut coordinator = ScrapingCoordinator::new(scraping_config, peer_id)?;

    // Add seed URLs
    coordinator.add_seeds(seeds).await;

    // Initialize indexing components if needed
    let (indexer, vector_index, chunk_storage) = if should_index {
        let vi = Arc::new(VectorIndex::new(config.embedding.dimensions, &config.index)?);
        let bm25_path = config.node.data_dir.join("bm25");
        let bm25_index = Arc::new(Bm25Index::new(&bm25_path)?);
        let cs = Arc::new(ChunkStorage::new(&config.node.data_dir)?);
        let idx = HybridIndexer::new(vi.clone(), bm25_index, cs.clone());
        (Some(idx), Some(vi), Some(cs))
    } else {
        (None, None, None)
    };

    let splitter = TextSplitter::new(config.chunking.clone());

    // Process URLs
    let mut pages_scraped = 0;
    let mut pages_indexed = 0;
    let mut chunks_created = 0;
    let mut embedding_errors = 0;
    let mut index_errors = 0;

    println!("Scraping progress:");
    println!("==================");

    while pages_scraped < max_pages {
        let next_url = coordinator.get_next_url().await;

        match next_url {
            Some(scored_url) => {
                let result = coordinator.process_url(&scored_url.url).await;

                if result.success {
                    pages_scraped += 1;

                    let word_count = result.content.as_ref().map(|c| c.word_count).unwrap_or(0);
                    let urls_found = result.discovered_urls.len();

                    println!(
                        "[{}/{}] {} - {} words, {} links found",
                        pages_scraped,
                        max_pages,
                        truncate_for_display(result.url.as_str(), 60),
                        word_count,
                        urls_found
                    );

                    // Add discovered URLs
                    coordinator
                        .add_discovered_urls(result.discovered_urls, scored_url.depth)
                        .await;

                    // Index content if requested
                    if let (Some(ref indexer), Some(ref engine), Some(content), Some(metadata)) =
                        (&indexer, &embedding_engine, result.content, result.metadata)
                    {
                        let doc = ScrapingCoordinator::to_document(&result.url, &content, &metadata);
                        let chunks = splitter.split_document(&doc);

                        if !chunks.is_empty() {
                            // Extract texts for batch embedding
                            let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
                            let num_chunks = texts.len();

                            // Generate real embeddings
                            match engine.embed_batch(&texts) {
                                Ok(embeddings) => {
                                    let chunks_with_embeddings: Vec<_> = chunks
                                        .into_iter()
                                        .zip(embeddings.into_iter())
                                        .collect();

                                    match indexer.index_batch(&chunks_with_embeddings) {
                                        Ok(keys) => {
                                            pages_indexed += 1;
                                            chunks_created += keys.len();
                                            tracing::debug!("Indexed {} chunks", keys.len());
                                        }
                                        Err(e) => {
                                            index_errors += 1;
                                            tracing::warn!("Failed to index chunks: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    embedding_errors += 1;
                                    tracing::warn!(
                                        "Embedding failed for {} ({} chunks): {}",
                                        truncate_for_display(result.url.as_str(), 40),
                                        num_chunks,
                                        e
                                    );
                                }
                            }
                        }
                    }
                } else {
                    tracing::debug!(
                        "Failed: {} - {}",
                        result.url,
                        result.error.unwrap_or_default()
                    );
                }
            }
            None => {
                // No URLs ready, check if we're done
                let stats = coordinator.stats().await;
                if stats.queue_size == 0 {
                    info!("No more URLs to process");
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }

    // Save index if we indexed content
    if let (Some(vi), Some(_cs)) = (vector_index, chunk_storage) {
        let index_path = config.node.data_dir.join("vector.index");
        vi.save(&index_path)?;
        info!("Saved vector index to {}", index_path.display());
    }

    // Print final stats
    let stats = coordinator.stats().await;
    println!("\nScraping complete!");
    println!("==================");
    println!("Pages scraped: {}", pages_scraped);
    println!("Pages indexed: {}", pages_indexed);
    println!("Chunks created: {}", chunks_created);
    println!("URLs discovered: {}", stats.urls_discovered);
    println!("Duplicates skipped: {}", stats.duplicates_skipped);
    println!("Queue remaining: {}", stats.queue_size);
    println!("Avg processing time: {:.1}ms", stats.avg_processing_time_ms);

    if embedding_errors > 0 || index_errors > 0 {
        println!("\nErrors:");
        if embedding_errors > 0 {
            println!("  Embedding failures: {}", embedding_errors);
        }
        if index_errors > 0 {
            println!("  Index failures: {}", index_errors);
        }
    }

    Ok(())
}

pub async fn show_scrape_stats(config: Config) -> Result<()> {
    println!("\nScraping Configuration:");
    println!("=======================");
    println!("Enabled: {}", config.scraping.enabled);
    println!("Max concurrent fetches: {}", config.scraping.max_concurrent_fetches);
    println!("Max depth: {}", config.scraping.max_depth);
    println!("Stay on domain: {}", config.scraping.stay_on_domain);
    println!("Politeness delay: {}ms", config.scraping.politeness_delay_ms);
    println!("Request timeout: {}s", config.scraping.request_timeout_secs);
    println!("User agent: {}", config.scraping.user_agent);
    println!("JS rendering: {}", config.scraping.enable_js_rendering);
    println!("Max pages per domain: {}", config.scraping.max_pages_per_domain);
    println!();
    println!("Exclude patterns: {:?}", config.scraping.exclude_patterns);
    println!("Include patterns: {:?}", config.scraping.include_patterns);

    Ok(())
}

