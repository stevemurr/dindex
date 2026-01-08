//! DIndex: Decentralized Semantic Search Index for LLM Consumption
//!
//! A federated semantic search system optimized for LLM consumption.

use anyhow::Result;
use clap::{Parser, Subcommand};
use dindex::{
    chunking::TextSplitter,
    config::Config,
    content::{extract_from_bytes, extract_from_path, ContentType},
    embedding::ModelManager,
    import::{DumpFormat, ImportCheckpoint, ImportCoordinatorBuilder, WikimediaSource},
    index::{ChunkStorage, DocumentRegistry, VectorIndex},
    network::NetworkNode,
    retrieval::{Bm25Index, HybridIndexer, HybridRetriever},
    scraping::{
        coordinator::{ScrapingConfig as ScrapingCoordConfig, ScrapingCoordinator},
        extractor::ExtractorConfig,
        fetcher::FetchConfig,
        politeness::PolitenessConfig,
    },
    types::{Document, DocumentIdentity, Query},
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use url::Url;

#[derive(Parser)]
#[command(name = "dindex")]
#[command(about = "Decentralized semantic search index for LLM consumption")]
#[command(version)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Data directory
    #[arg(short, long)]
    data_dir: Option<PathBuf>,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the DIndex node
    Start {
        /// Listen address
        #[arg(short, long)]
        listen: Option<String>,

        /// Bootstrap peers
        #[arg(short, long)]
        bootstrap: Vec<String>,
    },

    /// Index a document or directory
    Index {
        /// Path to document or directory
        path: PathBuf,

        /// Document title
        #[arg(short, long)]
        title: Option<String>,

        /// Document URL
        #[arg(short, long)]
        url: Option<String>,
    },

    /// Search the index
    Search {
        /// Search query
        query: String,

        /// Number of results
        #[arg(short, long, default_value = "10")]
        top_k: usize,

        /// Output format (json, text)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Show index statistics
    Stats,

    /// Download embedding model
    Download {
        /// Model name
        #[arg(default_value = "nomic-embed-text-v1.5")]
        model: String,
    },

    /// Export index for LLM consumption
    Export {
        /// Output path
        output: PathBuf,

        /// Format (json, jsonl)
        #[arg(short, long, default_value = "jsonl")]
        format: String,
    },

    /// Initialize a new DIndex configuration
    Init {
        /// Output directory
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Scrape URLs and index content
    Scrape {
        /// Seed URLs to start scraping from
        #[arg(required = true)]
        urls: Vec<String>,

        /// Maximum crawl depth
        #[arg(short, long, default_value = "2")]
        depth: u8,

        /// Stay within seed domains only
        #[arg(short, long)]
        stay_on_domain: bool,

        /// Maximum pages to scrape
        #[arg(short, long, default_value = "100")]
        max_pages: usize,

        /// Delay between requests in milliseconds
        #[arg(long, default_value = "1000")]
        delay_ms: u64,

        /// Index scraped content
        #[arg(long, default_value = "true")]
        index: bool,
    },

    /// Show scraping statistics
    ScrapeStats,

    /// Import content from offline dumps (Wikipedia, ZIM, etc.)
    Import {
        /// Path to dump file
        #[arg(required = true)]
        path: PathBuf,

        /// Dump format (auto-detected if not specified)
        #[arg(short, long, value_enum)]
        format: Option<CliDumpFormat>,

        /// Batch size for indexing
        #[arg(long, default_value = "100")]
        batch_size: usize,

        /// Resume from checkpoint
        #[arg(long)]
        resume: bool,

        /// Checkpoint file path
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        /// Skip content deduplication
        #[arg(long)]
        no_dedup: bool,

        /// Maximum documents to import
        #[arg(long)]
        max_docs: Option<usize>,

        /// Minimum content length (skip shorter documents)
        #[arg(long, default_value = "100")]
        min_length: usize,

        /// Quiet mode (no progress output)
        #[arg(short, long)]
        quiet: bool,
    },

    /// Show import checkpoint status
    ImportStatus {
        /// Path to checkpoint file
        checkpoint: PathBuf,
    },

    /// Migrate existing documents to unified registry
    MigrateRegistry {
        /// Only show what would be migrated (dry run)
        #[arg(long)]
        dry_run: bool,

        /// Skip documents that already exist in registry
        #[arg(long)]
        skip_existing: bool,
    },

    /// Show document registry statistics
    RegistryStats,
}

/// CLI dump format enum (mirrors DumpFormat but with clap support)
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliDumpFormat {
    /// Wikimedia XML dump (.xml.bz2)
    WikimediaXml,
    /// ZIM file (Kiwix format)
    Zim,
    /// WARC web archive
    Warc,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = match cli.verbose {
        0 => Level::INFO,
        1 => Level::DEBUG,
        _ => Level::TRACE,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Load or create config
    let mut config = if cli.config.exists() {
        let content = std::fs::read_to_string(&cli.config)?;
        toml::from_str(&content).unwrap_or_default()
    } else {
        Config::default()
    };

    // Override data dir if specified
    if let Some(data_dir) = cli.data_dir {
        config.node.data_dir = data_dir;
    }

    // Ensure data directory exists
    std::fs::create_dir_all(&config.node.data_dir)?;

    match cli.command {
        Commands::Start { listen, bootstrap } => {
            start_node(config, listen, bootstrap).await
        }
        Commands::Index { path, title, url } => {
            index_document(config, path, title, url).await
        }
        Commands::Search { query, top_k, format } => {
            search_index(config, query, top_k, format).await
        }
        Commands::Stats => {
            show_stats(config).await
        }
        Commands::Download { model } => {
            download_model(config, model).await
        }
        Commands::Export { output, format } => {
            export_index(config, output, format).await
        }
        Commands::Init { path } => {
            init_config(path).await
        }
        Commands::Scrape {
            urls,
            depth,
            stay_on_domain,
            max_pages,
            delay_ms,
            index,
        } => {
            scrape_urls(config, urls, depth, stay_on_domain, max_pages, delay_ms, index).await
        }
        Commands::ScrapeStats => {
            show_scrape_stats(config).await
        }
        Commands::Import {
            path,
            format,
            batch_size,
            resume,
            checkpoint,
            no_dedup,
            max_docs,
            min_length,
            quiet,
        } => {
            import_dump(
                config, path, format, batch_size, resume, checkpoint, no_dedup, max_docs, min_length, quiet,
            )
            .await
        }
        Commands::ImportStatus { checkpoint } => {
            show_import_status(checkpoint).await
        }
        Commands::MigrateRegistry { dry_run, skip_existing } => {
            migrate_registry(config, dry_run, skip_existing).await
        }
        Commands::RegistryStats => {
            show_registry_stats(config).await
        }
    }
}

async fn start_node(
    mut config: Config,
    listen: Option<String>,
    bootstrap: Vec<String>,
) -> Result<()> {
    if let Some(addr) = listen {
        config.node.listen_addr = addr;
    }
    config.node.bootstrap_peers.extend(bootstrap);

    info!("Starting DIndex node...");
    info!("Data directory: {}", config.node.data_dir.display());

    // Initialize components
    let (node, handle, mut event_rx) = NetworkNode::new(&config.node).await?;

    info!("Node started with peer ID: {}", handle.local_peer_id);

    // Run the node
    let node_config = config.node.clone();
    tokio::spawn(async move {
        if let Err(e) = node.run(&node_config).await {
            tracing::error!("Node error: {}", e);
        }
    });

    // Handle events
    while let Some(event) = event_rx.recv().await {
        info!("Network event: {:?}", event);
    }

    Ok(())
}

async fn index_document(
    config: Config,
    path: PathBuf,
    title: Option<String>,
    url: Option<String>,
) -> Result<()> {
    let path_str = path.to_string_lossy();

    // Check if path is a URL
    let (content, extracted_title, source_url) = if path_str.starts_with("http://")
        || path_str.starts_with("https://")
    {
        info!("Fetching URL: {}", path_str);

        // Fetch content from URL
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("dindex/0.1")
            .build()?;

        let response = client.get(path_str.as_ref()).send().await?;

        // Get content type from headers or URL
        let content_type_header = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        let content_type = if !content_type_header.is_empty() {
            ContentType::from_mime(content_type_header)
        } else {
            ContentType::from_url(&path_str)
        };

        let bytes = response.bytes().await?;
        info!(
            "Fetched {} bytes, content type: {:?}",
            bytes.len(),
            content_type
        );

        let extracted = extract_from_bytes(&bytes, content_type)?;
        (extracted.content, extracted.title, Some(path_str.to_string()))
    } else if path.is_file() {
        info!("Indexing file: {}", path.display());

        let extracted = extract_from_path(&path)?;
        (extracted.content, extracted.title, url)
    } else if path.is_dir() {
        info!("Indexing directory: {}", path.display());

        // Index all files in directory
        let mut combined = String::new();
        for entry in walkdir::WalkDir::new(&path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            match extract_from_path(entry.path()) {
                Ok(extracted) => {
                    combined.push_str(&extracted.content);
                    combined.push_str("\n\n");
                }
                Err(e) => {
                    tracing::warn!("Failed to extract {}: {}", entry.path().display(), e);
                }
            }
        }
        (combined, None, url)
    } else {
        anyhow::bail!("Path does not exist: {}", path.display());
    };

    // Create document
    let mut doc = Document::new(content);
    if let Some(t) = title {
        doc = doc.with_title(t);
    } else if let Some(t) = extracted_title {
        doc = doc.with_title(t);
    } else {
        doc = doc.with_title(path.file_name().unwrap_or_default().to_string_lossy());
    }
    if let Some(u) = source_url {
        doc = doc.with_url(u);
    }

    // Split into chunks
    let splitter = TextSplitter::new(config.chunking.clone());
    let chunks = splitter.split_document(&doc);

    info!("Created {} chunks", chunks.len());

    // Initialize index components
    let index_path = config.node.data_dir.join("vector.index");
    let vector_index = Arc::new(VectorIndex::new(
        config.embedding.dimensions,
        &config.index,
    )?);

    let bm25_path = config.node.data_dir.join("bm25");
    let bm25_index = Arc::new(Bm25Index::new(&bm25_path)?);

    let chunk_storage = Arc::new(ChunkStorage::new(&config.node.data_dir)?);

    let indexer = HybridIndexer::new(
        vector_index.clone(),
        bm25_index.clone(),
        chunk_storage.clone(),
    );

    // Generate dummy embeddings (in production, use embedding engine)
    // For now, create random embeddings for testing
    let chunks_with_embeddings: Vec<_> = chunks
        .into_iter()
        .map(|c| {
            let embedding: Vec<f32> = (0..config.embedding.dimensions)
                .map(|i| {
                    // Create deterministic pseudo-embedding based on content hash
                    let hash = xxhash_rust::xxh3::xxh3_64(c.content.as_bytes());
                    ((hash.wrapping_add(i as u64) % 1000) as f32 / 500.0) - 1.0
                })
                .collect();
            (c, embedding)
        })
        .collect();

    // Index chunks
    let keys = indexer.index_batch(&chunks_with_embeddings)?;
    info!("Indexed {} chunks with keys {:?}", keys.len(), &keys[..keys.len().min(5)]);

    // Save index
    vector_index.save(&index_path)?;
    indexer.save()?;

    info!("Index saved to {}", config.node.data_dir.display());

    Ok(())
}

async fn search_index(
    config: Config,
    query_text: String,
    top_k: usize,
    format: String,
) -> Result<()> {
    info!("Searching for: {}", query_text);

    // Load index
    let index_path = config.node.data_dir.join("vector.index");
    let vector_index = if index_path.exists() {
        Arc::new(VectorIndex::load(&index_path, &config.index)?)
    } else {
        Arc::new(VectorIndex::new(config.embedding.dimensions, &config.index)?)
    };

    let bm25_path = config.node.data_dir.join("bm25");
    let bm25_index = Arc::new(Bm25Index::new(&bm25_path)?);

    let chunk_storage = Arc::new(ChunkStorage::load(&config.node.data_dir)?);

    let retriever = Arc::new(HybridRetriever::new(
        vector_index,
        bm25_index,
        chunk_storage,
        None,
        config.retrieval.clone(),
    ));

    // Create query
    let query = Query::new(&query_text, top_k);

    // Generate query embedding (dummy for now)
    let query_embedding: Vec<f32> = (0..config.embedding.dimensions)
        .map(|i| {
            let hash = xxhash_rust::xxh3::xxh3_64(query_text.as_bytes());
            ((hash.wrapping_add(i as u64) % 1000) as f32 / 500.0) - 1.0
        })
        .collect();

    // Execute search
    let results = retriever.search(&query, Some(&query_embedding))?;

    // Output results
    match format.as_str() {
        "json" => {
            let json = serde_json::to_string_pretty(&results)?;
            println!("{}", json);
        }
        _ => {
            println!("\nSearch Results ({} found):\n", results.len());
            for (i, result) in results.iter().enumerate() {
                println!("{}. [Score: {:.4}]", i + 1, result.relevance_score);
                println!("   ID: {}", result.chunk.metadata.chunk_id);
                if let Some(title) = &result.chunk.metadata.source_title {
                    println!("   Title: {}", title);
                }
                println!("   Content: {}...", truncate_content(&result.chunk.content, 200));
                println!("   Matched by: {:?}", result.matched_by);
                println!();
            }
        }
    }

    Ok(())
}

async fn show_stats(config: Config) -> Result<()> {
    info!("Loading index statistics...");

    let chunk_storage = ChunkStorage::load(&config.node.data_dir)?;

    println!("\nDIndex Statistics:");
    println!("==================");
    println!("Data directory: {}", config.node.data_dir.display());
    println!("Total chunks: {}", chunk_storage.len());
    println!("Embedding dimensions: {}", config.embedding.dimensions);
    println!("Model: {}", config.embedding.model_name);

    // Check index files
    let index_path = config.node.data_dir.join("vector.index");
    if index_path.exists() {
        let metadata = std::fs::metadata(&index_path)?;
        println!("Vector index size: {} bytes", metadata.len());
    }

    Ok(())
}

async fn download_model(config: Config, model: String) -> Result<()> {
    info!("Downloading model: {}", model);

    let cache_dir = config.node.data_dir.join("models");
    let manager = ModelManager::new(&cache_dir)?;

    let (model_path, tokenizer_path) = manager.ensure_model(&model).await?;

    println!("\nModel downloaded successfully!");
    println!("Model path: {}", model_path.display());
    println!("Tokenizer path: {}", tokenizer_path.display());

    Ok(())
}

async fn export_index(config: Config, output: PathBuf, format: String) -> Result<()> {
    info!("Exporting index to: {}", output.display());

    let chunk_storage = ChunkStorage::load(&config.node.data_dir)?;
    let chunks = chunk_storage.all_embeddings();
    let chunk_count = chunks.len();

    match format.as_str() {
        "jsonl" => {
            use std::io::Write;
            let mut file = std::fs::File::create(&output)?;

            for (chunk_id, _embedding) in &chunks {
                if let Some(stored) = chunk_storage.get(chunk_id) {
                    let json = serde_json::to_string(&stored.chunk)?;
                    writeln!(file, "{}", json)?;
                }
            }
        }
        "json" => {
            let all_chunks: Vec<_> = chunks
                .iter()
                .filter_map(|(id, _)| chunk_storage.get(id).map(|s| s.chunk))
                .collect();
            let json = serde_json::to_string_pretty(&all_chunks)?;
            std::fs::write(&output, json)?;
        }
        _ => {
            anyhow::bail!("Unknown format: {}", format);
        }
    }

    println!("Exported {} chunks to {}", chunk_count, output.display());

    Ok(())
}

async fn init_config(path: PathBuf) -> Result<()> {
    let config = Config::default();
    let config_path = path.join("dindex.toml");

    // Generate TOML config
    let toml_content = format!(
        r#"# DIndex Configuration

[node]
listen_addr = "{}"
data_dir = ".dindex"
enable_mdns = true
replication_factor = 3
query_timeout_secs = 10

[embedding]
model_name = "{}"
dimensions = {}
truncated_dimensions = {}
max_sequence_length = {}
quantize_int8 = true

[index]
hnsw_m = {}
hnsw_ef_construction = {}
hnsw_ef_search = {}
memory_mapped = true
max_capacity = {}

[chunking]
chunk_size = {}
overlap_fraction = {}
min_chunk_size = {}
max_chunk_size = {}

[retrieval]
enable_dense = true
enable_bm25 = true
rrf_k = {}
candidate_count = {}
enable_reranking = true

[routing]
num_centroids = {}
lsh_bits = {}
lsh_num_hashes = {}
bloom_bits_per_item = {}
candidate_nodes = {}
"#,
        config.node.listen_addr,
        config.embedding.model_name,
        config.embedding.dimensions,
        config.embedding.truncated_dimensions,
        config.embedding.max_sequence_length,
        config.index.hnsw_m,
        config.index.hnsw_ef_construction,
        config.index.hnsw_ef_search,
        config.index.max_capacity,
        config.chunking.chunk_size,
        config.chunking.overlap_fraction,
        config.chunking.min_chunk_size,
        config.chunking.max_chunk_size,
        config.retrieval.rrf_k,
        config.retrieval.candidate_count,
        config.routing.num_centroids,
        config.routing.lsh_bits,
        config.routing.lsh_num_hashes,
        config.routing.bloom_bits_per_item,
        config.routing.candidate_nodes,
    );

    std::fs::write(&config_path, toml_content)?;
    println!("Created configuration file: {}", config_path.display());

    // Create data directory
    let data_dir = path.join(".dindex");
    std::fs::create_dir_all(&data_dir)?;
    println!("Created data directory: {}", data_dir.display());

    Ok(())
}

fn truncate_content(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() > max_len {
        // Find a valid char boundary at or before max_len
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        s[..end].to_string()
    } else {
        s
    }
}

async fn scrape_urls(
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
        let index_path = config.node.data_dir.join("vector.index");
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

    println!("\nScraping progress:");
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
                        truncate_content(result.url.as_str(), 60),
                        word_count,
                        urls_found
                    );

                    // Add discovered URLs
                    coordinator
                        .add_discovered_urls(result.discovered_urls, scored_url.depth)
                        .await;

                    // Index content if requested
                    if let (Some(ref indexer), Some(content), Some(metadata)) =
                        (&indexer, result.content, result.metadata)
                    {
                        let doc = ScrapingCoordinator::to_document(&result.url, &content, &metadata);
                        let chunks = splitter.split_document(&doc);

                        if !chunks.is_empty() {
                            let chunks_with_embeddings: Vec<_> = chunks
                                .into_iter()
                                .map(|c| {
                                    let embedding: Vec<f32> = (0..config.embedding.dimensions)
                                        .map(|i| {
                                            let hash = xxhash_rust::xxh3::xxh3_64(c.content.as_bytes());
                                            ((hash.wrapping_add(i as u64) % 1000) as f32 / 500.0) - 1.0
                                        })
                                        .collect();
                                    (c, embedding)
                                })
                                .collect();

                            if let Ok(keys) = indexer.index_batch(&chunks_with_embeddings) {
                                pages_indexed += 1;
                                tracing::debug!("Indexed {} chunks", keys.len());
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
    if let (Some(vi), Some(cs)) = (vector_index, chunk_storage) {
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
    println!("URLs discovered: {}", stats.urls_discovered);
    println!("Duplicates skipped: {}", stats.duplicates_skipped);
    println!("Queue remaining: {}", stats.queue_size);
    println!("Avg processing time: {:.1}ms", stats.avg_processing_time_ms);

    Ok(())
}

async fn show_scrape_stats(config: Config) -> Result<()> {
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

// Add walkdir and toml to dependencies
use walkdir;

async fn import_dump(
    config: Config,
    path: PathBuf,
    format: Option<CliDumpFormat>,
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

    // Detect or use specified format
    let detected_format = format
        .map(|f| match f {
            CliDumpFormat::WikimediaXml => DumpFormat::WikimediaXml,
            CliDumpFormat::Zim => DumpFormat::Zim,
            CliDumpFormat::Warc => DumpFormat::Warc,
        })
        .or_else(|| DumpFormat::detect(&path));

    let Some(dump_format) = detected_format else {
        anyhow::bail!(
            "Could not detect dump format for: {}. Specify format with --format",
            path.display()
        );
    };

    info!("Importing from: {} (format: {:?})", path.display(), dump_format);

    // Determine checkpoint path
    let checkpoint_path = checkpoint.or_else(|| {
        if config.import.enable_checkpoints {
            let filename = path.file_name()?.to_str()?;
            Some(config.import.checkpoint_dir.join(format!("{}.checkpoint", filename)))
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

    // Create import coordinator
    let mut coordinator = ImportCoordinatorBuilder::new(&config.node.data_dir)
        .with_batch_size(batch_size)
        .with_dedup(!no_dedup)
        .with_min_content_length(min_length)
        .with_max_documents(max_docs)
        .with_embedding_dims(config.embedding.dimensions)
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
            source = source.with_namespaces(Some(config.import.wikipedia_namespaces.clone()));

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

async fn show_import_status(checkpoint_path: PathBuf) -> Result<()> {
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

async fn migrate_registry(config: Config, dry_run: bool, skip_existing: bool) -> Result<()> {
    info!("Migrating existing documents to unified registry...");

    // Load or create the registry
    let registry = DocumentRegistry::load(
        &config.node.data_dir,
        config.dedup.simhash_distance_threshold,
    )?;

    // Load chunk storage
    let chunk_storage = ChunkStorage::load(&config.node.data_dir)?;

    // Get all existing chunks
    let chunks = chunk_storage.chunk_ids();
    if chunks.is_empty() {
        println!("No existing chunks found to migrate.");
        return Ok(());
    }

    println!("Found {} existing chunks", chunks.len());

    // Group chunks by document ID
    let mut doc_chunks: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for chunk_id in &chunks {
        if let Some(stored) = chunk_storage.get(chunk_id) {
            let doc_id = &stored.chunk.metadata.document_id;
            doc_chunks
                .entry(doc_id.clone())
                .or_default()
                .push(chunk_id.clone());
        }
    }

    println!("Found {} unique documents", doc_chunks.len());

    if dry_run {
        println!("\nDry run - no changes will be made");
        println!("==================================");
    }

    let mut migrated = 0;
    let mut skipped = 0;
    let mut duplicates = 0;

    for (doc_id, chunk_ids) in &doc_chunks {
        // Get content from first chunk to compute identity
        let first_chunk = chunk_storage.get(&chunk_ids[0]);
        let Some(stored) = first_chunk else {
            continue;
        };

        // Collect full content from all chunks
        let mut content = String::new();
        for cid in chunk_ids {
            if let Some(sc) = chunk_storage.get(cid) {
                content.push_str(&sc.chunk.content);
                content.push(' ');
            }
        }

        // Compute identity
        let identity = DocumentIdentity::compute(&content);

        // Check if already in registry
        if skip_existing {
            if let Some(_existing) = registry.get(&identity.content_id) {
                skipped += 1;
                continue;
            }
        }

        // Check for duplicates
        let dup_check = registry.check_duplicate(&identity);
        match dup_check {
            dindex::index::DuplicateCheckResult::ExactMatch { entry } => {
                if !dry_run && !skip_existing {
                    // Update existing entry with this doc's chunks
                    registry.update_metadata(
                        &entry.content_id,
                        stored.chunk.metadata.source_url.clone(),
                        None,
                    );
                }
                duplicates += 1;
                continue;
            }
            dindex::index::DuplicateCheckResult::NearDuplicate { entry, .. } => {
                if !dry_run && !skip_existing {
                    registry.update_metadata(
                        &entry.content_id,
                        stored.chunk.metadata.source_url.clone(),
                        None,
                    );
                }
                duplicates += 1;
                continue;
            }
            dindex::index::DuplicateCheckResult::New => {
                // Register new document
                if !dry_run {
                    registry.register(
                        identity,
                        stored.chunk.metadata.source_title.clone(),
                        stored.chunk.metadata.source_url.clone(),
                        "migrated",
                        Some(("original_doc_id", doc_id.as_str())),
                        chunk_ids.clone(),
                    );
                }
                migrated += 1;
            }
        }
    }

    // Save registry
    if !dry_run {
        registry.save()?;
        println!("\nRegistry saved to {}", config.node.data_dir.display());
    }

    println!("\nMigration Summary");
    println!("=================");
    println!("Documents migrated:  {}", migrated);
    println!("Documents skipped:   {}", skipped);
    println!("Duplicates found:    {}", duplicates);
    println!("Total processed:     {}", doc_chunks.len());

    Ok(())
}

async fn show_registry_stats(config: Config) -> Result<()> {
    // Try to load the registry
    let registry = match DocumentRegistry::load(
        &config.node.data_dir,
        config.dedup.simhash_distance_threshold,
    ) {
        Ok(r) => r,
        Err(e) => {
            println!("No document registry found: {}", e);
            println!("\nTo create a registry, import documents with:");
            println!("  dindex import <dump-file>");
            println!("Or migrate existing documents with:");
            println!("  dindex migrate-registry");
            return Ok(());
        }
    };

    let stats = registry.stats();

    println!("\nDocument Registry Statistics");
    println!("============================");
    println!("Total documents:     {}", stats.total_documents);
    println!("Total chunks:        {}", stats.total_chunks);
    println!("Total URLs:          {}", stats.total_urls);
    println!("SimHash buckets:     {}", stats.buckets_used);

    if !stats.source_counts.is_empty() {
        println!("\nDocuments by source:");
        for (source, count) in &stats.source_counts {
            println!("  {}: {}", source, count);
        }
    }

    println!("\nConfiguration:");
    println!("  Dedup enabled:     {}", config.dedup.enabled);
    println!("  SimHash threshold: {}", config.dedup.simhash_distance_threshold);
    println!("  Data directory:    {}", config.node.data_dir.display());

    Ok(())
}
