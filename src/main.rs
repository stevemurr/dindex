//! DIndex: Decentralized Semantic Search Index for LLM Consumption
//!
//! A federated semantic search system optimized for LLM consumption.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use daemonize::Daemonize;
use indicatif::{ProgressBar, ProgressStyle};
use dindex::{
    config::Config,
    daemon::server::IpcServer,
    import::DumpFormat,
};
use std::fs::File;
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

mod commands;

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
    /// Manage the DIndex daemon
    #[command(name = "daemon")]
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },

    /// Start the DIndex node (daemon + P2P)
    Start {
        /// Listen address
        #[arg(short, long)]
        listen: Option<String>,

        /// Bootstrap peers
        #[arg(short, long)]
        bootstrap: Vec<String>,

        /// Run in foreground (don't daemonize)
        #[arg(short, long)]
        foreground: bool,
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
        format: Option<DumpFormat>,

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

    /// Development utilities
    #[command(name = "dev")]
    Dev {
        #[command(subcommand)]
        action: DevAction,
    },
}

/// Daemon management actions
#[derive(Subcommand)]
enum DaemonAction {
    /// Start the daemon
    Start {
        /// Run in foreground (don't daemonize)
        #[arg(short, long)]
        foreground: bool,
    },
    /// Stop the running daemon
    Stop,
    /// Check daemon status
    Status,
    /// Restart the daemon
    Restart,
}

/// Development utility actions
#[derive(Subcommand)]
enum DevAction {
    /// Reset the local index (deletes all indexed content)
    ResetIndex {
        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load or create config early (needed for daemonization paths)
    let mut config: Config = if cli.config.exists() {
        match Config::load(&cli.config) {
            Ok(config) => config,
            Err(e) => {
                eprintln!("Warning: Failed to load config file '{}': {}", cli.config.display(), e);
                eprintln!("Using default configuration.");
                Config::default()
            }
        }
    } else {
        Config::default()
    };

    // Override data dir if specified
    if let Some(ref data_dir) = cli.data_dir {
        config.node.data_dir = data_dir.clone();
    }

    // Ensure data directory exists
    std::fs::create_dir_all(&config.node.data_dir)?;

    // Check if we need to daemonize BEFORE starting tokio
    let should_daemonize = match &cli.command {
        Commands::Start { foreground, .. } => !foreground,
        Commands::Daemon { action: DaemonAction::Start { foreground } } => !foreground,
        _ => false,
    };

    if should_daemonize {
        let log_path = config.node.data_dir.join("dindex.log");
        let err_path = config.node.data_dir.join("dindex.err");
        let socket_path = IpcServer::default_socket_path();

        let stdout = File::create(&log_path)
            .with_context(|| format!("Failed to create log file: {}", log_path.display()))?;
        let stderr = File::create(&err_path)
            .with_context(|| format!("Failed to create error log: {}", err_path.display()))?;

        let daemonize = Daemonize::new()
            .working_directory(&config.node.data_dir)
            .stdout(stdout)
            .stderr(stderr);

        // Use execute() instead of start() so parent doesn't exit immediately
        match daemonize.execute() {
            daemonize::Outcome::Parent(Ok(_)) => {
                // We're the parent - wait for daemon to be ready with progress feedback
                let spinner_style = ProgressStyle::default_spinner()
                    .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                    .template("{spinner:.cyan} {msg}")
                    .unwrap();

                let spinner = ProgressBar::new_spinner();
                spinner.set_style(spinner_style);
                spinner.enable_steady_tick(std::time::Duration::from_millis(80));
                spinner.set_message("Starting daemon...");

                // Track completed stages
                let mut completed_stages: Vec<&str> = Vec::new();

                // Wait for socket to appear (daemon is ready)
                let start_time = std::time::Instant::now();
                let timeout = std::time::Duration::from_secs(120); // 2 min timeout
                let poll_interval = std::time::Duration::from_millis(250);

                // Define stages in order they appear in logs
                let stages = [
                    ("Starting DIndex", "Initializing"),
                    ("Loading vector index", "Loading vector index"),
                    ("BM25 index loaded", "Loading BM25 index"),
                    ("Chunk storage loaded", "Loading chunk storage"),
                    ("Starting IPC server", "Starting network"),
                ];

                loop {
                    if socket_path.exists() {
                        // Print all remaining stages as complete
                        for (_, display_name) in &stages {
                            if !completed_stages.contains(display_name) {
                                spinner.println(format!("  {} {}", console::style("✓").green(), display_name));
                            }
                        }
                        spinner.finish_and_clear();
                        println!("{} Daemon ready!", console::style("✓").green().bold());
                        println!("  Logs: {}", log_path.display());
                        return Ok(());
                    }

                    if start_time.elapsed() > timeout {
                        spinner.finish_and_clear();
                        eprintln!("{} Daemon startup timed out", console::style("✗").red().bold());
                        eprintln!("  Check logs at: {}", log_path.display());
                        std::process::exit(1);
                    }

                    // Check log file for status updates
                    if let Ok(content) = std::fs::read_to_string(&log_path) {
                        for (log_marker, display_name) in &stages {
                            if content.contains(log_marker) && !completed_stages.contains(display_name) {
                                // Print completed stage on its own line
                                spinner.println(format!("  {} {}", console::style("✓").green(), display_name));
                                completed_stages.push(display_name);

                                // Update spinner with next stage
                                let next_idx = completed_stages.len();
                                if next_idx < stages.len() {
                                    spinner.set_message(format!("{}...", stages[next_idx].1));
                                } else {
                                    spinner.set_message("Finalizing...");
                                }
                            }
                        }
                    }

                    std::thread::sleep(poll_interval);
                }
            }
            daemonize::Outcome::Parent(Err(e)) => {
                anyhow::bail!("Failed to daemonize: {}", e);
            }
            daemonize::Outcome::Child(Ok(_)) => {
                // We're the child - continue to start the daemon
                // Set up logging for file output
                let subscriber = FmtSubscriber::builder()
                    .with_max_level(Level::INFO)
                    .with_target(false)
                    .with_ansi(false)
                    .finish();
                let _ = tracing::subscriber::set_global_default(subscriber);
            }
            daemonize::Outcome::Child(Err(e)) => {
                eprintln!("Daemon child process error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Setup logging for foreground mode
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
    }

    // Now start the tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async_main(cli, config))
}

async fn async_main(cli: Cli, config: Config) -> Result<()> {
    match cli.command {
        Commands::Daemon { action } => commands::daemon::handle_daemon(config, action).await,
        Commands::Start {
            listen,
            bootstrap,
            foreground: _, // Already handled daemonization
        } => {
            commands::daemon::start_node_inner(config, listen, bootstrap).await
        }
        Commands::Index { path, title, url } => {
            commands::index::index_document(config, path, title, url).await
        }
        Commands::Search { query, top_k, format } => {
            commands::search::search_index(config, query, top_k, format).await
        }
        Commands::Stats => {
            commands::stats::show_stats(config).await
        }
        Commands::Export { output, format } => {
            commands::export::export_index(config, output, format).await
        }
        Commands::Init { path } => {
            commands::init::init_config(path).await
        }
        Commands::Scrape {
            urls,
            depth,
            stay_on_domain,
            max_pages,
            delay_ms,
            index,
        } => {
            commands::scrape::scrape_urls(config, urls, depth, stay_on_domain, max_pages, delay_ms, index).await
        }
        Commands::ScrapeStats => {
            commands::scrape::show_scrape_stats(config).await
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
            commands::import::import_dump(
                config, path, format, batch_size, resume, checkpoint, no_dedup, max_docs, min_length, quiet,
            )
            .await
        }
        Commands::ImportStatus { checkpoint } => {
            commands::import::show_import_status(checkpoint).await
        }
        Commands::MigrateRegistry { dry_run, skip_existing } => {
            commands::stats::migrate_registry(config, dry_run, skip_existing).await
        }
        Commands::RegistryStats => {
            commands::stats::show_registry_stats(config).await
        }
        Commands::Dev { action } => {
            commands::dev::handle_dev(config, action).await
        }
    }
}
