//! DIndex: Decentralized Semantic Search Index for LLM Consumption
//!
//! A federated semantic search system optimized for LLM consumption.

use anyhow::Result;
use clap::{Parser, Subcommand};
use dindex::{
    chunking::TextSplitter,
    config::Config,
    embedding::ModelManager,
    index::{ChunkStorage, VectorIndex},
    network::NetworkNode,
    retrieval::{Bm25Index, HybridIndexer, HybridRetriever},
    types::{Document, Query},
};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

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
    info!("Indexing: {}", path.display());

    // Read document content
    let content = if path.is_file() {
        std::fs::read_to_string(&path)?
    } else if path.is_dir() {
        // Index all text files in directory
        let mut combined = String::new();
        for entry in walkdir::WalkDir::new(&path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if let Ok(text) = std::fs::read_to_string(entry.path()) {
                combined.push_str(&text);
                combined.push_str("\n\n");
            }
        }
        combined
    } else {
        anyhow::bail!("Path does not exist: {}", path.display());
    };

    // Create document
    let mut doc = Document::new(content);
    if let Some(t) = title {
        doc = doc.with_title(t);
    } else {
        doc = doc.with_title(path.file_name().unwrap_or_default().to_string_lossy());
    }
    if let Some(u) = url {
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
        format!("{}", &s[..max_len])
    } else {
        s
    }
}

// Add walkdir and toml to dependencies
use walkdir;
