//! Embedding service for shared engine initialization
//!
//! Provides a consistent way to initialize the embedding engine
//! across all commands (index, search, import, scrape).

use crate::config::Config;
use crate::embedding::{EmbeddingEngine, ModelManager};
use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::info;

/// Initialize embedding engine from config
///
/// Downloads model if not present, then creates engine.
/// Prints status information about the execution provider (CPU/GPU).
pub async fn init_embedding_engine(config: &Config) -> Result<Arc<EmbeddingEngine>> {
    let cache_dir = config.node.data_dir.join("models");
    let manager = ModelManager::new(&cache_dir)?;

    // Ensure model is downloaded
    let model_name = &config.embedding.model_name;
    if !manager.model_exists(model_name)? {
        info!("Model '{}' not found. Downloading...", model_name);
        manager
            .download_model(model_name)
            .await
            .context("Failed to download embedding model")?;
    }

    // Create embedding config with paths, merging with user config for GPU settings
    let mut embedding_config = manager.create_config(model_name).await?;
    embedding_config.use_gpu = config.embedding.use_gpu;
    embedding_config.gpu_device_id = config.embedding.gpu_device_id;
    embedding_config.num_threads = config.embedding.num_threads;

    // Log execution provider info
    print_embedding_status(&embedding_config);

    // Create engine
    let engine = EmbeddingEngine::new(&embedding_config)
        .context("Failed to initialize embedding engine")?;

    info!(
        "Embedding engine ready: {} ({} dimensions)",
        model_name,
        engine.dimensions()
    );

    Ok(Arc::new(engine))
}

/// Print embedding execution status
fn print_embedding_status(config: &crate::config::EmbeddingConfig) {
    if config.use_gpu {
        #[cfg(feature = "cuda")]
        {
            println!("  Embeddings: GPU (CUDA device {})", config.gpu_device_id);
        }
        #[cfg(not(feature = "cuda"))]
        {
            println!("  Embeddings: CPU (GPU requested but 'cuda' feature not enabled)");
        }
    } else {
        println!("  Embeddings: CPU ({} threads)", config.num_threads);
    }
}

/// Check if model exists without downloading
pub fn check_model_exists(config: &Config) -> Result<bool> {
    let cache_dir = config.node.data_dir.join("models");
    let manager = ModelManager::new(&cache_dir)?;
    manager.model_exists(&config.embedding.model_name)
}

/// Print a helpful error message if model is not found
pub fn model_not_found_error(config: &Config) {
    eprintln!("Error: Embedding model '{}' not found.", config.embedding.model_name);
    eprintln!();
    eprintln!("To download the model, run:");
    eprintln!("  dindex download {}", config.embedding.model_name);
    eprintln!();
    eprintln!("Available models:");
    eprintln!("  - nomic-embed-text-v1.5 (recommended, 768 dims, Matryoshka)");
    eprintln!("  - e5-small-v2 (smaller, 384 dims)");
    eprintln!("  - bge-base-en-v1.5 (768 dims)");
    eprintln!("  - all-MiniLM-L6-v2 (smallest, 384 dims)");
}

/// Generate a deterministic hash-based embedding (fallback when embedding engine unavailable)
///
/// This produces embeddings that are deterministic for the same content but have no
/// semantic meaning. Use only as a fallback for testing or when the embedding model
/// is unavailable. Values are in the range [-1, 1].
///
/// WARNING: Hash-based embeddings will not produce meaningful search results.
/// Always prefer using a real embedding engine when possible.
pub fn hash_based_embedding(content: &str, dims: usize) -> Vec<f32> {
    (0..dims)
        .map(|i| {
            let hash = xxhash_rust::xxh3::xxh3_64(content.as_bytes());
            ((hash.wrapping_add(i as u64) % 1000) as f32 / 500.0) - 1.0
        })
        .collect()
}
