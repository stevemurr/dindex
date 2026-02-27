//! Embedding service for shared engine initialization
//!
//! Provides a consistent way to initialize the embedding engine
//! across all commands (index, search, import, scrape).
//!
//! Supports both the new pluggable backend system and legacy configuration
//! for backward compatibility.

use crate::config::Config;
use crate::embedding::EmbeddingEngine;
use crate::embedding::model::ModelRegistry;
use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::info;

/// Initialize embedding engine from config
///
/// Creates the embedding engine with the configured backend.
/// Supports both new-style backend configuration and legacy fields.
pub fn init_embedding_engine(config: &Config) -> Result<Arc<EmbeddingEngine>> {
    // Log what we're doing
    if let Some(ref backend_config) = config.embedding.resolve_backend() {
        match backend_config {
            crate::config::BackendConfig::Http { endpoint, model, .. } => {
                info!("Initializing HTTP embedding backend: {} ({})", endpoint, model);
            }
            crate::config::BackendConfig::Local { model_name, .. } => {
                info!("Initializing local embedding backend: {}", model_name);
            }
        }
    } else {
        info!(
            "Initializing embedding engine with model: {}",
            config.embedding.model_name
        );
    }

    // Print execution provider info
    print_embedding_status(&config.embedding);

    // Create engine using the backend factory
    let engine = EmbeddingEngine::new(&config.embedding)
        .context("Failed to initialize embedding engine")?;

    info!(
        "Embedding engine ready: {} ({} dimensions)",
        engine.backend_name(),
        engine.dimensions()
    );

    Ok(Arc::new(engine))
}

/// Print embedding execution status
fn print_embedding_status(config: &crate::config::EmbeddingConfig) {
    // Check if using HTTP backend
    if let Some(ref backend_config) = config.resolve_backend() {
        match backend_config {
            crate::config::BackendConfig::Http { endpoint, .. } => {
                println!("  Embeddings: HTTP ({})", endpoint);
                return;
            }
            crate::config::BackendConfig::Local { .. } => {
                // Fall through to local status
            }
        }
    }

    // Local backend status
    #[cfg(feature = "metal")]
    if config.use_gpu {
        println!("  Embeddings: GPU (Metal)");
        return;
    }

    #[cfg(feature = "cuda")]
    if config.use_gpu {
        println!("  Embeddings: GPU (CUDA device {})", config.gpu_device_id);
        return;
    }

    if config.use_gpu {
        println!("  Embeddings: CPU (GPU requested but no GPU feature enabled)");
        println!("    Rebuild with --features cuda or --features metal");
    } else {
        println!("  Embeddings: CPU ({} threads)", config.num_threads);
    }
}

/// Check if a model is known in the registry
pub fn check_model_exists(config: &Config) -> Result<bool> {
    // If using HTTP backend, we can't check model existence locally
    if let Some(ref backend_config) = config.embedding.resolve_backend() {
        if matches!(backend_config, crate::config::BackendConfig::Http { .. }) {
            // For HTTP backends, we assume the model exists on the server
            return Ok(true);
        }
    }

    // With embed_anything, models are downloaded on demand
    // We just check if the model is in our registry or looks like a valid HF ID
    let model_name = &config.embedding.model_name;
    Ok(ModelRegistry::get(model_name).is_some())
}

/// Print a helpful error message if model is not found
pub fn model_not_found_error(config: &Config) {
    use crate::embedding::model::print_models;

    eprintln!("Error: Unknown model '{}'.", config.embedding.model_name);
    eprintln!();
    print_models();
}

/// Generate an embedding with hash-based fallback
///
/// Tries the real embedding engine first. If it fails or is not available,
/// falls back to a deterministic hash-based embedding.
pub fn generate_with_fallback(
    engine: Option<&EmbeddingEngine>,
    content: &str,
    fallback_dims: usize,
) -> Vec<f32> {
    if let Some(engine) = engine {
        match engine.embed(content) {
            Ok(embedding) => return embedding,
            Err(e) => {
                tracing::warn!("Embedding generation failed, using fallback: {}", e);
                return hash_based_embedding(content, engine.dimensions());
            }
        }
    }

    tracing::warn!("No embedding engine available, using hash-based fallback");
    hash_based_embedding(content, fallback_dims)
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
    let mut embedding: Vec<f32> = (0..dims)
        .map(|i| {
            let hash = xxhash_rust::xxh3::xxh3_64(content.as_bytes());
            ((hash.wrapping_add(i as u64) % 1000) as f32 / 500.0) - 1.0
        })
        .collect();

    // Normalize to unit length
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut embedding {
            *v /= norm;
        }
    }

    embedding
}
