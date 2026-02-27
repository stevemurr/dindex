//! Embedding service for shared engine initialization
//!
//! Provides a consistent way to initialize the embedding engine
//! across all commands (index, search, import, scrape).

use crate::config::Config;
use crate::embedding::EmbeddingEngine;
use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::info;

/// Initialize embedding engine from config
///
/// Creates the embedding engine with the configured backend.
pub fn init_embedding_engine(config: &Config) -> Result<Arc<EmbeddingEngine>> {
    if let Some(ref backend_config) = config.embedding.resolve_backend() {
        match backend_config {
            crate::config::BackendConfig::Http { endpoint, model, .. } => {
                info!("Initializing HTTP embedding backend: {} ({})", endpoint, model);
            }
        }
    } else {
        info!("Initializing embedding engine (no backend configured)");
    }

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
