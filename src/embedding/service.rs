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

/// Generate an embedding, returning an error if no engine is available or embedding fails.
///
/// Unlike the previous `generate_with_fallback`, this does NOT silently produce
/// hash-based embeddings. Callers must handle the error explicitly — either by
/// skipping the chunk or by explicitly opting into `hash_based_embedding`.
pub fn generate_embedding(
    engine: Option<&EmbeddingEngine>,
    content: &str,
) -> Result<Vec<f32>> {
    let engine = engine.ok_or_else(|| {
        anyhow::anyhow!(
            "No embedding engine available. Configure an embedding backend in dindex.toml \
             (e.g., backend = \"http\", endpoint = \"...\")."
        )
    })?;

    engine.embed(content)
}

/// Generate a deterministic hash-based embedding for testing purposes.
///
/// This produces embeddings that are deterministic for the same content but have
/// **no semantic meaning**. Use only for:
/// - Unit/integration tests that need embeddings but not semantic quality
/// - Explicit fallback when the user has opted in via configuration
///
/// WARNING: Hash-based embeddings will not produce meaningful search results.
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
