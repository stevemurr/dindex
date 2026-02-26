//! Embedding engine implementation
//!
//! This module provides a backward-compatible wrapper around the pluggable
//! embedding backend system. It maintains the same public API as before
//! while delegating to the configured backend.

use crate::config::EmbeddingConfig;
use crate::embedding::backend::{create_backend_from_legacy, EmbeddingBackend, EmbeddingError};
use crate::types::Embedding;
use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::info;

/// Embedding engine for generating vector embeddings from text
///
/// This is a backward-compatible wrapper around the new pluggable backend system.
/// It maintains the same public API while delegating to the configured backend.
pub struct EmbeddingEngine {
    /// The underlying embedding backend
    backend: Arc<dyn EmbeddingBackend>,
    /// Truncated dimensions for routing (Matryoshka support)
    truncated_dimensions: usize,
}

impl EmbeddingEngine {
    /// Create a new embedding engine from config
    ///
    /// Uses the new backend system internally while maintaining backward compatibility.
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        info!(
            "Initializing embedding engine with model: {}",
            config.model_name
        );

        // Create backend using the factory
        let backend = create_backend_from_legacy(config).map_err(|e| match e {
            EmbeddingError::ModelNotFound(msg) => anyhow::anyhow!("Model not found: {}", msg),
            EmbeddingError::Config(msg) => anyhow::anyhow!("Configuration error: {}", msg),
            EmbeddingError::Other(e) => e,
            e => anyhow::anyhow!("{}", e),
        })?;

        // Get truncated dimensions from config or backend
        let truncated_dimensions = if config.truncated_dimensions > 0 {
            config.truncated_dimensions
        } else {
            backend.truncated_dimensions()
        };

        info!(
            "Embedding engine initialized: {} ({} dimensions, {} truncated)",
            backend.name(),
            backend.dimensions(),
            truncated_dimensions
        );

        Ok(Self {
            backend,
            truncated_dimensions,
        })
    }

    /// Generate embedding for a single text
    pub fn embed(&self, text: &str) -> Result<Embedding> {
        self.backend
            .embed(text)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to generate embedding")
    }

    /// Generate embeddings for a batch of texts
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.backend
            .embed_batch(texts)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to generate batch embeddings")
    }

    /// Get the full embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.backend.dimensions()
    }

    /// Get the truncated dimensions for routing
    pub fn truncated_dimensions(&self) -> usize {
        self.truncated_dimensions
    }

    /// Get the backend name
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// Get a reference to the underlying backend
    pub fn backend(&self) -> &Arc<dyn EmbeddingBackend> {
        &self.backend
    }
}

/// Normalize an embedding vector to unit length
pub fn normalize_embedding(embedding: &Embedding) -> Embedding {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter().map(|x| x / norm).collect()
    } else {
        embedding.clone()
    }
}

/// Truncate embedding for Matryoshka models
/// Earlier dimensions contain more important information
pub fn truncate_matryoshka(embedding: &Embedding, target_dims: usize) -> Embedding {
    let truncated: Vec<f32> = embedding.iter().take(target_dims).copied().collect();
    normalize_embedding(&truncated)
}

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Embeddings must have same dimension");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_embedding() {
        let embedding = vec![3.0, 4.0];
        let normalized = normalize_embedding(&embedding);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);

        let c = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_truncate_matryoshka() {
        // Use 1024 dimensions to match the new default (bge-m3)
        let embedding: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let truncated = truncate_matryoshka(&embedding, 256);
        assert_eq!(truncated.len(), 256);
    }
}
