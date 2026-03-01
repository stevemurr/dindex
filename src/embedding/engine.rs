//! Embedding engine implementation
//!
//! Wraps the pluggable HTTP embedding backend system with a consistent API
//! used across all commands (index, search, import, scrape).

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
            .map_err(|e| anyhow::anyhow!(e))
            .context("Failed to generate embedding")
    }

    /// Generate embeddings for a batch of texts
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        self.backend
            .embed_batch(texts)
            .map_err(|e| anyhow::anyhow!(e))
            .context("Failed to generate batch embeddings")
    }

    /// Generate embeddings for a batch of texts with count validation.
    ///
    /// Returns an error if the backend returns a different number of embeddings
    /// than the number of input texts. This prevents the silent data loss that
    /// occurs when using `.zip()` on mismatched iterators.
    pub fn embed_batch_validated(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let expected = texts.len();
        let embeddings = self.embed_batch(texts)?;
        if embeddings.len() != expected {
            anyhow::bail!(
                "Embedding count mismatch: sent {} texts, got {} embeddings back. \
                 This may indicate the backend dropped or batched incorrectly.",
                expected,
                embeddings.len()
            );
        }
        Ok(embeddings)
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

/// Compute cosine similarity between two embeddings.
///
/// Returns 0.0 if dimensions mismatch (with a debug assertion in debug builds).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Embeddings must have same dimension");
    if a.len() != b.len() {
        tracing::warn!(
            "cosine_similarity called with mismatched dimensions: {} vs {}",
            a.len(),
            b.len()
        );
        return 0.0;
    }

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
    use crate::embedding::backend::{EmbeddingBackend, EmbeddingResult};

    /// Mock backend that returns a fixed number of embeddings (for testing validation)
    #[derive(Debug)]
    struct MockBackend {
        dims: usize,
        /// If set, embed_batch returns this many embeddings regardless of input size
        override_count: Option<usize>,
    }

    impl EmbeddingBackend for MockBackend {
        fn embed(&self, _text: &str) -> EmbeddingResult<Embedding> {
            Ok(vec![0.1; self.dims])
        }

        fn embed_batch(&self, texts: &[String]) -> EmbeddingResult<Vec<Embedding>> {
            let count = self.override_count.unwrap_or(texts.len());
            Ok((0..count).map(|_| vec![0.1; self.dims]).collect())
        }

        fn dimensions(&self) -> usize {
            self.dims
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    fn make_engine(override_count: Option<usize>) -> EmbeddingEngine {
        EmbeddingEngine {
            backend: Arc::new(MockBackend {
                dims: 4,
                override_count,
            }),
            truncated_dimensions: 4,
        }
    }

    #[test]
    fn test_normalize_embedding() {
        let embedding = vec![3.0, 4.0];
        let normalized = crate::util::normalize_embedding(&embedding);
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
    fn test_embed_batch_validated_ok() {
        let engine = make_engine(None); // returns same count as input
        let texts = vec!["hello".to_string(), "world".to_string()];
        let result = engine.embed_batch_validated(&texts);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn test_embed_batch_validated_mismatch() {
        let engine = make_engine(Some(1)); // always returns 1 embedding
        let texts = vec!["hello".to_string(), "world".to_string()];
        let result = engine.embed_batch_validated(&texts);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("count mismatch"),
            "error should mention count mismatch, got: {}",
            err
        );
    }

    #[test]
    fn test_embed_batch_validated_empty() {
        let engine = make_engine(None);
        let texts: Vec<String> = vec![];
        let result = engine.embed_batch_validated(&texts);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
