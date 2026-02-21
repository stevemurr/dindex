//! Embedding backend trait definitions
//!
//! Defines the core trait that all embedding backends must implement.

use crate::types::Embedding;
use std::fmt::Debug;

/// Errors that can occur during embedding operations
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Model was not found or could not be loaded
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Embedding generation failed
    #[error("Embedding failed: {0}")]
    EmbeddingFailed(String),

    /// Rate limited by the API
    #[error("Rate limited, retry after {retry_after_ms:?}ms")]
    RateLimited {
        /// Suggested retry delay in milliseconds, if provided by the API
        retry_after_ms: Option<u64>,
    },

    /// Network or HTTP error
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Generic error wrapper
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type for embedding operations
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

/// Core trait for embedding backends
///
/// All embedding backends (local, HTTP, etc.) must implement this trait.
/// The trait is designed to be object-safe for use with `dyn EmbeddingBackend`.
pub trait EmbeddingBackend: Send + Sync + Debug {
    /// Generate embedding for a single text
    fn embed(&self, text: &str) -> EmbeddingResult<Embedding>;

    /// Generate embeddings for a batch of texts
    ///
    /// Backends should implement this efficiently for batch processing.
    /// The default implementation calls `embed` for each text.
    fn embed_batch(&self, texts: &[String]) -> EmbeddingResult<Vec<Embedding>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Get the full embedding dimensions
    fn dimensions(&self) -> usize;

    /// Get the truncated dimensions for routing (Matryoshka support)
    ///
    /// Default implementation returns full dimensions (no truncation).
    fn truncated_dimensions(&self) -> usize {
        self.dimensions()
    }

    /// Get the backend name (e.g., "local", "http", "openai")
    fn name(&self) -> &str;
}
