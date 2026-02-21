//! Local embedding backend using embed_anything
//!
//! This backend runs embedding inference locally using the embed_anything crate
//! with candle as the inference backend. Supports CPU, CUDA, and Metal acceleration.

use super::traits::{EmbeddingBackend, EmbeddingError, EmbeddingResult};
use crate::types::Embedding;
use std::fmt;
use embed_anything::embeddings::embed::{Embedder, EmbeddingResult as EAResult};
use std::sync::Arc;
use tracing::{debug, info};

/// Configuration for the local embedding backend
#[derive(Debug, Clone)]
pub struct LocalConfig {
    /// Model name (e.g., "all-MiniLM-L6-v2", "bge-base-en-v1.5")
    pub model_name: String,
    /// Full embedding dimensions
    pub dimensions: usize,
    /// Truncated dimensions for routing (Matryoshka support)
    pub truncated_dimensions: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
}

/// Local embedding backend using embed_anything
pub struct LocalBackend {
    /// The embedder from embed_anything
    embedder: Arc<Embedder>,
    /// Configuration
    config: LocalConfig,
}

impl fmt::Debug for LocalBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalBackend")
            .field("config", &self.config)
            .field("embedder", &"<Embedder>")
            .finish()
    }
}

impl LocalBackend {
    /// Create a new local embedding backend
    ///
    /// Downloads the model automatically if not cached.
    pub fn new(config: LocalConfig) -> EmbeddingResult<Self> {
        info!(
            "Initializing local embedding backend with model: {}",
            config.model_name
        );

        // Resolve model name to HuggingFace model ID
        let model_id = Self::resolve_model_id(&config.model_name)?;

        info!("Loading model: {}", model_id);

        // Use Embedder::from_pretrained_hf which auto-detects architecture from config.json
        let embedder = Embedder::from_pretrained_hf(&model_id, None, None, None)
            .map_err(|e| EmbeddingError::ModelNotFound(format!("Failed to load model: {}", e)))?;

        info!(
            "Local embedding backend initialized: {} dimensions, max {} tokens",
            config.dimensions, config.max_sequence_length
        );

        Ok(Self {
            embedder: Arc::new(embedder),
            config,
        })
    }

    /// Resolve model name to HuggingFace model ID
    fn resolve_model_id(model_name: &str) -> EmbeddingResult<String> {
        match model_name {
            // Sentence transformers (default, fast)
            "all-MiniLM-L6-v2" => Ok("sentence-transformers/all-MiniLM-L6-v2".to_string()),
            "all-MiniLM-L12-v2" => Ok("sentence-transformers/all-MiniLM-L12-v2".to_string()),

            // BGE models (English, BertModel architecture)
            "bge-base-en-v1.5" => Ok("BAAI/bge-base-en-v1.5".to_string()),
            "bge-large-en-v1.5" => Ok("BAAI/bge-large-en-v1.5".to_string()),
            "bge-small-en-v1.5" => Ok("BAAI/bge-small-en-v1.5".to_string()),

            // E5 models
            "e5-small-v2" => Ok("intfloat/e5-small-v2".to_string()),
            "e5-base-v2" => Ok("intfloat/e5-base-v2".to_string()),
            "e5-large-v2" => Ok("intfloat/e5-large-v2".to_string()),

            // Note: bge-m3 uses XLMRobertaModel which is not supported by embed_anything
            "bge-m3" => Err(EmbeddingError::Config(
                "bge-m3 is not supported (uses XLMRobertaModel architecture). \
                 Use bge-base-en-v1.5 or all-MiniLM-L6-v2 instead."
                    .to_string(),
            )),

            // Allow direct HuggingFace model IDs (must use BertModel architecture)
            name if name.contains('/') => Ok(name.to_string()),

            _ => Err(EmbeddingError::Config(format!(
                "Unknown model: {}. Supported models: all-MiniLM-L6-v2, bge-base-en-v1.5, \
                 bge-large-en-v1.5, e5-base-v2, e5-large-v2, or a HuggingFace model ID \
                 with BertModel architecture",
                model_name
            ))),
        }
    }
}

impl EmbeddingBackend for LocalBackend {
    fn embed(&self, text: &str) -> EmbeddingResult<Embedding> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::EmbeddingFailed("No embedding generated".to_string()))
    }

    fn embed_batch(&self, texts: &[String]) -> EmbeddingResult<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Embedding batch of {} texts", texts.len());

        // Clone texts for the async block (needed for spawn_blocking)
        let texts_owned: Vec<String> = texts.to_vec();
        let embedder = self.embedder.clone();
        let batch_size = texts.len().min(32);
        let dimensions = self.config.dimensions;

        // Run the embedding operation
        // We need to handle both cases: called from async context (daemon) or sync context (CLI)
        let results = if tokio::runtime::Handle::try_current().is_ok() {
            // We're in a tokio runtime - use spawn_blocking to run in a separate thread pool
            // This avoids any potential issues with blocking the async runtime
            std::thread::scope(|s| {
                s.spawn(|| {
                    let text_refs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
                    futures::executor::block_on(async {
                        embedder.embed(&text_refs, Some(batch_size), None).await
                    })
                })
                .join()
                .expect("Embedding thread panicked")
            })
        } else {
            // Not in a tokio runtime - use futures executor directly
            let text_refs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
            futures::executor::block_on(async {
                embedder.embed(&text_refs, Some(batch_size), None).await
            })
        }
        .map_err(|e| EmbeddingError::EmbeddingFailed(format!("Embedding failed: {}", e)))?;

        // Convert EmbeddingResult to Vec<f32> and normalize
        let embeddings: Vec<Embedding> = results
            .into_iter()
            .map(|result| {
                let embedding = match result {
                    EAResult::DenseVector(v) => v,
                    EAResult::MultiVector(vecs) => {
                        // For multi-vector, take the mean
                        if vecs.is_empty() {
                            vec![0.0; dimensions]
                        } else {
                            let dim = vecs[0].len();
                            let mut mean = vec![0.0f32; dim];
                            for v in &vecs {
                                for (i, val) in v.iter().enumerate() {
                                    mean[i] += val;
                                }
                            }
                            let count = vecs.len() as f32;
                            for val in &mut mean {
                                *val /= count;
                            }
                            mean
                        }
                    }
                };
                normalize_embedding(&embedding)
            })
            .collect();

        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn truncated_dimensions(&self) -> usize {
        self.config.truncated_dimensions
    }

    fn name(&self) -> &str {
        "local"
    }
}

/// Normalize an embedding vector to unit length
fn normalize_embedding(embedding: &Embedding) -> Embedding {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter().map(|x| x / norm).collect()
    } else {
        embedding.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_model_id() {
        // Test known models
        let id = LocalBackend::resolve_model_id("all-MiniLM-L6-v2").unwrap();
        assert_eq!(id, "sentence-transformers/all-MiniLM-L6-v2");

        let id = LocalBackend::resolve_model_id("bge-base-en-v1.5").unwrap();
        assert_eq!(id, "BAAI/bge-base-en-v1.5");

        // Test direct HF ID
        let id = LocalBackend::resolve_model_id("BAAI/bge-base-en-v1.5").unwrap();
        assert_eq!(id, "BAAI/bge-base-en-v1.5");

        // Test unsupported model (bge-m3 uses XLMRobertaModel)
        assert!(LocalBackend::resolve_model_id("bge-m3").is_err());

        // Test unknown model
        assert!(LocalBackend::resolve_model_id("unknown-model").is_err());
    }

    #[test]
    fn test_normalize_embedding() {
        let embedding = vec![3.0, 4.0];
        let normalized = normalize_embedding(&embedding);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }
}
