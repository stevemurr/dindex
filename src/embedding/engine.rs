//! Embedding engine implementation
//!
//! Uses embed_anything with candle backend for text embeddings.
//! Supports CPU, CUDA (--features cuda), and Metal (--features metal).

use crate::config::EmbeddingConfig;
use crate::types::Embedding;
use anyhow::{Context, Result};
use embed_anything::embeddings::embed::{Embedder, EmbeddingResult};
use std::sync::Arc;
use tracing::{debug, info};

/// Embedding engine for generating vector embeddings from text
///
/// Uses embed_anything with candle backend for efficient inference.
/// Supports CPU, CUDA, and Metal acceleration.
pub struct EmbeddingEngine {
    /// The embedder from embed_anything
    embedder: Arc<Embedder>,
    /// Model configuration
    config: EmbeddingConfig,
}

impl EmbeddingEngine {
    /// Create a new embedding engine
    ///
    /// Downloads the model automatically if not cached.
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        info!(
            "Initializing embedding engine with model: {}",
            config.model_name
        );

        // Resolve model name to HuggingFace model ID
        let model_id = Self::resolve_model_id(&config.model_name)?;

        info!("Loading model: {}", model_id);

        // Use Embedder::from_pretrained_hf which auto-detects architecture from config.json
        let embedder = Embedder::from_pretrained_hf(&model_id, None, None, None)
            .context("Failed to load embedding model")?;

        info!(
            "Embedding engine initialized: {} dimensions, max {} tokens",
            config.dimensions, config.max_sequence_length
        );

        Ok(Self {
            embedder: Arc::new(embedder),
            config: config.clone(),
        })
    }

    /// Resolve model name to HuggingFace model ID
    fn resolve_model_id(model_name: &str) -> Result<String> {
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
            "bge-m3" => Err(anyhow::anyhow!(
                "bge-m3 is not supported (uses XLMRobertaModel architecture). \
                 Use bge-base-en-v1.5 or all-MiniLM-L6-v2 instead."
            )),

            // Allow direct HuggingFace model IDs (must use BertModel architecture)
            name if name.contains('/') => Ok(name.to_string()),

            _ => Err(anyhow::anyhow!(
                "Unknown model: {}. Supported models: all-MiniLM-L6-v2, bge-base-en-v1.5, \
                 bge-large-en-v1.5, e5-base-v2, e5-large-v2, or a HuggingFace model ID \
                 with BertModel architecture",
                model_name
            ))
        }
    }

    /// Generate embedding for a single text
    pub fn embed(&self, text: &str) -> Result<Embedding> {
        let embeddings = self.embed_batch(&[text.to_string()])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))
    }

    /// Generate embeddings for a batch of texts
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Embedding batch of {} texts", texts.len());

        // Clone texts for the async block (needed for spawn_blocking)
        let texts_owned: Vec<String> = texts.to_vec();
        let embedder = self.embedder.clone();
        let batch_size = texts.len().min(32);

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
                }).join().expect("Embedding thread panicked")
            })
        } else {
            // Not in a tokio runtime - use futures executor directly
            let text_refs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
            futures::executor::block_on(async {
                embedder.embed(&text_refs, Some(batch_size), None).await
            })
        }.context("Embedding failed")?;

        // Convert EmbeddingResult to Vec<f32> and normalize
        let embeddings: Vec<Embedding> = results
            .into_iter()
            .map(|result| {
                let embedding = match result {
                    EmbeddingResult::DenseVector(v) => v,
                    EmbeddingResult::MultiVector(vecs) => {
                        // For multi-vector, take the mean
                        if vecs.is_empty() {
                            vec![0.0; self.config.dimensions]
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

    /// Truncate embedding for Matryoshka (variable dimension)
    pub fn truncate_embedding(&self, embedding: &Embedding, target_dims: usize) -> Embedding {
        truncate_matryoshka(embedding, target_dims)
    }

    /// Get the full embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Get the truncated dimensions for routing
    pub fn truncated_dimensions(&self) -> usize {
        self.config.truncated_dimensions
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

    #[test]
    fn test_resolve_model_id() {
        // Test known models
        let id = EmbeddingEngine::resolve_model_id("all-MiniLM-L6-v2").unwrap();
        assert_eq!(id, "sentence-transformers/all-MiniLM-L6-v2");

        let id = EmbeddingEngine::resolve_model_id("bge-base-en-v1.5").unwrap();
        assert_eq!(id, "BAAI/bge-base-en-v1.5");

        // Test direct HF ID
        let id = EmbeddingEngine::resolve_model_id("BAAI/bge-base-en-v1.5").unwrap();
        assert_eq!(id, "BAAI/bge-base-en-v1.5");

        // Test unsupported model (bge-m3 uses XLMRobertaModel)
        assert!(EmbeddingEngine::resolve_model_id("bge-m3").is_err());

        // Test unknown model
        assert!(EmbeddingEngine::resolve_model_id("unknown-model").is_err());
    }
}
