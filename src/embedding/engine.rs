//! Embedding engine implementation
//!
//! Requires the `onnx` feature (enabled by default) for real embeddings.
//! Enable the `cuda` feature for GPU acceleration.

use crate::config::EmbeddingConfig;
use crate::types::Embedding;
use anyhow::{Context, Result};
use ort::{execution_providers::CPUExecutionProvider, session::Session, value::Tensor};
#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
use parking_lot::Mutex;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

/// Embedding engine for generating vector embeddings from text
///
/// Uses ONNX Runtime for CPU-optimized inference.
pub struct EmbeddingEngine {
    /// ONNX session for inference (wrapped in Mutex for interior mutability)
    session: Mutex<Session>,
    /// Tokenizer for text preprocessing
    tokenizer: Tokenizer,
    /// Model configuration
    config: EmbeddingConfig,
}

impl EmbeddingEngine {
    /// Create a new embedding engine
    ///
    /// Requires the model and tokenizer files to be downloaded first.
    /// Use `dindex download <model-name>` to download the required files.
    pub fn new(config: &EmbeddingConfig) -> Result<Self> {
        info!(
            "Initializing embedding engine with model: {}",
            config.model_name
        );

        // Get model and tokenizer paths
        let model_path = config
            .model_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path not specified"))?;
        let tokenizer_path = config
            .tokenizer_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer path not specified"))?;

        // Build session with appropriate execution provider
        let session = Self::build_session(config, model_path)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        info!(
            "Embedding engine initialized: {} dimensions, max {} tokens",
            config.dimensions, config.max_sequence_length
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            config: config.clone(),
        })
    }

    /// Build ONNX session with configured execution provider
    fn build_session(config: &EmbeddingConfig, model_path: &std::path::Path) -> Result<Session> {
        #[cfg(feature = "cuda")]
        if config.use_gpu {
            info!("Attempting to use CUDA GPU acceleration (device {})", config.gpu_device_id);

            let cuda_provider = CUDAExecutionProvider::default()
                .with_device_id(config.gpu_device_id as i32)
                .build();

            // Try CUDA first, fall back to CPU
            match Session::builder()?
                .with_execution_providers([cuda_provider, CPUExecutionProvider::default().build()])?
                .with_intra_threads(config.num_threads)?
                .commit_from_file(model_path)
            {
                Ok(session) => {
                    info!("CUDA GPU acceleration enabled");
                    return Ok(session);
                }
                Err(e) => {
                    warn!("CUDA initialization failed, falling back to CPU: {}", e);
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        if config.use_gpu {
            warn!("GPU acceleration requested but 'cuda' feature not enabled. Using CPU.");
            warn!("Rebuild with: cargo build --features cuda");
        }

        // CPU fallback
        info!("Using CPU execution provider");
        Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")
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

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        // Find max length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.config.max_sequence_length);

        // Prepare input tensors
        let batch_size = texts.len();
        let mut input_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let len = ids.len().min(max_len);

            // Add IDs with padding
            for i in 0..max_len {
                if i < len {
                    input_ids.push(ids[i] as i64);
                    attention_mask.push(1);
                    token_type_ids.push(0);
                } else {
                    input_ids.push(0); // Padding token
                    attention_mask.push(0);
                    token_type_ids.push(0);
                }
            }
        }

        // Create input tensors using (shape, data) tuple format
        let shape = [batch_size, max_len];

        // Run inference and extract output data (copy to owned)
        let (output_shape, output_data): (Vec<usize>, Vec<f32>) = {
            let mut session = self.session.lock();
            let outputs = session.run(ort::inputs![
                "input_ids" => Tensor::from_array((shape, input_ids))?,
                "attention_mask" => Tensor::from_array((shape, attention_mask))?,
                "token_type_ids" => Tensor::from_array((shape, token_type_ids))?,
            ])?;

            // Extract embeddings from output - convert to owned data
            if let Some(t) = outputs.get("last_hidden_state") {
                let arr = t.try_extract_array::<f32>()?;
                (arr.shape().to_vec(), arr.iter().copied().collect())
            } else if let Some(t) = outputs.get("sentence_embedding") {
                let arr = t.try_extract_array::<f32>()?;
                (arr.shape().to_vec(), arr.iter().copied().collect())
            } else {
                let (_, v) = outputs
                    .iter()
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("No output tensor found"))?;
                let arr = v.try_extract_array::<f32>()?;
                (arr.shape().to_vec(), arr.iter().copied().collect())
            }
        };

        // Reconstruct as ndarray view from owned data
        let output_view =
            ndarray::ArrayViewD::from_shape(output_shape.as_slice(), output_data.as_slice())?;

        let embeddings = match output_view.ndim() {
            3 => {
                // [batch, seq_len, hidden] - need pooling
                self.mean_pool(&output_view, &encodings, max_len)?
            }
            2 => {
                // [batch, hidden] - already pooled
                (0..batch_size)
                    .map(|i| output_view.slice(ndarray::s![i, ..]).to_vec())
                    .collect()
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unexpected output shape: {:?}",
                    output_view.shape()
                ))
            }
        };

        // Normalize embeddings
        let normalized: Vec<Embedding> = embeddings
            .into_iter()
            .map(|e| normalize_embedding(&e))
            .collect();

        Ok(normalized)
    }

    /// Mean pooling with attention mask
    fn mean_pool(
        &self,
        output: &ndarray::ArrayViewD<f32>,
        encodings: &[tokenizers::Encoding],
        max_len: usize,
    ) -> Result<Vec<Embedding>> {
        let batch_size = output.shape()[0];
        let hidden_size = output.shape()[2];

        let mut embeddings = Vec::with_capacity(batch_size);

        for (i, encoding) in encodings.iter().enumerate() {
            let seq_len = encoding.get_ids().len().min(max_len);
            let mut pooled = vec![0.0f32; hidden_size];

            // Sum over valid tokens (with attention)
            for j in 0..seq_len {
                for k in 0..hidden_size {
                    pooled[k] += output[[i, j, k]];
                }
            }

            // Average
            let count = seq_len as f32;
            for val in &mut pooled {
                *val /= count;
            }

            embeddings.push(pooled);
        }

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
        let embedding: Vec<f32> = (0..768).map(|i| i as f32).collect();
        let truncated = truncate_matryoshka(&embedding, 256);
        assert_eq!(truncated.len(), 256);
    }
}
