//! Embedding backend configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Default timeout for HTTP backend requests
fn default_timeout() -> u64 {
    30
}

/// Default batch size for HTTP backend requests
fn default_batch_size() -> usize {
    100
}

/// Backend configuration for embedding providers
///
/// Supports OpenAI-compatible HTTP endpoints (OpenAI, Azure, LM Studio, vLLM, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "lowercase")]
pub enum BackendConfig {
    /// OpenAI-compatible HTTP endpoint
    ///
    /// Works with: OpenAI API, Azure OpenAI, LM Studio, vLLM,
    /// Ollama (OpenAI compat mode), text-embeddings-inference
    Http {
        /// API endpoint URL (e.g., "https://api.openai.com/v1/embeddings")
        endpoint: String,
        /// API key (optional, can also use OPENAI_API_KEY env var)
        #[serde(default)]
        api_key: Option<String>,
        /// Model name (e.g., "text-embedding-3-small")
        model: String,
        /// Embedding dimensions
        dimensions: usize,
        /// Request timeout in seconds
        #[serde(default = "default_timeout")]
        timeout_secs: u64,
        /// Maximum batch size for requests
        #[serde(default = "default_batch_size")]
        max_batch_size: usize,
    },
}

/// Embedding model configuration
///
/// Supports two configuration styles:
///
/// 1. **New style** (recommended): Use `backend = "http"` with endpoint fields
/// ```toml
/// [embedding]
/// backend = "http"
/// endpoint = "https://api.openai.com/v1/embeddings"
/// model = "text-embedding-3-small"
/// dimensions = 1536
/// ```
///
/// 2. **Legacy style**: Use the flat fields (backward compatible)
/// ```toml
/// [embedding]
/// model_name = "all-MiniLM-L6-v2"
/// dimensions = 384
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Backend type: "http" or "local" (default: "local" for legacy compat)
    #[serde(default)]
    pub backend: Option<String>,

    /// HTTP backend: API endpoint URL
    #[serde(default)]
    pub endpoint: Option<String>,
    /// HTTP backend: API key (optional, can also use OPENAI_API_KEY env var)
    #[serde(default)]
    pub api_key: Option<String>,
    /// HTTP backend: Model name for API requests (e.g., "bge-m3")
    #[serde(default)]
    pub model: Option<String>,
    /// HTTP backend: Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// HTTP backend: Maximum batch size for requests
    #[serde(default = "default_batch_size")]
    pub max_batch_size: usize,

    // ---- Legacy/local fields ----

    /// Model name for local backend (e.g., "all-MiniLM-L6-v2")
    #[serde(default = "default_model_name")]
    pub model_name: String,
    /// Path to model files (optional)
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    /// Path to tokenizer files (optional)
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,
    /// Embedding dimensions
    #[serde(default = "default_dimensions")]
    pub dimensions: usize,
    /// Truncated dimensions for Matryoshka (routing)
    #[serde(default = "default_dimensions")]
    pub truncated_dimensions: usize,
    /// Maximum sequence length
    #[serde(default = "default_max_seq_length")]
    pub max_sequence_length: usize,
    /// Enable INT8 quantization (deprecated)
    #[serde(default)]
    pub quantize_int8: bool,
    /// Number of threads for inference
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
    /// Use GPU acceleration
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
    /// GPU device ID
    #[serde(default)]
    pub gpu_device_id: usize,
}

fn default_model_name() -> String {
    "all-MiniLM-L6-v2".to_string()
}

fn default_dimensions() -> usize {
    384
}

fn default_max_seq_length() -> usize {
    256
}

fn default_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(8)
}

fn default_use_gpu() -> bool {
    false
}

impl EmbeddingConfig {
    /// Resolve the backend configuration from flat fields
    pub fn resolve_backend(&self) -> Option<BackendConfig> {
        match self.backend.as_deref() {
            Some("http") => {
                let endpoint = self.endpoint.clone()?;
                let model = self.model.clone().unwrap_or_else(|| self.model_name.clone());
                Some(BackendConfig::Http {
                    endpoint,
                    api_key: self.api_key.clone(),
                    model,
                    dimensions: self.dimensions,
                    timeout_secs: self.timeout_secs,
                    max_batch_size: self.max_batch_size,
                })
            }
            _ => None,
        }
    }

    /// Resolve model paths from the data directory if not explicitly set.
    /// Kept for backward compatibility with legacy local config layouts.
    pub fn resolve_paths(&mut self, data_dir: &std::path::Path) {
        if self.model_path.is_none() || self.tokenizer_path.is_none() {
            let model_dir = data_dir.join("models").join(&self.model_name);
            let model_file = model_dir.join("model.onnx");
            let tokenizer_file = model_dir.join("tokenizer.json");

            if model_file.exists() && self.model_path.is_none() {
                self.model_path = Some(model_file);
            }
            if tokenizer_file.exists() && self.tokenizer_path.is_none() {
                self.tokenizer_path = Some(tokenizer_file);
            }
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: None,
            endpoint: None,
            api_key: None,
            model: None,
            timeout_secs: 30,
            max_batch_size: 100,
            model_name: "all-MiniLM-L6-v2".to_string(),
            model_path: None,
            tokenizer_path: None,
            dimensions: 384,
            truncated_dimensions: 384,
            max_sequence_length: 256,
            quantize_int8: false,
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
                .min(8),
            use_gpu: default_use_gpu(),
            gpu_device_id: 0,
        }
    }
}
