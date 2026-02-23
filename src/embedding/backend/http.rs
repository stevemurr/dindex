//! HTTP embedding backend for OpenAI-compatible APIs
//!
//! This backend supports any OpenAI-compatible embedding API, including:
//! - OpenAI API
//! - Azure OpenAI
//! - Local servers (LM Studio, vLLM, Ollama with OpenAI compat, text-embeddings-inference)

use super::traits::{EmbeddingBackend, EmbeddingError, EmbeddingResult};
use crate::types::Embedding;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Configuration for the HTTP embedding backend
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// API endpoint (e.g., "https://api.openai.com/v1/embeddings")
    pub endpoint: String,
    /// API key (optional, can be from env var)
    pub api_key: Option<String>,
    /// Model name (e.g., "text-embedding-3-small")
    pub model: String,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum batch size for requests
    pub max_batch_size: usize,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.openai.com/v1/embeddings".to_string(),
            api_key: None,
            model: "text-embedding-3-small".to_string(),
            dimensions: 1536,
            timeout_secs: 30,
            max_batch_size: 100,
        }
    }
}

/// HTTP embedding backend for OpenAI-compatible APIs
#[derive(Debug)]
pub struct HttpBackend {
    client: Client,
    config: HttpConfig,
}

/// OpenAI embedding request format
#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a str>,
}

/// OpenAI embedding response format
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// OpenAI error response format
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ApiError,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ApiError {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
}

impl HttpBackend {
    /// Create a new HTTP embedding backend
    pub fn new(config: HttpConfig) -> EmbeddingResult<Self> {
        info!(
            "Initializing HTTP embedding backend: endpoint={}, model={}",
            config.endpoint, config.model
        );

        // Build headers
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Get API key from config or environment
        let api_key = config
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok());

        if let Some(key) = &api_key {
            let auth_value = format!("Bearer {}", key);
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&auth_value)
                    .map_err(|e| EmbeddingError::Config(format!("Invalid API key format: {}", e)))?,
            );
        } else {
            // Check if the endpoint looks like it needs authentication
            if config.endpoint.contains("openai.com") || config.endpoint.contains("azure.com") {
                warn!("No API key provided for {}", config.endpoint);
            }
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .default_headers(headers)
            .build()
            .map_err(|e| EmbeddingError::Config(format!("Failed to build HTTP client: {}", e)))?;

        info!(
            "HTTP embedding backend initialized: {} dimensions",
            config.dimensions
        );

        Ok(Self { client, config })
    }

    /// Make an embedding request to the API
    fn request_embeddings(&self, texts: &[&str]) -> EmbeddingResult<Vec<Embedding>> {
        let request = EmbeddingRequest {
            model: &self.config.model,
            input: texts.to_vec(),
            // Only send dimensions if using a model that supports it (OpenAI text-embedding-3-*)
            dimensions: if self.config.model.contains("text-embedding-3") {
                Some(self.config.dimensions)
            } else {
                None
            },
            encoding_format: Some("float"),
        };

        debug!(
            "Sending embedding request to {} for {} texts",
            self.config.endpoint,
            texts.len()
        );

        // Serialize request body upfront so we can send it from a scoped thread.
        // reqwest::blocking::Client panics when called from within a tokio runtime,
        // so we run the actual HTTP call on a separate thread.
        let body = serde_json::to_vec(&request)
            .map_err(|e| EmbeddingError::EmbeddingFailed(format!("Failed to serialize request: {}", e)))?;

        let response = std::thread::scope(|s| {
            s.spawn(|| {
                self.client
                    .post(&self.config.endpoint)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(body)
                    .send()
            })
            .join()
        })
        .map_err(|_| EmbeddingError::EmbeddingFailed("HTTP request thread panicked".to_string()))?
        .map_err(|e| EmbeddingError::EmbeddingFailed(format!("HTTP request failed: {}", e)))?;

        let status = response.status();

        // Handle rate limiting
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .map(|s| s * 1000); // Convert to milliseconds

            return Err(EmbeddingError::RateLimited {
                retry_after_ms: retry_after,
            });
        }

        // Handle other errors
        if !status.is_success() {
            let error_text = response.text().unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse as API error
            if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&error_text) {
                return Err(EmbeddingError::EmbeddingFailed(format!(
                    "API error ({}): {}",
                    status, error_response.error.message
                )));
            }

            return Err(EmbeddingError::EmbeddingFailed(format!(
                "HTTP error ({}): {}",
                status, error_text
            )));
        }

        // Parse successful response
        let embedding_response: EmbeddingResponse = response.json().map_err(|e| {
            EmbeddingError::EmbeddingFailed(format!("Failed to parse response: {}", e))
        })?;

        if let Some(usage) = &embedding_response.usage {
            debug!(
                "Embedding request used {} tokens",
                usage.total_tokens
            );
        }

        // Sort by index and extract embeddings
        let mut data = embedding_response.data;
        data.sort_by_key(|d| d.index);

        let embeddings: Vec<Embedding> = data
            .into_iter()
            .map(|d| normalize_embedding(&d.embedding))
            .collect();

        Ok(embeddings)
    }
}

impl EmbeddingBackend for HttpBackend {
    fn embed(&self, text: &str) -> EmbeddingResult<Embedding> {
        let embeddings = self.request_embeddings(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::EmbeddingFailed("No embedding returned".to_string()))
    }

    fn embed_batch(&self, texts: &[String]) -> EmbeddingResult<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Process in batches if needed
        if text_refs.len() <= self.config.max_batch_size {
            return self.request_embeddings(&text_refs);
        }

        // Process larger batches in chunks
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in text_refs.chunks(self.config.max_batch_size) {
            let embeddings = self.request_embeddings(chunk)?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    fn name(&self) -> &str {
        "http"
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
    fn test_http_config_default() {
        let config = HttpConfig::default();
        assert_eq!(config.endpoint, "https://api.openai.com/v1/embeddings");
        assert_eq!(config.model, "text-embedding-3-small");
        assert_eq!(config.dimensions, 1536);
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_batch_size, 100);
    }

    #[test]
    fn test_normalize_embedding() {
        let embedding = vec![3.0, 4.0];
        let normalized = normalize_embedding(&embedding);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }
}
