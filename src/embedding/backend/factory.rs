//! Backend factory for creating embedding backends from configuration

use super::http::{HttpBackend, HttpConfig};
use super::traits::{EmbeddingBackend, EmbeddingResult};
use crate::config::BackendConfig;
use std::sync::Arc;
use tracing::info;

/// Create an embedding backend from configuration
pub fn create_backend(config: &BackendConfig) -> EmbeddingResult<Arc<dyn EmbeddingBackend>> {
    match config {
        BackendConfig::Http {
            endpoint,
            api_key,
            model,
            dimensions,
            timeout_secs,
            max_batch_size,
        } => {
            info!("Creating HTTP embedding backend: endpoint={}", endpoint);

            let http_config = HttpConfig {
                endpoint: endpoint.clone(),
                api_key: api_key.clone(),
                model: model.clone(),
                dimensions: *dimensions,
                timeout_secs: *timeout_secs,
                max_batch_size: *max_batch_size,
            };

            let backend = HttpBackend::new(http_config)?;
            Ok(Arc::new(backend))
        }
    }
}

/// Create a backend from legacy EmbeddingConfig (backward compatibility)
///
/// Resolves the backend configuration from EmbeddingConfig fields.
/// Falls back to an HTTP backend using the configured endpoint, or returns
/// an error if no HTTP endpoint is configured.
pub fn create_backend_from_legacy(
    config: &crate::config::EmbeddingConfig,
) -> EmbeddingResult<Arc<dyn EmbeddingBackend>> {
    // Check if a backend config is specified
    if let Some(backend_config) = config.resolve_backend() {
        return create_backend(&backend_config);
    }

    // No backend configured — require HTTP configuration
    Err(super::traits::EmbeddingError::Config(
        "No embedding backend configured. Add an HTTP backend to your config.toml:\n\
         [embedding]\n\
         backend = \"http\"\n\
         endpoint = \"http://localhost:8002/v1/embeddings\"\n\
         model = \"bge-m3\"\n\
         dimensions = 1024"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_http_backend() {
        let config = BackendConfig::Http {
            endpoint: "http://localhost:8080/v1/embeddings".to_string(),
            api_key: None,
            model: "test-model".to_string(),
            dimensions: 384,
            timeout_secs: 30,
            max_batch_size: 100,
        };

        // This should succeed (just creates the client, doesn't make requests)
        let result = create_backend(&config);
        assert!(result.is_ok());

        let backend = result.unwrap();
        assert_eq!(backend.name(), "http");
        assert_eq!(backend.dimensions(), 384);
    }

    #[test]
    fn test_create_backend_from_legacy_no_backend() {
        let config = crate::config::EmbeddingConfig::default();
        let result = create_backend_from_legacy(&config);
        assert!(result.is_err(), "should error when no backend configured");
    }
}
