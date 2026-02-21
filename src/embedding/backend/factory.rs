//! Backend factory for creating embedding backends from configuration
//!
//! Provides a unified way to create embedding backends based on configuration.

use super::http::{HttpBackend, HttpConfig};
use super::local::{LocalBackend, LocalConfig};
use super::traits::{EmbeddingBackend, EmbeddingResult};
use crate::config::BackendConfig;
use std::sync::Arc;
use tracing::info;

/// Create an embedding backend from configuration
///
/// Returns an `Arc<dyn EmbeddingBackend>` that can be shared across threads.
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

        BackendConfig::Local {
            model_name,
            dimensions,
            truncated_dimensions,
            max_sequence_length,
        } => {
            info!("Creating local embedding backend: model={}", model_name);

            let local_config = LocalConfig {
                model_name: model_name.clone(),
                dimensions: *dimensions,
                truncated_dimensions: truncated_dimensions.unwrap_or(*dimensions),
                max_sequence_length: *max_sequence_length,
            };

            let backend = LocalBackend::new(local_config)?;
            Ok(Arc::new(backend))
        }
    }
}

/// Create a backend from legacy EmbeddingConfig (backward compatibility)
///
/// This function is provided for backward compatibility with the old
/// EmbeddingConfig structure. It creates a LocalBackend.
pub fn create_backend_from_legacy(
    config: &crate::config::EmbeddingConfig,
) -> EmbeddingResult<Arc<dyn EmbeddingBackend>> {
    // Check if a backend config is specified
    if let Some(backend_config) = &config.backend {
        return create_backend(backend_config);
    }

    // Fall back to local backend using legacy fields
    info!(
        "Creating local embedding backend from legacy config: model={}",
        config.model_name
    );

    let local_config = LocalConfig {
        model_name: config.model_name.clone(),
        dimensions: config.dimensions,
        truncated_dimensions: config.truncated_dimensions,
        max_sequence_length: config.max_sequence_length,
    };

    let backend = LocalBackend::new(local_config)?;
    Ok(Arc::new(backend))
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
}
