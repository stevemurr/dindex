//! Tokenizer factory: resolves the best tokenizer from config.

use super::{BpeTokenizer, HeuristicTokenizer, SharedTokenizer};
use crate::config::EmbeddingConfig;
use std::sync::Arc;
use tracing::{info, warn};

/// Create the best available tokenizer for the given embedding config.
///
/// Resolution order:
/// 1. Explicit `tokenizer_encoding` in config (e.g., `"cl100k_base"`)
/// 2. Model name lookup in hardcoded map
/// 3. Fallback to `HeuristicTokenizer` with a warning
pub fn create_tokenizer(config: &EmbeddingConfig) -> SharedTokenizer {
    // 1. Explicit encoding in config
    if let Some(ref encoding) = config.tokenizer_encoding {
        match BpeTokenizer::new(encoding) {
            Ok(tok) => {
                info!("Using BPE tokenizer: {}", encoding);
                return Arc::new(tok);
            }
            Err(e) => {
                warn!("Failed to create BPE tokenizer for encoding '{}': {}", encoding, e);
            }
        }
    }

    // 2. Infer from model name
    let model_name = config
        .model
        .as_deref()
        .unwrap_or(&config.model_name);

    if let Some(encoding) = encoding_for_model(model_name) {
        match BpeTokenizer::new(encoding) {
            Ok(tok) => {
                info!(
                    "Inferred BPE tokenizer '{}' from model '{}'",
                    encoding, model_name
                );
                return Arc::new(tok);
            }
            Err(e) => {
                warn!(
                    "Failed to create inferred tokenizer '{}' for model '{}': {}",
                    encoding, model_name, e
                );
            }
        }
    }

    // 3. Fallback
    warn!(
        "No BPE tokenizer available for model '{}'; using heuristic tokenizer. \
         Token counts will be approximate. Set `tokenizer_encoding` in config for accuracy.",
        model_name
    );
    Arc::new(HeuristicTokenizer::new())
}

/// Map well-known embedding model names to their tiktoken encoding.
fn encoding_for_model(model: &str) -> Option<&'static str> {
    let model_lower = model.to_lowercase();

    // OpenAI text-embedding-3-* and text-embedding-ada-002 use cl100k_base
    if model_lower.starts_with("text-embedding-3-")
        || model_lower.starts_with("text-embedding-ada")
    {
        return Some("cl100k_base");
    }

    // BGE models (BAAI) — use cl100k_base as best approximation
    // BGE-M3 uses its own tokenizer but cl100k_base is close enough
    if model_lower.starts_with("bge-") || model_lower.contains("bge") {
        return Some("cl100k_base");
    }

    // Nomic embed models
    if model_lower.starts_with("nomic-embed") {
        return Some("cl100k_base");
    }

    // E5 models (Microsoft)
    if model_lower.starts_with("e5-") || model_lower.contains("/e5-") {
        return Some("cl100k_base");
    }

    // Cohere embed models
    if model_lower.starts_with("embed-") {
        return Some("cl100k_base");
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_for_openai_models() {
        assert_eq!(
            encoding_for_model("text-embedding-3-small"),
            Some("cl100k_base")
        );
        assert_eq!(
            encoding_for_model("text-embedding-3-large"),
            Some("cl100k_base")
        );
        assert_eq!(
            encoding_for_model("text-embedding-ada-002"),
            Some("cl100k_base")
        );
    }

    #[test]
    fn test_encoding_for_bge_models() {
        assert_eq!(encoding_for_model("bge-m3"), Some("cl100k_base"));
        assert_eq!(encoding_for_model("bge-large-en"), Some("cl100k_base"));
        assert_eq!(
            encoding_for_model("BAAI/bge-small-en-v1.5"),
            Some("cl100k_base")
        );
    }

    #[test]
    fn test_encoding_for_nomic() {
        assert_eq!(
            encoding_for_model("nomic-embed-text-v1.5"),
            Some("cl100k_base")
        );
    }

    #[test]
    fn test_encoding_for_unknown_model() {
        assert_eq!(encoding_for_model("my-custom-model"), None);
        assert_eq!(encoding_for_model("all-MiniLM-L6-v2"), None);
    }

    #[test]
    fn test_create_tokenizer_with_explicit_encoding() {
        let mut config = EmbeddingConfig::default();
        config.tokenizer_encoding = Some("cl100k_base".to_string());
        let tok = create_tokenizer(&config);
        assert_eq!(tok.name(), "cl100k_base");
    }

    #[test]
    fn test_create_tokenizer_with_model_inference() {
        let mut config = EmbeddingConfig::default();
        config.model = Some("text-embedding-3-small".to_string());
        let tok = create_tokenizer(&config);
        assert_eq!(tok.name(), "cl100k_base");
    }

    #[test]
    fn test_create_tokenizer_fallback_heuristic() {
        let config = EmbeddingConfig::default(); // model_name = "all-MiniLM-L6-v2"
        let tok = create_tokenizer(&config);
        assert_eq!(tok.name(), "heuristic");
    }

    #[test]
    fn test_encoding_for_e5_models() {
        assert_eq!(encoding_for_model("e5-small"), Some("cl100k_base"));
        assert_eq!(encoding_for_model("e5-large"), Some("cl100k_base"));
        assert_eq!(
            encoding_for_model("intfloat/e5-large-v2"),
            Some("cl100k_base")
        );
    }

    #[test]
    fn test_encoding_for_cohere_models() {
        assert_eq!(
            encoding_for_model("embed-english-v3.0"),
            Some("cl100k_base")
        );
        assert_eq!(
            encoding_for_model("embed-multilingual-v3.0"),
            Some("cl100k_base")
        );
    }

    #[test]
    fn test_create_tokenizer_with_invalid_encoding_falls_back() {
        // Explicit but invalid encoding should fall through to model inference or heuristic
        let mut config = EmbeddingConfig::default();
        config.tokenizer_encoding = Some("nonexistent_encoding_xyz".to_string());
        // model_name is "all-MiniLM-L6-v2" which is unknown, so should fall back to heuristic
        let tok = create_tokenizer(&config);
        assert_eq!(
            tok.name(),
            "heuristic",
            "invalid encoding with unknown model should fall back to heuristic"
        );
    }

    #[test]
    fn test_create_tokenizer_model_name_used_as_fallback() {
        // When config.model is None, model_name should be used for inference
        let mut config = EmbeddingConfig::default();
        config.model = None;
        config.model_name = "bge-m3".to_string();
        config.tokenizer_encoding = None;
        let tok = create_tokenizer(&config);
        // bge-m3 is recognized by encoding_for_model, so should get BPE
        assert_eq!(
            tok.name(),
            "cl100k_base",
            "model_name 'bge-m3' should be used when config.model is None"
        );
    }
}
