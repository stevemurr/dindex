//! BPE tokenizer wrapping tiktoken-rs for accurate token counting.
//!
//! This provides exact token counts that match the model's actual tokenizer,
//! ensuring chunks stay within model limits.

use super::Tokenizer;
use tiktoken_rs::CoreBPE;

/// BPE tokenizer backed by tiktoken-rs.
///
/// Supports cl100k_base (GPT-4, text-embedding-3-*) and o200k_base encodings.
/// The vocabulary is embedded in the binary, so this works fully offline.
pub struct BpeTokenizer {
    bpe: CoreBPE,
    encoding_name: String,
    max_tokens: Option<usize>,
}

impl BpeTokenizer {
    /// Create a BPE tokenizer for the given encoding name.
    ///
    /// Supported encodings: `"cl100k_base"`, `"o200k_base"`, `"p50k_base"`, `"r50k_base"`.
    pub fn new(encoding_name: &str) -> Result<Self, String> {
        let bpe = match encoding_name {
            "cl100k_base" => tiktoken_rs::cl100k_base()
                .map_err(|e| format!("Failed to load cl100k_base: {}", e))?,
            "o200k_base" => tiktoken_rs::o200k_base()
                .map_err(|e| format!("Failed to load o200k_base: {}", e))?,
            "p50k_base" => tiktoken_rs::p50k_base()
                .map_err(|e| format!("Failed to load p50k_base: {}", e))?,
            "r50k_base" => tiktoken_rs::r50k_base()
                .map_err(|e| format!("Failed to load r50k_base: {}", e))?,
            other => return Err(format!("Unknown encoding: {}", other)),
        };

        Ok(Self {
            bpe,
            encoding_name: encoding_name.to_string(),
            max_tokens: None,
        })
    }

    /// Set the maximum token length for this tokenizer's model.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl Tokenizer for BpeTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }

    fn name(&self) -> &str {
        &self.encoding_name
    }

    fn max_token_length(&self) -> Option<usize> {
        self.max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl100k_base_basic() {
        let tok = BpeTokenizer::new("cl100k_base").unwrap();
        let count = tok.count_tokens("Hello world");
        assert!(count >= 2, "should be at least 2 tokens, got {}", count);
    }

    #[test]
    fn test_cl100k_base_empty() {
        let tok = BpeTokenizer::new("cl100k_base").unwrap();
        assert_eq!(tok.count_tokens(""), 0);
    }

    #[test]
    fn test_unknown_encoding_errors() {
        let result = BpeTokenizer::new("nonexistent_encoding");
        assert!(result.is_err());
    }

    #[test]
    fn test_name_returns_encoding() {
        let tok = BpeTokenizer::new("cl100k_base").unwrap();
        assert_eq!(tok.name(), "cl100k_base");
    }

    #[test]
    fn test_max_token_length() {
        let tok = BpeTokenizer::new("cl100k_base").unwrap().with_max_tokens(8191);
        assert_eq!(tok.max_token_length(), Some(8191));
    }

    #[test]
    fn test_exact_token_count_known_text() {
        // "Hello, world!" in cl100k_base is typically 4 tokens
        let tok = BpeTokenizer::new("cl100k_base").unwrap();
        let count = tok.count_tokens("Hello, world!");
        assert!(
            count >= 3 && count <= 5,
            "'Hello, world!' should be ~4 tokens in cl100k_base, got {}",
            count
        );
    }

    #[test]
    fn test_cjk_text() {
        let tok = BpeTokenizer::new("cl100k_base").unwrap();
        let count = tok.count_tokens("你好世界");
        // Each CJK char is typically 1-2 tokens in cl100k_base
        assert!(count >= 2, "CJK should have multiple tokens, got {}", count);
    }
}
