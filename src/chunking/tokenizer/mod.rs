//! Tokenizer abstraction for accurate token counting and text splitting.
//!
//! Provides a `Tokenizer` trait with two implementations:
//! - `HeuristicTokenizer`: improved word-level estimation (CJK-aware)
//! - `BpeTokenizer`: wraps tiktoken-rs for accurate BPE token counts
//!
//! Use `create_tokenizer` to resolve the best available tokenizer from config.

mod bpe;
mod factory;
mod heuristic;

pub use bpe::BpeTokenizer;
pub use factory::create_tokenizer;
pub use heuristic::HeuristicTokenizer;

use std::sync::Arc;

/// Shared tokenizer handle (thread-safe).
pub type SharedTokenizer = Arc<dyn Tokenizer>;

/// Trait for counting and producing tokens from text.
///
/// Implementations range from heuristic (fast, approximate) to
/// BPE-based (exact, matches model tokenizer).
pub trait Tokenizer: Send + Sync {
    /// Count the number of tokens in `text`.
    fn count_tokens(&self, text: &str) -> usize;

    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// Maximum token length this tokenizer's model supports, if known.
    fn max_token_length(&self) -> Option<usize> {
        None
    }
}
