//! Document chunking system
//!
//! Features:
//! - Semantic-aware text splitting
//! - Configurable chunk sizes with overlap
//! - Section hierarchy detection
//! - Pluggable tokenizers (heuristic or BPE)

mod splitter;
pub mod tokenizer;

pub use splitter::*;
pub use tokenizer::{
    create_tokenizer, BpeTokenizer, HeuristicTokenizer, SharedTokenizer, Tokenizer,
};
