//! Document chunking system
//!
//! Features:
//! - Semantic-aware text splitting
//! - Configurable chunk sizes with overlap
//! - Section hierarchy detection

mod splitter;
mod tokenizer;

pub use splitter::*;
pub use tokenizer::*;
