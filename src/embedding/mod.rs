//! Embedding engine with pluggable HTTP backends
//!
//! Supports OpenAI-compatible HTTP APIs for embedding generation:
//! - OpenAI, Azure OpenAI
//! - vLLM, Ollama, LM Studio
//! - text-embeddings-inference
//!
//! # Example Configuration
//!
//! ## OpenAI API
//! ```toml
//! [embedding]
//! backend = "http"
//! endpoint = "https://api.openai.com/v1/embeddings"
//! model = "text-embedding-3-small"
//! dimensions = 1536
//! ```
//!
//! ## Local LM Studio / vLLM
//! ```toml
//! [embedding]
//! backend = "http"
//! endpoint = "http://localhost:1234/v1/embeddings"
//! model = "nomic-embed-text-v1.5"
//! dimensions = 768
//! ```

pub mod backend;
mod engine;
mod service;

pub use engine::*;
pub use service::{
    generate_with_fallback, hash_based_embedding,
    init_embedding_engine,
};
