//! Embedding engine with pluggable backends
//!
//! Supports multiple embedding providers:
//! - **HTTP backend**: OpenAI-compatible APIs (OpenAI, Azure, LM Studio, vLLM, etc.)
//! - **Local backend**: embed_anything with candle inference (CPU/CUDA/Metal)
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
//!
//! ## Legacy local (embed_anything)
//! ```toml
//! [embedding]
//! model_name = "all-MiniLM-L6-v2"
//! dimensions = 384
//! ```

pub mod backend;
mod engine;
pub mod model;
mod service;

pub use engine::*;
pub use model::{ModelInfo, ModelRegistry};
pub use service::{
    check_model_exists, generate_with_fallback, hash_based_embedding,
    init_embedding_engine, model_not_found_error,
};
