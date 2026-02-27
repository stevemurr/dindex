//! Pluggable embedding backend system
//!
//! This module provides a trait-based abstraction for embedding backends.
//!
//! - **HTTP backend**: OpenAI-compatible APIs (OpenAI, Azure, LM Studio, vLLM, Ollama, etc.)
//!
//! # Example Configuration
//!
//! ```toml
//! # OpenAI
//! [embedding]
//! backend = "http"
//! endpoint = "https://api.openai.com/v1/embeddings"
//! model = "text-embedding-3-small"
//! dimensions = 1536
//! # api_key from OPENAI_API_KEY env var
//!
//! # Local LM Studio / vLLM
//! [embedding]
//! backend = "http"
//! endpoint = "http://localhost:1234/v1/embeddings"
//! model = "nomic-embed-text-v1.5"
//! dimensions = 768
//! ```

mod factory;
mod http;
mod traits;

pub use factory::{create_backend, create_backend_from_legacy};
pub use http::{HttpBackend, HttpConfig};
pub use traits::{EmbeddingBackend, EmbeddingError, EmbeddingResult};
