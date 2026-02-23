//! Pluggable embedding backend system
//!
//! This module provides a trait-based abstraction for embedding backends,
//! allowing the use of different embedding providers:
//!
//! - **HTTP backend**: OpenAI-compatible APIs (OpenAI, Azure, LM Studio, vLLM, etc.)
//! - **Local backend**: embed_anything with candle inference (CPU/CUDA/Metal) (requires `local` feature)
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
//!
//! # Legacy local (embed_anything, requires --features local)
//! [embedding]
//! backend = "local"
//! model_name = "all-MiniLM-L6-v2"
//! dimensions = 384
//! ```

mod factory;
mod http;
#[cfg(feature = "local")]
mod local;
mod traits;

pub use factory::{create_backend, create_backend_from_legacy};
pub use http::{HttpBackend, HttpConfig};
#[cfg(feature = "local")]
pub use local::{LocalBackend, LocalConfig};
pub use traits::{EmbeddingBackend, EmbeddingError, EmbeddingResult};
