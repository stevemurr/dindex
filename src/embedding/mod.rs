//! Embedding engine using embed_anything
//!
//! Features:
//! - CPU inference by default
//! - CUDA GPU acceleration (--features cuda)
//! - Metal GPU acceleration on Apple Silicon (--features metal)
//! - Support for Matryoshka embeddings (variable dimension truncation)
//! - Automatic model downloading via HuggingFace hub
//!
//! Default model: BGE-M3 (1024 dimensions, multilingual, best quality)

mod engine;
pub mod model;
mod quantize;
mod service;

pub use engine::*;
pub use model::{ModelInfo, ModelRegistry};
pub use quantize::*;
pub use service::{check_model_exists, hash_based_embedding, init_embedding_engine, model_not_found_error};
