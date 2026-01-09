//! Embedding engine using ONNX Runtime
//!
//! Features:
//! - CPU-optimized inference with INT8 quantization
//! - Support for Matryoshka embeddings (variable dimension truncation)
//! - Batch processing for efficiency

mod engine;
mod model;
mod quantize;
mod service;

pub use engine::*;
pub use model::*;
pub use quantize::*;
pub use service::{check_model_exists, hash_based_embedding, init_embedding_engine, model_not_found_error};
