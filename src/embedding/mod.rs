//! Embedding engine using ONNX Runtime
//!
//! Features:
//! - CPU-optimized inference with INT8 quantization
//! - Support for Matryoshka embeddings (variable dimension truncation)
//! - Batch processing for efficiency

mod engine;
mod model;
mod quantize;

pub use engine::*;
pub use model::*;
pub use quantize::*;
