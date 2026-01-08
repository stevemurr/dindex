//! Vector index using USearch (HNSW)
//!
//! Features:
//! - High-performance HNSW index with INT8 quantization
//! - Memory-mapped storage for disk-based operation
//! - Batch insertion and querying

mod hnsw;
mod storage;

pub use hnsw::*;
pub use storage::*;
