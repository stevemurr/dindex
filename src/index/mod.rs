//! Vector index using USearch (HNSW)
//!
//! Features:
//! - High-performance HNSW index with INT8 quantization
//! - Memory-mapped storage for disk-based operation
//! - Batch insertion and querying
//! - Document registry for deduplication
//! - Unified document processor

mod hnsw;
mod processor;
mod registry;
mod storage;

pub use hnsw::*;
pub use processor::*;
pub use registry::*;
pub use storage::*;
