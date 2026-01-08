//! Semantic routing for distributed query handling
//!
//! Features:
//! - Content centroids for node advertisement
//! - Locality-Sensitive Hashing (LSH) for fast similarity
//! - Bloom filters for negative filtering

mod bloom;
mod centroids;
mod lsh;
mod router;

pub use bloom::*;
pub use centroids::*;
pub use lsh::*;
pub use router::*;
