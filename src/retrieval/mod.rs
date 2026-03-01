//! Hybrid retrieval system
//!
//! Combines:
//! - Dense vector search (HNSW)
//! - BM25 lexical search
//! - Reciprocal Rank Fusion (RRF) for score aggregation

mod bm25;
mod fusion;
mod hybrid;
mod reranker;
pub mod scoring;
mod snippet;

pub use bm25::*;
pub use fusion::*;
pub use hybrid::*;
pub use reranker::*;
pub use scoring::{
    AggregatorDemotion, NormalizedScore, OverlapReranker, ScoreBreakdown, ScoringPipeline,
    ScoringStage,
};
pub use snippet::*;
