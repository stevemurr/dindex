//! Index, chunking, and retrieval configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Vector index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// HNSW M parameter (connections per layer)
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter
    pub hnsw_ef_construction: usize,
    /// HNSW ef_search parameter
    pub hnsw_ef_search: usize,
    /// Enable memory mapping
    pub memory_mapped: bool,
    /// Maximum index capacity
    pub max_capacity: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 100,
            memory_mapped: true,
            max_capacity: 1_000_000,
        }
    }
}

/// Chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Base chunk size in tokens
    pub chunk_size: usize,
    /// Overlap between chunks (as fraction, e.g., 0.15)
    pub overlap_fraction: f32,
    /// Minimum chunk size (won't create smaller chunks)
    pub min_chunk_size: usize,
    /// Maximum chunk size
    pub max_chunk_size: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap_fraction: 0.15,
            min_chunk_size: 50,
            max_chunk_size: 1024,
        }
    }
}

/// Retrieval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Enable dense (vector) retrieval
    pub enable_dense: bool,
    /// Enable BM25 text retrieval
    pub enable_bm25: bool,
    /// RRF k parameter
    pub rrf_k: usize,
    /// Number of candidates to fetch before reranking
    pub candidate_count: usize,
    /// Enable cross-encoder reranking
    pub enable_reranking: bool,
    /// Reranker model path
    pub reranker_model_path: Option<PathBuf>,
    /// Minimum quality score before triggering adaptive fan-out (0.0-1.0)
    #[serde(default = "default_fanout_quality_threshold")]
    pub fanout_quality_threshold: f32,
    /// Minimum number of results before considering fan-out
    #[serde(default = "default_fanout_min_results")]
    pub fanout_min_results: usize,
    /// Maximum number of additional peers to query during fan-out
    #[serde(default = "default_max_fanout_peers")]
    pub max_fanout_peers: usize,
    /// Minimum average result score before triggering fan-out (0.0-1.0)
    #[serde(default = "default_fanout_score_threshold")]
    pub fanout_score_threshold: f32,
    /// Fraction of original timeout to use for tier-2 fan-out queries (0.0-1.0)
    #[serde(default = "default_fanout_timeout_fraction")]
    pub fanout_timeout_fraction: f32,
    /// Minimum score multiplier for aggregator pages (0.0-1.0).
    /// At aggregator_score=1.0, the result score is multiplied by this value.
    #[serde(default = "default_aggregator_min_multiplier")]
    pub aggregator_min_multiplier: f32,
}

fn default_fanout_quality_threshold() -> f32 {
    0.5
}

fn default_fanout_min_results() -> usize {
    3
}

fn default_max_fanout_peers() -> usize {
    10
}

fn default_fanout_score_threshold() -> f32 {
    0.3
}

fn default_fanout_timeout_fraction() -> f32 {
    0.5
}

fn default_aggregator_min_multiplier() -> f32 {
    0.5
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            enable_dense: true,
            enable_bm25: true,
            rrf_k: 60,
            candidate_count: 50,
            enable_reranking: true,
            reranker_model_path: None,
            fanout_quality_threshold: default_fanout_quality_threshold(),
            fanout_min_results: default_fanout_min_results(),
            max_fanout_peers: default_max_fanout_peers(),
            fanout_score_threshold: default_fanout_score_threshold(),
            fanout_timeout_fraction: default_fanout_timeout_fraction(),
            aggregator_min_multiplier: default_aggregator_min_multiplier(),
        }
    }
}
