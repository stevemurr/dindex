//! Routing configuration

use serde::{Deserialize, Serialize};

/// Routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Number of centroids per node
    pub num_centroids: usize,
    /// LSH signature bits
    pub lsh_bits: usize,
    /// Number of LSH hash functions
    pub lsh_num_hashes: usize,
    /// Bloom filter size (bits per item)
    pub bloom_bits_per_item: usize,
    /// Number of candidate nodes for queries
    pub candidate_nodes: usize,
    /// Number of LSH bands for banding technique
    #[serde(default = "default_lsh_num_bands")]
    pub lsh_num_bands: usize,
    /// Centroid similarity threshold for matching (0.0-1.0)
    #[serde(default = "default_centroid_similarity_threshold")]
    pub centroid_similarity_threshold: f32,
    /// Bloom filter false positive rate per band (0.0-1.0)
    #[serde(default = "default_bloom_false_positive_rate")]
    pub bloom_false_positive_rate: f64,
}

fn default_lsh_num_bands() -> usize {
    8
}

fn default_centroid_similarity_threshold() -> f32 {
    0.5
}

fn default_bloom_false_positive_rate() -> f64 {
    0.01
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            num_centroids: 100,
            lsh_bits: 128,
            lsh_num_hashes: 8,
            bloom_bits_per_item: 10,
            candidate_nodes: 5,
            lsh_num_bands: default_lsh_num_bands(),
            centroid_similarity_threshold: default_centroid_similarity_threshold(),
            bloom_false_positive_rate: default_bloom_false_positive_rate(),
        }
    }
}
