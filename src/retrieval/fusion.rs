//! Result fusion strategies
//!
//! Implements Reciprocal Rank Fusion (RRF) for combining results
//! from multiple retrieval methods

use crate::types::{ChunkId, SearchResult};
use std::collections::HashMap;

/// Reciprocal Rank Fusion (RRF) parameters
#[derive(Debug, Clone)]
pub struct RrfConfig {
    /// K parameter for RRF (default: 60)
    pub k: usize,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { k: 60 }
    }
}

/// A ranked result from a single retrieval method
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub chunk_id: ChunkId,
    pub rank: usize,
    pub original_score: f32,
    pub method: String,
}

/// Fused result after combining multiple ranking sources
#[derive(Debug, Clone)]
pub struct FusedResult {
    pub chunk_id: ChunkId,
    pub rrf_score: f32,
    pub contributing_methods: Vec<String>,
    pub rank_per_method: HashMap<String, usize>,
}

/// Compute Reciprocal Rank Fusion score for multiple ranking lists
///
/// RRF score = Σ 1/(k + rank_r(d)) for all rankers r
///
/// This method works on ranks rather than scores, requiring no calibration
/// across heterogeneous retrieval methods.
pub fn reciprocal_rank_fusion(
    ranked_lists: &[Vec<RankedResult>],
    config: &RrfConfig,
) -> Vec<FusedResult> {
    // Aggregate scores per chunk
    let mut chunk_scores: HashMap<ChunkId, FusedResult> = HashMap::new();

    for results in ranked_lists {
        for result in results {
            let rrf_contribution = 1.0 / (config.k as f32 + result.rank as f32);

            chunk_scores
                .entry(result.chunk_id.clone())
                .and_modify(|fused| {
                    fused.rrf_score += rrf_contribution;
                    if !fused.contributing_methods.contains(&result.method) {
                        fused.contributing_methods.push(result.method.clone());
                    }
                    fused.rank_per_method.insert(result.method.clone(), result.rank);
                })
                .or_insert_with(|| {
                    let mut rank_per_method = HashMap::new();
                    rank_per_method.insert(result.method.clone(), result.rank);
                    FusedResult {
                        chunk_id: result.chunk_id.clone(),
                        rrf_score: rrf_contribution,
                        contributing_methods: vec![result.method.clone()],
                        rank_per_method,
                    }
                });
        }
    }

    // Sort by RRF score descending
    let mut results: Vec<FusedResult> = chunk_scores.into_values().collect();
    results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap());

    results
}

/// Convert raw search results to ranked results
pub fn to_ranked_results(
    results: &[(ChunkId, f32)],
    method: &str,
) -> Vec<RankedResult> {
    results
        .iter()
        .enumerate()
        .map(|(rank, (chunk_id, score))| RankedResult {
            chunk_id: chunk_id.clone(),
            rank: rank + 1, // 1-indexed ranks
            original_score: *score,
            method: method.to_string(),
        })
        .collect()
}

/// Simple linear combination of scores (alternative to RRF)
pub fn linear_combination(
    ranked_lists: &[Vec<RankedResult>],
    weights: &[f32],
) -> Vec<(ChunkId, f32)> {
    assert_eq!(ranked_lists.len(), weights.len());

    let mut chunk_scores: HashMap<ChunkId, f32> = HashMap::new();

    for (results, &weight) in ranked_lists.iter().zip(weights.iter()) {
        // Normalize scores within each list
        let max_score = results
            .iter()
            .map(|r| r.original_score)
            .fold(f32::MIN, f32::max);
        let min_score = results
            .iter()
            .map(|r| r.original_score)
            .fold(f32::MAX, f32::min);
        let range = max_score - min_score;

        for result in results {
            let normalized = if range > 0.0 {
                (result.original_score - min_score) / range
            } else {
                1.0
            };

            *chunk_scores.entry(result.chunk_id.clone()).or_default() += weight * normalized;
        }
    }

    let mut results: Vec<(ChunkId, f32)> = chunk_scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fusion() {
        let dense_results = vec![
            RankedResult {
                chunk_id: "chunk1".to_string(),
                rank: 1,
                original_score: 0.95,
                method: "dense".to_string(),
            },
            RankedResult {
                chunk_id: "chunk2".to_string(),
                rank: 2,
                original_score: 0.80,
                method: "dense".to_string(),
            },
            RankedResult {
                chunk_id: "chunk3".to_string(),
                rank: 3,
                original_score: 0.70,
                method: "dense".to_string(),
            },
        ];

        let bm25_results = vec![
            RankedResult {
                chunk_id: "chunk2".to_string(),
                rank: 1,
                original_score: 5.2,
                method: "bm25".to_string(),
            },
            RankedResult {
                chunk_id: "chunk1".to_string(),
                rank: 2,
                original_score: 4.1,
                method: "bm25".to_string(),
            },
            RankedResult {
                chunk_id: "chunk4".to_string(),
                rank: 3,
                original_score: 3.5,
                method: "bm25".to_string(),
            },
        ];

        let config = RrfConfig::default();
        let fused = reciprocal_rank_fusion(&[dense_results, bm25_results], &config);

        // chunk1 and chunk2 should be top ranked since they appear in both
        assert!(fused.len() >= 2);
        let top_ids: Vec<&str> = fused.iter().take(2).map(|r| r.chunk_id.as_str()).collect();
        assert!(top_ids.contains(&"chunk1"));
        assert!(top_ids.contains(&"chunk2"));

        // Check that contributing methods are tracked
        let chunk1_result = fused.iter().find(|r| r.chunk_id == "chunk1").unwrap();
        assert_eq!(chunk1_result.contributing_methods.len(), 2);
    }
}
