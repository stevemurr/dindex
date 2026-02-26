//! Result fusion strategies
//!
//! Implements Reciprocal Rank Fusion (RRF) for combining results
//! from multiple retrieval methods

use crate::types::ChunkId;
use std::collections::HashMap;

/// Retrieval method identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetrievalMethod {
    Dense,
    Bm25,
}

impl RetrievalMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Bm25 => "bm25",
        }
    }
}

impl std::fmt::Display for RetrievalMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

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
    pub method: RetrievalMethod,
}

/// Fused result after combining multiple ranking sources
#[derive(Debug, Clone)]
pub struct FusedResult {
    pub chunk_id: ChunkId,
    pub rrf_score: f32,
    pub contributing_methods: Vec<RetrievalMethod>,
    pub rank_per_method: HashMap<RetrievalMethod, usize>,
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
                        fused.contributing_methods.push(result.method);
                    }
                    fused.rank_per_method.insert(result.method, result.rank);
                })
                .or_insert_with(|| {
                    let mut rank_per_method = HashMap::new();
                    rank_per_method.insert(result.method, result.rank);
                    FusedResult {
                        chunk_id: result.chunk_id.clone(),
                        rrf_score: rrf_contribution,
                        contributing_methods: vec![result.method],
                        rank_per_method,
                    }
                });
        }
    }

    // Sort by RRF score descending
    let mut results: Vec<FusedResult> = chunk_scores.into_values().collect();
    results.sort_by(|a, b| b.rrf_score.total_cmp(&a.rrf_score));

    results
}

/// Convert raw search results to ranked results
pub fn to_ranked_results(
    results: &[(ChunkId, f32)],
    method: RetrievalMethod,
) -> Vec<RankedResult> {
    results
        .iter()
        .enumerate()
        .map(|(rank, (chunk_id, score))| RankedResult {
            chunk_id: chunk_id.clone(),
            rank: rank + 1, // 1-indexed ranks
            original_score: *score,
            method,
        })
        .collect()
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
                method: RetrievalMethod::Dense,
            },
            RankedResult {
                chunk_id: "chunk2".to_string(),
                rank: 2,
                original_score: 0.80,
                method: RetrievalMethod::Dense,
            },
            RankedResult {
                chunk_id: "chunk3".to_string(),
                rank: 3,
                original_score: 0.70,
                method: RetrievalMethod::Dense,
            },
        ];

        let bm25_results = vec![
            RankedResult {
                chunk_id: "chunk2".to_string(),
                rank: 1,
                original_score: 5.2,
                method: RetrievalMethod::Bm25,
            },
            RankedResult {
                chunk_id: "chunk1".to_string(),
                rank: 2,
                original_score: 4.1,
                method: RetrievalMethod::Bm25,
            },
            RankedResult {
                chunk_id: "chunk4".to_string(),
                rank: 3,
                original_score: 3.5,
                method: RetrievalMethod::Bm25,
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
