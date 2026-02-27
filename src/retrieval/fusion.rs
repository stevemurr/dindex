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

    // Normalize scores to [0, 1] range.
    // Theoretical max is num_lists/(k+1), achieved when a doc is rank 1 in all lists.
    let num_lists = ranked_lists.len();
    if num_lists > 0 {
        let theoretical_max = num_lists as f32 / (config.k as f32 + 1.0);
        for result in &mut results {
            result.rrf_score /= theoretical_max;
        }
    }

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

    #[test]
    fn test_rrf_single_list() {
        let results = vec![
            RankedResult {
                chunk_id: "a".to_string(),
                rank: 1,
                original_score: 0.9,
                method: RetrievalMethod::Dense,
            },
            RankedResult {
                chunk_id: "b".to_string(),
                rank: 2,
                original_score: 0.7,
                method: RetrievalMethod::Dense,
            },
        ];

        let config = RrfConfig { k: 60 };
        let fused = reciprocal_rank_fusion(&[results], &config);

        assert_eq!(fused.len(), 2);
        // Scores are normalized: rank 1 in single list = 1.0, rank 2 = (k+1)/(k+2)
        assert_eq!(fused[0].chunk_id, "a");
        assert_eq!(fused[1].chunk_id, "b");

        let expected_a = 1.0;
        let expected_b = 61.0 / 62.0;
        assert!((fused[0].rrf_score - expected_a).abs() < 1e-6);
        assert!((fused[1].rrf_score - expected_b).abs() < 1e-6);

        // Each result should only have one contributing method
        assert_eq!(fused[0].contributing_methods.len(), 1);
        assert_eq!(fused[0].contributing_methods[0], RetrievalMethod::Dense);
    }

    #[test]
    fn test_rrf_empty_lists() {
        let config = RrfConfig::default();

        // No lists at all
        let fused = reciprocal_rank_fusion(&[], &config);
        assert!(fused.is_empty());

        // Single empty list
        let fused = reciprocal_rank_fusion(&[vec![]], &config);
        assert!(fused.is_empty());

        // Multiple empty lists
        let fused = reciprocal_rank_fusion(&[vec![], vec![]], &config);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_with_one_empty_list() {
        let results = vec![RankedResult {
            chunk_id: "only".to_string(),
            rank: 1,
            original_score: 1.0,
            method: RetrievalMethod::Bm25,
        }];

        let config = RrfConfig { k: 60 };
        let fused = reciprocal_rank_fusion(&[results, vec![]], &config);

        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].chunk_id, "only");
    }

    #[test]
    fn test_retrieval_method_as_str() {
        assert_eq!(RetrievalMethod::Dense.as_str(), "dense");
        assert_eq!(RetrievalMethod::Bm25.as_str(), "bm25");
    }

    #[test]
    fn test_retrieval_method_display() {
        assert_eq!(format!("{}", RetrievalMethod::Dense), "dense");
        assert_eq!(format!("{}", RetrievalMethod::Bm25), "bm25");
    }

    #[test]
    fn test_to_ranked_results() {
        let raw = vec![
            ("c1".to_string(), 0.9f32),
            ("c2".to_string(), 0.7f32),
            ("c3".to_string(), 0.5f32),
        ];

        let ranked = to_ranked_results(&raw, RetrievalMethod::Dense);

        assert_eq!(ranked.len(), 3);
        // Ranks should be 1-indexed
        assert_eq!(ranked[0].rank, 1);
        assert_eq!(ranked[0].chunk_id, "c1");
        assert_eq!(ranked[0].original_score, 0.9);
        assert_eq!(ranked[0].method, RetrievalMethod::Dense);

        assert_eq!(ranked[1].rank, 2);
        assert_eq!(ranked[2].rank, 3);
    }

    #[test]
    fn test_to_ranked_results_empty() {
        let raw: Vec<(String, f32)> = vec![];
        let ranked = to_ranked_results(&raw, RetrievalMethod::Bm25);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_rrf_config_default() {
        let config = RrfConfig::default();
        assert_eq!(config.k, 60);
    }

    #[test]
    fn test_rrf_rank_per_method_tracking() {
        let dense = vec![RankedResult {
            chunk_id: "x".to_string(),
            rank: 3,
            original_score: 0.5,
            method: RetrievalMethod::Dense,
        }];
        let bm25 = vec![RankedResult {
            chunk_id: "x".to_string(),
            rank: 1,
            original_score: 5.0,
            method: RetrievalMethod::Bm25,
        }];

        let config = RrfConfig::default();
        let fused = reciprocal_rank_fusion(&[dense, bm25], &config);

        assert_eq!(fused.len(), 1);
        let result = &fused[0];
        assert_eq!(result.rank_per_method[&RetrievalMethod::Dense], 3);
        assert_eq!(result.rank_per_method[&RetrievalMethod::Bm25], 1);

        // Normalized RRF: (1/63 + 1/61) / (2/61) = 62/63
        let expected = 62.0 / 63.0;
        assert!((result.rrf_score - expected).abs() < 1e-6);
    }
}
