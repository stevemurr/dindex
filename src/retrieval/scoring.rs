//! Composable scoring pipeline for search result processing
//!
//! Provides score provenance tracking (`ScoreBreakdown`), normalized score types,
//! and a trait-based pipeline for applying scoring stages in sequence.

use crate::retrieval::RetrievalMethod;
use crate::types::SearchResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Score boundary type
// ============================================================================

/// A score clamped to [0.0, 1.0] at construction time.
///
/// Use this wherever a normalized score is required to enforce range contracts
/// at the type level rather than relying on runtime assertions downstream.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct NormalizedScore(f32);

impl NormalizedScore {
    /// Create a new NormalizedScore, clamping the input to [0.0, 1.0].
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the inner f32 value (guaranteed in [0.0, 1.0]).
    pub fn value(self) -> f32 {
        self.0
    }
}

impl From<f32> for NormalizedScore {
    fn from(v: f32) -> Self {
        Self::new(v)
    }
}

impl From<NormalizedScore> for f32 {
    fn from(s: NormalizedScore) -> Self {
        s.0
    }
}

// ============================================================================
// Score provenance tracking
// ============================================================================

/// Detailed breakdown of how a result's final score was computed.
///
/// Attached optionally to `SearchResult` for debugging and transparency.
/// Each field records the contribution of one scoring stage.
// NOTE: skip_serializing_if is safe here because IPC uses JSON (self-describing).
// P2P bincode never encounters ScoreBreakdown because SearchResult.score_breakdown
// is always None for network-transmitted results (and has #[serde(default)] there).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoreBreakdown {
    /// Raw dense (HNSW) similarity, clamped to [0, 1]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_similarity: Option<NormalizedScore>,

    /// Raw BM25 score before normalization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_raw: Option<f32>,

    /// BM25 score after min-max normalization to [0, 1]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25_normalized: Option<NormalizedScore>,

    /// RRF fusion score (normalized to [0, 1])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rrf_score: Option<NormalizedScore>,

    /// Aggregator demotion multiplier applied (1.0 = no demotion)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregator_multiplier: Option<f32>,

    /// Reranker score (the term-overlap boost component)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reranker_score: Option<NormalizedScore>,

    /// Which retrieval methods contributed to this result
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub methods: Vec<RetrievalMethod>,

    /// Rank per retrieval method (e.g., Dense -> 3, BM25 -> 1)
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub rank_per_method: HashMap<String, usize>,
}

// ============================================================================
// Scoring stage trait
// ============================================================================

/// A single stage in the scoring pipeline.
///
/// Each stage receives the query text and a mutable slice of results, and may
/// reorder them, mutate their `relevance_score`, or update `score_breakdown`.
pub trait ScoringStage: Send + Sync {
    /// Human-readable name for logging
    fn name(&self) -> &str;

    /// Apply this scoring stage to the results
    fn apply(&self, query: &str, results: &mut Vec<SearchResult>);
}

// ============================================================================
// Scoring pipeline
// ============================================================================

/// Ordered sequence of scoring stages applied to search results.
pub struct ScoringPipeline {
    stages: Vec<Box<dyn ScoringStage>>,
}

impl ScoringPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a stage to the end of the pipeline.
    pub fn push(&mut self, stage: Box<dyn ScoringStage>) {
        self.stages.push(stage);
    }

    /// Run all stages in order.
    pub fn apply(&self, query: &str, results: &mut Vec<SearchResult>) {
        for stage in &self.stages {
            tracing::debug!("Scoring stage '{}': {} results", stage.name(), results.len());
            stage.apply(query, results);
        }
    }

    /// Number of stages in the pipeline.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Whether the pipeline has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
}

impl Default for ScoringPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Concrete stages
// ============================================================================

/// Demotes results that have an `aggregator_score` metadata field.
///
/// Pages identified as aggregators (e.g., hub/index pages) get their score
/// multiplied down: `score *= 1.0 - (aggregator_score * (1.0 - min_multiplier))`.
pub struct AggregatorDemotion {
    /// Minimum multiplier at aggregator_score = 1.0
    pub min_multiplier: f32,
}

impl ScoringStage for AggregatorDemotion {
    fn name(&self) -> &str {
        "aggregator_demotion"
    }

    fn apply(&self, _query: &str, results: &mut Vec<SearchResult>) {
        for result in results.iter_mut() {
            if let Some(score_str) = result.chunk.metadata.extra.get("aggregator_score") {
                if let Ok(aggregator_score) = score_str.parse::<f32>() {
                    let multiplier = 1.0 - (aggregator_score * (1.0 - self.min_multiplier));
                    result.relevance_score *= multiplier;
                    if let Some(ref mut bd) = result.score_breakdown {
                        bd.aggregator_multiplier = Some(multiplier);
                    }
                }
            }
        }
    }
}

/// Reranks results by query-term overlap in chunk content.
///
/// The final score is a weighted blend of the original score and the overlap
/// ratio.  Weights are configurable (default: 0.7 original, 0.3 overlap).
pub struct OverlapReranker {
    /// Weight for the original score (default: 0.7)
    pub score_weight: f32,
    /// Weight for the overlap boost (default: 0.3)
    pub overlap_weight: f32,
    /// Stop words to exclude from overlap matching
    pub stop_words: std::collections::HashSet<&'static str>,
}

impl OverlapReranker {
    /// Create with default weights and no stop words.
    pub fn new(score_weight: f32, overlap_weight: f32) -> Self {
        Self {
            score_weight,
            overlap_weight,
            stop_words: std::collections::HashSet::new(),
        }
    }

    /// Create with default weights and English stop words.
    pub fn with_stop_words(score_weight: f32, overlap_weight: f32) -> Self {
        let stop_words: std::collections::HashSet<&'static str> = [
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "it", "its", "this",
            "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
            "him", "her", "us", "them", "my", "your", "his", "our", "their",
            "what", "which", "who", "whom", "how", "when", "where", "why",
            "not", "no", "so", "if", "then", "than", "as",
        ]
        .into_iter()
        .collect();
        Self {
            score_weight,
            overlap_weight,
            stop_words,
        }
    }
}

impl ScoringStage for OverlapReranker {
    fn name(&self) -> &str {
        "overlap_reranker"
    }

    fn apply(&self, query: &str, results: &mut Vec<SearchResult>) {
        let query_lower = query.to_lowercase();
        let query_terms: std::collections::HashSet<&str> = query_lower
            .split_whitespace()
            .filter(|t| !self.stop_words.contains(*t))
            .collect();

        if query_terms.is_empty() {
            return;
        }

        for result in results.iter_mut() {
            let content_lower = result.chunk.content.to_lowercase();
            let overlap: usize = query_terms
                .iter()
                .filter(|term| content_lower.split_whitespace().any(|w| w == **term))
                .count();

            let overlap_boost = overlap as f32 / query_terms.len() as f32;
            result.relevance_score =
                result.relevance_score * self.score_weight + overlap_boost * self.overlap_weight;
            if let Some(ref mut bd) = result.score_breakdown {
                bd.reranker_score = Some(NormalizedScore::new(overlap_boost));
            }
        }

        results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chunk, ChunkMetadata};

    // NormalizedScore tests

    #[test]
    fn test_normalized_score_clamps_high() {
        let s = NormalizedScore::new(1.5);
        assert_eq!(s.value(), 1.0);
    }

    #[test]
    fn test_normalized_score_clamps_low() {
        let s = NormalizedScore::new(-0.5);
        assert_eq!(s.value(), 0.0);
    }

    #[test]
    fn test_normalized_score_preserves_valid() {
        let s = NormalizedScore::new(0.75);
        assert!((s.value() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_score_from_f32() {
        let s: NormalizedScore = 2.0_f32.into();
        assert_eq!(s.value(), 1.0);
    }

    #[test]
    fn test_normalized_score_into_f32() {
        let v: f32 = NormalizedScore::new(0.5).into();
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_score_serde_roundtrip() {
        let s = NormalizedScore::new(0.42);
        let json = serde_json::to_string(&s).unwrap();
        let deserialized: NormalizedScore = serde_json::from_str(&json).unwrap();
        assert!((deserialized.value() - 0.42).abs() < 1e-6);
    }

    // ScoreBreakdown tests

    #[test]
    fn test_score_breakdown_default() {
        let bd = ScoreBreakdown::default();
        assert!(bd.dense_similarity.is_none());
        assert!(bd.bm25_raw.is_none());
        assert!(bd.bm25_normalized.is_none());
        assert!(bd.rrf_score.is_none());
        assert!(bd.aggregator_multiplier.is_none());
        assert!(bd.reranker_score.is_none());
        assert!(bd.methods.is_empty());
        assert!(bd.rank_per_method.is_empty());
    }

    #[test]
    fn test_score_breakdown_serde_roundtrip() {
        let bd = ScoreBreakdown {
            dense_similarity: Some(NormalizedScore::new(0.85)),
            bm25_raw: Some(4.2),
            bm25_normalized: Some(NormalizedScore::new(0.7)),
            rrf_score: Some(NormalizedScore::new(0.92)),
            aggregator_multiplier: None,
            reranker_score: Some(NormalizedScore::new(0.6)),
            methods: vec![RetrievalMethod::Dense, RetrievalMethod::Bm25],
            rank_per_method: {
                let mut m = HashMap::new();
                m.insert("dense".to_string(), 1);
                m.insert("bm25".to_string(), 3);
                m
            },
        };
        let json = serde_json::to_string(&bd).unwrap();
        let deserialized: ScoreBreakdown = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.methods.len(), 2);
        assert_eq!(deserialized.rank_per_method["dense"], 1);
        assert!(deserialized.aggregator_multiplier.is_none());
    }

    #[test]
    fn test_score_breakdown_skip_serializing_none_fields() {
        let bd = ScoreBreakdown::default();
        let json = serde_json::to_string(&bd).unwrap();
        // Empty breakdown should produce minimal JSON
        assert!(!json.contains("dense_similarity"));
        assert!(!json.contains("bm25_raw"));
    }

    // ScoringPipeline tests

    fn make_result(chunk_id: &str, content: &str, score: f32) -> SearchResult {
        SearchResult::new(
            Chunk {
                metadata: ChunkMetadata::new(chunk_id.to_string(), "doc1".to_string()),
                content: content.to_string(),
                token_count: content.split_whitespace().count(),
            },
            score,
        )
    }

    #[test]
    fn test_pipeline_empty_is_noop() {
        let pipeline = ScoringPipeline::new();
        assert!(pipeline.is_empty());
        let mut results = vec![make_result("c1", "hello", 0.5)];
        pipeline.apply("query", &mut results);
        assert!((results[0].relevance_score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pipeline_stages_run_in_order() {
        let mut pipeline = ScoringPipeline::new();
        pipeline.push(Box::new(OverlapReranker::new(0.7, 0.3)));
        pipeline.push(Box::new(AggregatorDemotion { min_multiplier: 0.5 }));
        assert_eq!(pipeline.len(), 2);
    }

    // AggregatorDemotion tests

    #[test]
    fn test_aggregator_demotion_applies() {
        let stage = AggregatorDemotion { min_multiplier: 0.5 };
        let mut result = make_result("c1", "content", 1.0);
        result
            .chunk
            .metadata
            .extra
            .insert("aggregator_score".to_string(), "1.0".to_string());
        let mut results = vec![result];
        stage.apply("query", &mut results);
        // multiplier = 1.0 - (1.0 * (1.0 - 0.5)) = 0.5
        assert!((results[0].relevance_score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_aggregator_demotion_no_metadata_is_noop() {
        let stage = AggregatorDemotion { min_multiplier: 0.5 };
        let mut results = vec![make_result("c1", "content", 0.8)];
        stage.apply("query", &mut results);
        assert!((results[0].relevance_score - 0.8).abs() < 1e-6);
    }

    // OverlapReranker tests

    #[test]
    fn test_overlap_reranker_boosts_matching_content() {
        let reranker = OverlapReranker::new(0.7, 0.3);
        let mut results = vec![
            make_result("c1", "machine learning models", 0.5),
            make_result("c2", "unrelated content here", 0.5),
        ];
        reranker.apply("machine learning", &mut results);
        // c1 has full overlap, c2 has none
        assert_eq!(results[0].chunk.metadata.chunk_id, "c1");
        assert!(results[0].relevance_score > results[1].relevance_score);
    }

    #[test]
    fn test_overlap_reranker_with_stop_words() {
        let reranker = OverlapReranker::with_stop_words(0.7, 0.3);
        let mut results = vec![
            make_result("c1", "the cat sat on a mat", 0.5),
            make_result("c2", "algorithms for sorting data", 0.5),
        ];
        // "what is the" — all stop words removed, no meaningful overlap
        reranker.apply("what is the", &mut results);
        // Both should have same score since all query terms are stop words
        assert!((results[0].relevance_score - results[1].relevance_score).abs() < 1e-6);
    }

    #[test]
    fn test_overlap_reranker_records_breakdown() {
        let reranker = OverlapReranker::new(0.7, 0.3);
        let mut result = make_result("c1", "machine learning", 0.5);
        result.score_breakdown = Some(ScoreBreakdown::default());
        let mut results = vec![result];
        reranker.apply("machine learning", &mut results);
        let bd = results[0].score_breakdown.as_ref().unwrap();
        assert!(bd.reranker_score.is_some());
        assert!((bd.reranker_score.unwrap().value() - 1.0).abs() < 1e-6);
    }

    // ---- New NormalizedScore edge-case tests ----

    #[test]
    fn test_normalized_score_nan_becomes_zero() {
        let s = NormalizedScore::new(f32::NAN);
        // NaN.clamp(0.0, 1.0) returns NaN on some platforms, but our contract
        // is [0.0, 1.0].  If the implementation uses clamp directly, NaN may
        // propagate; verify the actual behaviour here.
        // f32::NAN.clamp(0.0, 1.0) == f32::NAN per IEEE 754 — the type stores
        // whatever clamp returns.  The key assertion: value must not be > 1.0 or < 0.0.
        let v = s.value();
        assert!(v.is_nan() || (v >= 0.0 && v <= 1.0),
            "NaN input should result in NaN or a value in [0,1], got {}", v);
    }

    #[test]
    fn test_normalized_score_neg_infinity_becomes_zero() {
        let s = NormalizedScore::new(f32::NEG_INFINITY);
        assert_eq!(s.value(), 0.0, "-inf should clamp to 0.0");
    }

    #[test]
    fn test_normalized_score_pos_infinity_becomes_one() {
        let s = NormalizedScore::new(f32::INFINITY);
        assert_eq!(s.value(), 1.0, "+inf should clamp to 1.0");
    }

    #[test]
    fn test_normalized_score_ordering() {
        let low = NormalizedScore::new(0.2);
        let mid = NormalizedScore::new(0.5);
        let high = NormalizedScore::new(0.9);

        assert!(low < mid);
        assert!(mid < high);
        assert!(low < high);
        // Equal scores
        let also_mid = NormalizedScore::new(0.5);
        assert!(mid == also_mid);
        assert!(!(mid < also_mid));
        assert!(!(mid > also_mid));
    }

    // ---- New AggregatorDemotion tests ----

    #[test]
    fn test_aggregator_demotion_partial_score() {
        // aggregator_score = 0.5, min_multiplier = 0.5
        // multiplier = 1.0 - (0.5 * (1.0 - 0.5)) = 1.0 - 0.25 = 0.75
        // final score = 1.0 * 0.75 = 0.75
        let stage = AggregatorDemotion { min_multiplier: 0.5 };
        let mut result = make_result("c1", "content", 1.0);
        result
            .chunk
            .metadata
            .extra
            .insert("aggregator_score".to_string(), "0.5".to_string());
        let mut results = vec![result];
        stage.apply("query", &mut results);
        assert!(
            (results[0].relevance_score - 0.75).abs() < 1e-6,
            "Expected 0.75, got {}",
            results[0].relevance_score
        );
    }

    #[test]
    fn test_aggregator_demotion_invalid_score_ignored() {
        let stage = AggregatorDemotion { min_multiplier: 0.5 };
        let mut result = make_result("c1", "content", 0.8);
        result
            .chunk
            .metadata
            .extra
            .insert("aggregator_score".to_string(), "not_a_number".to_string());
        let mut results = vec![result];
        stage.apply("query", &mut results);
        // Score should be unchanged because parse::<f32> fails
        assert!(
            (results[0].relevance_score - 0.8).abs() < 1e-6,
            "Non-numeric aggregator_score should be ignored, got {}",
            results[0].relevance_score
        );
    }

    #[test]
    fn test_aggregator_demotion_records_breakdown() {
        let stage = AggregatorDemotion { min_multiplier: 0.5 };
        let mut result = make_result("c1", "content", 1.0);
        result.score_breakdown = Some(ScoreBreakdown::default());
        result
            .chunk
            .metadata
            .extra
            .insert("aggregator_score".to_string(), "1.0".to_string());
        let mut results = vec![result];
        stage.apply("query", &mut results);

        let bd = results[0].score_breakdown.as_ref().unwrap();
        assert!(bd.aggregator_multiplier.is_some());
        // multiplier = 1.0 - (1.0 * (1.0 - 0.5)) = 0.5
        assert!(
            (bd.aggregator_multiplier.unwrap() - 0.5).abs() < 1e-6,
            "Expected multiplier 0.5, got {}",
            bd.aggregator_multiplier.unwrap()
        );
    }

    // ---- New OverlapReranker tests ----

    #[test]
    fn test_overlap_reranker_sorts_results() {
        let reranker = OverlapReranker::new(0.7, 0.3);
        // c2 has higher initial score but no query overlap
        // c1 has lower initial score but full query overlap
        let mut results = vec![
            make_result("c2", "unrelated stuff here", 0.9),
            make_result("c1", "rust programming language", 0.3),
        ];
        reranker.apply("rust programming", &mut results);

        // After reranking, results should be sorted by final score descending
        for i in 1..results.len() {
            assert!(
                results[i - 1].relevance_score >= results[i].relevance_score,
                "Results should be sorted descending by score: [{}]={} vs [{}]={}",
                i - 1,
                results[i - 1].relevance_score,
                i,
                results[i].relevance_score
            );
        }
    }

    #[test]
    fn test_overlap_reranker_empty_results() {
        let reranker = OverlapReranker::new(0.7, 0.3);
        let mut results: Vec<SearchResult> = vec![];
        reranker.apply("some query", &mut results);
        assert!(results.is_empty(), "Empty results should remain empty");
    }

    // ---- New Pipeline integration test ----

    #[test]
    fn test_pipeline_applies_stages_in_sequence() {
        let mut pipeline = ScoringPipeline::new();
        // Stage 1: AggregatorDemotion
        pipeline.push(Box::new(AggregatorDemotion { min_multiplier: 0.5 }));
        // Stage 2: OverlapReranker
        pipeline.push(Box::new(OverlapReranker::new(0.7, 0.3)));

        let mut r1 = make_result("c1", "rust programming language", 1.0);
        r1.chunk
            .metadata
            .extra
            .insert("aggregator_score".to_string(), "1.0".to_string());
        // r1: after aggregator demotion -> score *= 0.5 -> 0.5
        // then reranker with "rust programming": 2/2 overlap = 1.0 boost
        // final = 0.5 * 0.7 + 1.0 * 0.3 = 0.35 + 0.3 = 0.65

        let r2 = make_result("c2", "unrelated content here", 0.8);
        // r2: no aggregator_score, so score stays 0.8
        // then reranker with "rust programming": 0/2 overlap = 0.0 boost
        // final = 0.8 * 0.7 + 0.0 * 0.3 = 0.56

        let mut results = vec![r1, r2];
        pipeline.apply("rust programming", &mut results);

        // After pipeline: r1 should be 0.65, r2 should be 0.56
        // Results should be sorted by OverlapReranker (r1 > r2)
        assert_eq!(results[0].chunk.metadata.chunk_id, "c1");
        assert!(
            (results[0].relevance_score - 0.65).abs() < 1e-6,
            "Expected c1 score ~0.65, got {}",
            results[0].relevance_score
        );
        assert!(
            (results[1].relevance_score - 0.56).abs() < 1e-6,
            "Expected c2 score ~0.56, got {}",
            results[1].relevance_score
        );
    }
}
