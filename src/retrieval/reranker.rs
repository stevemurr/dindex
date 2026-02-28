//! Search result reranking
//!
//! Provides a simple query-term-overlap reranker for improving search result ordering.

use crate::types::SearchResult;

/// Simple reranker using query term overlap heuristic
pub struct SimpleReranker;

impl SimpleReranker {
    /// Rerank using a simple heuristic (query term overlap)
    pub fn rerank(query: &str, results: &mut [SearchResult]) {
        let query_lower = query.to_lowercase();
        let query_terms: std::collections::HashSet<&str> =
            query_lower.split_whitespace().collect();

        for result in results.iter_mut() {
            let content_lower = result.chunk.content.to_lowercase();
            let content_words: std::collections::HashSet<&str> =
                content_lower.split_whitespace().collect();
            let overlap: usize = query_terms
                .iter()
                .filter(|term| content_words.contains(*term))
                .count();

            // Boost score based on term overlap
            let overlap_boost = overlap as f32 / query_terms.len().max(1) as f32;
            result.relevance_score = result.relevance_score * 0.7 + overlap_boost * 0.3;
        }

        // Sort by adjusted scores
        results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Chunk, ChunkMetadata};

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
    fn test_simple_reranker_reorders_by_query_overlap() {
        let mut results = vec![
            make_result("c1", "The cat sat on the mat", 0.9),
            make_result("c2", "Machine learning and neural networks", 0.8),
            make_result("c3", "Deep learning models for machine translation", 0.7),
        ];

        SimpleReranker::rerank("machine learning", &mut results);

        // c2 has both "machine" and "learning" -> full overlap boost
        // c3 has both "machine" and "learning" -> full overlap boost
        // c1 has neither -> no overlap boost
        // c2 had higher original score than c3, so c2 should still be first
        assert_eq!(results[0].chunk.metadata.chunk_id, "c2");
        assert_eq!(results[1].chunk.metadata.chunk_id, "c3");
        assert_eq!(results[2].chunk.metadata.chunk_id, "c1");
    }

    #[test]
    fn test_simple_reranker_empty_results() {
        let mut results: Vec<SearchResult> = Vec::new();
        SimpleReranker::rerank("some query", &mut results);
        assert!(results.is_empty());
    }

    #[test]
    fn test_simple_reranker_single_result() {
        let mut results = vec![make_result("c1", "machine learning basics", 0.5)];
        SimpleReranker::rerank("machine learning", &mut results);
        assert_eq!(results.len(), 1);
        // Score should be 0.5 * 0.7 + 1.0 * 0.3 = 0.65 (full overlap: 2/2 terms match)
        let expected = 0.5 * 0.7 + 1.0 * 0.3;
        assert!((results[0].relevance_score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_simple_reranker_partial_overlap_scores() {
        let mut results = vec![
            make_result("c1", "only machine here", 0.5),
            make_result("c2", "no relevant words", 0.5),
        ];

        SimpleReranker::rerank("machine learning", &mut results);

        // c1 has 1/2 terms matching -> overlap_boost = 0.5
        // c2 has 0/2 terms matching -> overlap_boost = 0.0
        let score_c1 = 0.5 * 0.7 + 0.5 * 0.3; // 0.35 + 0.15 = 0.50
        let score_c2 = 0.5 * 0.7 + 0.0 * 0.3; // 0.35 + 0.00 = 0.35

        assert_eq!(results[0].chunk.metadata.chunk_id, "c1");
        assert!((results[0].relevance_score - score_c1).abs() < 1e-6);
        assert_eq!(results[1].chunk.metadata.chunk_id, "c2");
        assert!((results[1].relevance_score - score_c2).abs() < 1e-6);
    }

    #[test]
    fn test_simple_reranker_case_insensitive() {
        let mut results = vec![make_result("c1", "MACHINE LEARNING models", 0.5)];

        SimpleReranker::rerank("machine learning", &mut results);

        // Both terms should match case-insensitively
        let expected = 0.5 * 0.7 + 1.0 * 0.3;
        assert!((results[0].relevance_score - expected).abs() < 1e-6);
    }
}
