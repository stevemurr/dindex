//! Cross-encoder reranking
//!
//! Uses ms-marco-MiniLM-L-6-v2 for reranking search results

use crate::types::SearchResult;

#[cfg(feature = "onnx")]
use anyhow::{Context, Result};
#[cfg(feature = "onnx")]
use std::path::Path;
#[cfg(feature = "onnx")]
use tracing::{debug, info};

#[cfg(feature = "onnx")]
use {
    ort::{execution_providers::CPUExecutionProvider, session::Session, value::Tensor},
    parking_lot::Mutex,
    tokenizers::Tokenizer,
};

/// Cross-encoder reranker for improving search result ordering
#[cfg(feature = "onnx")]
pub struct Reranker {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
}

#[cfg(feature = "onnx")]
impl Reranker {
    /// Create a new reranker from model files
    pub fn new(model_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        info!("Loading reranker model...");

        let session = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_intra_threads(4)?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load reranker model")?;

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_length: 512,
        })
    }

    /// Rerank search results based on query-document relevance
    pub fn rerank(&self, query: &str, results: &mut Vec<SearchResult>) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }

        debug!("Reranking {} results", results.len());

        // Prepare query-document pairs
        let pairs: Vec<String> = results
            .iter()
            .map(|r| format!("{} [SEP] {}", query, r.chunk.content))
            .collect();

        // Score all pairs
        let scores = self.score_batch(&pairs)?;

        // Update relevance scores
        for (result, score) in results.iter_mut().zip(scores.iter()) {
            result.relevance_score = *score;
        }

        // Sort by new scores
        results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));

        Ok(())
    }

    /// Score a batch of query-document pairs
    fn score_batch(&self, pairs: &[String]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all pairs
        let encodings = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        // Find max length
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);

        let batch_size = pairs.len();
        let mut input_ids: Vec<i64> = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let len = ids.len().min(max_len);

            for i in 0..max_len {
                if i < len {
                    input_ids.push(ids[i] as i64);
                    attention_mask.push(1);
                } else {
                    input_ids.push(0);
                    attention_mask.push(0);
                }
            }
        }

        // Create input tensors using (shape, data) tuple format
        let shape = [batch_size, max_len];

        // Run inference and extract output data (copy to owned)
        let (logits_shape, logits_data): (Vec<usize>, Vec<f32>) = {
            let mut session = self.session.lock();
            let outputs = session.run(ort::inputs![
                "input_ids" => Tensor::from_array((shape, input_ids))?,
                "attention_mask" => Tensor::from_array((shape, attention_mask))?,
            ])?;

            // Extract scores (logits) - convert to owned data
            let (_, v) = outputs
                .iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No output tensor"))?;

            let arr = v.try_extract_array::<f32>()?;
            (arr.shape().to_vec(), arr.iter().copied().collect())
        };

        // Reconstruct as ndarray view from owned data
        let logits_view =
            ndarray::ArrayViewD::from_shape(logits_shape.as_slice(), logits_data.as_slice())?;

        // For cross-encoders, typically take the logit for the positive class
        // or apply sigmoid for relevance score
        let scores: Vec<f32> = (0..batch_size)
            .map(|i| {
                // If model outputs 2 logits (binary classification), take positive class
                // Otherwise take single logit and apply sigmoid
                if logits_view.ndim() > 1 && logits_view.shape()[1] > 1 {
                    logits_view[[i, 1]]
                } else {
                    sigmoid(logits_view[[i, 0]])
                }
            })
            .collect();

        Ok(scores)
    }
}

#[cfg(feature = "onnx")]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Placeholder reranker for when no model is available
pub struct SimpleReranker;

impl SimpleReranker {
    /// Rerank using a simple heuristic (query term overlap)
    pub fn rerank(query: &str, results: &mut Vec<SearchResult>) {
        let query_lower = query.to_lowercase();
        let query_terms: std::collections::HashSet<&str> =
            query_lower.split_whitespace().collect();

        for result in results.iter_mut() {
            let content_lower = result.chunk.content.to_lowercase();
            let overlap: usize = query_terms
                .iter()
                .filter(|term| content_lower.contains(*term))
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
