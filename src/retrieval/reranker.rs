//! Cross-encoder reranking
//!
//! Uses ms-marco-MiniLM-L-6-v2 for reranking search results

use crate::types::SearchResult;
use anyhow::{Context, Result};
use std::path::Path;
use tracing::{debug, info};

#[cfg(feature = "onnx")]
use {
    ndarray::Array2,
    ort::{execution_providers::CPUExecutionProvider, session::Session},
    tokenizers::Tokenizer,
};

/// Cross-encoder reranker for improving search result ordering
#[cfg(feature = "onnx")]
pub struct Reranker {
    session: Session,
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
            session,
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
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

        // Create input tensors
        let input_ids_array = Array2::from_shape_vec((batch_size, max_len), input_ids)?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, max_len), attention_mask)?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_array,
            "attention_mask" => attention_mask_array,
        ]?)?;

        // Extract scores (logits)
        let output_tensor = outputs
            .iter()
            .next()
            .map(|(_, v)| v)
            .ok_or_else(|| anyhow::anyhow!("No output tensor"))?;

        let logits = output_tensor.try_extract_tensor::<f32>()?;
        let logits_view = logits.view();

        // For cross-encoders, typically take the logit for the positive class
        // or apply sigmoid for relevance score
        let scores: Vec<f32> = (0..batch_size)
            .map(|i| {
                // If model outputs 2 logits (binary classification), take positive class
                // Otherwise take single logit and apply sigmoid
                if logits_view.shape().len() > 1 && logits_view.shape()[1] > 1 {
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
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}
