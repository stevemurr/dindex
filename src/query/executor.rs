//! Local query executor

use crate::embedding::EmbeddingEngine;
use crate::network::QueryRequest;
use crate::retrieval::HybridRetriever;
use crate::types::{Query, SearchResult};
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tracing::debug;

/// Local query executor for handling incoming queries
pub struct QueryExecutor {
    /// Hybrid retriever for local search
    retriever: Arc<HybridRetriever>,
    /// Embedding engine
    embedding_engine: Option<Arc<EmbeddingEngine>>,
}

/// Query execution result
#[derive(Debug)]
pub struct ExecutionResult {
    pub results: Vec<SearchResult>,
    pub processing_time_ms: u64,
}

impl QueryExecutor {
    /// Create a new query executor
    pub fn new(
        retriever: Arc<HybridRetriever>,
        embedding_engine: Option<Arc<EmbeddingEngine>>,
    ) -> Self {
        Self {
            retriever,
            embedding_engine,
        }
    }

    /// Execute a query locally
    pub fn execute(&self, query: &Query) -> Result<ExecutionResult> {
        let start = Instant::now();

        // Get or compute embedding
        let embedding = if let Some(engine) = &self.embedding_engine {
            Some(engine.embed(&query.text)?)
        } else {
            None
        };

        // Execute search
        let results = self.retriever.search(query, embedding.as_ref())?;

        let processing_time = start.elapsed().as_millis() as u64;

        debug!(
            "Executed query '{}': {} results in {}ms",
            truncate(&query.text, 50),
            results.len(),
            processing_time
        );

        Ok(ExecutionResult {
            results,
            processing_time_ms: processing_time,
        })
    }

    /// Execute a network query request
    pub fn execute_request(&self, request: &QueryRequest) -> Result<ExecutionResult> {
        let start = Instant::now();

        // Use provided embedding if available
        let results = if let Some(embedding) = &request.query_embedding {
            self.retriever.search(&request.query, Some(embedding))?
        } else {
            self.execute(&request.query)?.results
        };

        let processing_time = start.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            results,
            processing_time_ms: processing_time,
        })
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.to_string()
    }
}

/// Batch query executor for parallel execution
pub struct BatchExecutor {
    executor: QueryExecutor,
}

impl BatchExecutor {
    pub fn new(executor: QueryExecutor) -> Self {
        Self { executor }
    }

    /// Execute multiple queries
    pub fn execute_batch(&self, queries: &[Query]) -> Vec<Result<ExecutionResult>> {
        queries.iter().map(|q| self.executor.execute(q)).collect()
    }
}
