//! Query coordinator for distributed search

use crate::config::Config;
use crate::embedding::EmbeddingEngine;
use crate::network::{NetworkHandle, QueryRequest, QueryResponse};
use crate::retrieval::{
    reciprocal_rank_fusion, to_ranked_results, FusedResult, HybridRetriever, RrfConfig, SimpleReranker,
};
use crate::routing::{CandidateNode, QueryRouter};
use crate::types::{Embedding, NodeId, Query, SearchResult};
use anyhow::Result;
use libp2p::PeerId;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Query coordinator for distributed search
pub struct QueryCoordinator {
    /// Local hybrid retriever
    local_retriever: Option<Arc<HybridRetriever>>,
    /// Embedding engine
    embedding_engine: Option<Arc<EmbeddingEngine>>,
    /// Query router for finding candidate nodes
    router: Arc<QueryRouter>,
    /// Network handle for P2P queries
    network: Option<NetworkHandle>,
    /// Configuration
    config: Config,
}

/// Query execution plan
#[derive(Debug)]
pub struct QueryPlan {
    /// Query embedding
    pub embedding: Embedding,
    /// Candidate nodes to query
    pub candidate_nodes: Vec<CandidateNode>,
    /// Whether to query local index
    pub query_local: bool,
    /// Timeout for remote queries
    pub timeout: Duration,
}

/// Aggregated query results
#[derive(Debug)]
pub struct AggregatedResults {
    /// Final search results
    pub results: Vec<SearchResult>,
    /// Nodes that responded
    pub responding_nodes: Vec<NodeId>,
    /// Nodes that timed out
    pub timed_out_nodes: Vec<NodeId>,
    /// Total query time
    pub total_time_ms: u64,
    /// Quality estimation (0-1)
    pub quality_estimate: f32,
}

impl QueryCoordinator {
    /// Create a new query coordinator
    pub fn new(
        local_retriever: Option<Arc<HybridRetriever>>,
        embedding_engine: Option<Arc<EmbeddingEngine>>,
        router: Arc<QueryRouter>,
        network: Option<NetworkHandle>,
        config: Config,
    ) -> Self {
        Self {
            local_retriever,
            embedding_engine,
            router,
            network,
            config,
        }
    }

    /// Execute a query across the distributed network
    pub async fn execute(&self, query: &Query) -> Result<AggregatedResults> {
        let start = Instant::now();

        // Generate query embedding
        let embedding = self.embed_query(&query.text)?;

        // Create query plan
        let plan = self.create_plan(query, &embedding)?;

        debug!(
            "Query plan: {} candidate nodes, local={}",
            plan.candidate_nodes.len(),
            plan.query_local
        );

        // Execute queries
        let mut all_results: Vec<(NodeId, Vec<SearchResult>)> = Vec::new();
        let mut responding_nodes = Vec::new();
        let mut timed_out_nodes = Vec::new();

        // Query local index
        if plan.query_local {
            if let Some(retriever) = &self.local_retriever {
                match retriever.search(query, Some(&embedding)) {
                    Ok(results) => {
                        let local_id = "local".to_string();
                        responding_nodes.push(local_id.clone());
                        all_results.push((local_id, results));
                    }
                    Err(e) => {
                        warn!("Local search failed: {}", e);
                    }
                }
            }
        }

        // Query remote nodes (if network available)
        if let Some(network) = &self.network {
            let peer_ids: Vec<PeerId> = plan
                .candidate_nodes
                .iter()
                .filter_map(|c| c.node_id.parse().ok())
                .collect();

            if !peer_ids.is_empty() {
                match network
                    .query(peer_ids.clone(), query.clone(), Some(embedding.clone()), plan.timeout)
                    .await
                {
                    Ok(responses) => {
                        for response in responses {
                            responding_nodes.push(response.request_id.clone());
                            all_results.push((response.request_id, response.results));
                        }
                    }
                    Err(e) => {
                        warn!("Remote query failed: {}", e);
                        for peer_id in peer_ids {
                            timed_out_nodes.push(peer_id.to_string());
                        }
                    }
                }
            }
        }

        // Aggregate results using RRF
        let aggregated = self.aggregate_results(all_results, query.top_k);

        // Rerank results
        let mut final_results = aggregated;
        SimpleReranker::rerank(&query.text, &mut final_results);
        final_results.truncate(query.top_k);

        // Calculate quality estimate
        let quality = self.estimate_quality(&responding_nodes, &timed_out_nodes, &plan);

        let total_time = start.elapsed().as_millis() as u64;

        info!(
            "Query completed in {}ms: {} results from {} nodes",
            total_time,
            final_results.len(),
            responding_nodes.len()
        );

        Ok(AggregatedResults {
            results: final_results,
            responding_nodes,
            timed_out_nodes,
            total_time_ms: total_time,
            quality_estimate: quality,
        })
    }

    /// Embed query text
    fn embed_query(&self, text: &str) -> Result<Embedding> {
        if let Some(engine) = &self.embedding_engine {
            engine.embed(text)
        } else {
            // Return dummy embedding for testing
            Ok(vec![0.0; self.config.embedding.dimensions])
        }
    }

    /// Create a query execution plan
    fn create_plan(&self, query: &Query, embedding: &Embedding) -> Result<QueryPlan> {
        // Find candidate nodes via semantic routing
        let lsh_signature = self.router.hash_query(embedding);
        let candidate_nodes = self.router.find_candidates(embedding, Some(&lsh_signature));

        // Determine if we should query local
        let query_local = self.local_retriever.is_some();

        Ok(QueryPlan {
            embedding: embedding.clone(),
            candidate_nodes,
            query_local,
            timeout: Duration::from_secs(self.config.node.query_timeout_secs),
        })
    }

    /// Aggregate results from multiple nodes using RRF
    fn aggregate_results(
        &self,
        node_results: Vec<(NodeId, Vec<SearchResult>)>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        if node_results.is_empty() {
            return Vec::new();
        }

        // Convert to ranked lists for RRF
        let ranked_lists: Vec<Vec<crate::retrieval::RankedResult>> = node_results
            .iter()
            .map(|(node_id, results)| {
                results
                    .iter()
                    .enumerate()
                    .map(|(rank, r)| crate::retrieval::RankedResult {
                        chunk_id: r.chunk.metadata.chunk_id.clone(),
                        rank: rank + 1,
                        original_score: r.relevance_score,
                        method: node_id.clone(),
                    })
                    .collect()
            })
            .collect();

        let rrf_config = RrfConfig {
            k: self.config.retrieval.rrf_k,
        };
        let fused = reciprocal_rank_fusion(&ranked_lists, &rrf_config);

        // Map back to SearchResult
        let mut chunk_map: HashMap<String, SearchResult> = HashMap::new();
        for (_, results) in node_results {
            for result in results {
                chunk_map
                    .entry(result.chunk.metadata.chunk_id.clone())
                    .or_insert(result);
            }
        }

        fused
            .into_iter()
            .take(top_k)
            .filter_map(|f| {
                chunk_map.get(&f.chunk_id).map(|r| {
                    let mut result = r.clone();
                    result.relevance_score = f.rrf_score;
                    result.matched_by = f.contributing_methods;
                    result
                })
            })
            .collect()
    }

    /// Estimate query result quality
    fn estimate_quality(
        &self,
        responding: &[NodeId],
        timed_out: &[NodeId],
        plan: &QueryPlan,
    ) -> f32 {
        let total_expected = plan.candidate_nodes.len() + if plan.query_local { 1 } else { 0 };
        if total_expected == 0 {
            return 1.0;
        }

        let responded = responding.len();
        let response_ratio = responded as f32 / total_expected as f32;

        // Weight by node relevance (top nodes matter more)
        let mut weighted_coverage = 0.0;
        let mut total_weight = 0.0;

        for (i, candidate) in plan.candidate_nodes.iter().enumerate() {
            let weight = 1.0 / (i as f32 + 1.0); // Higher weight for top candidates
            total_weight += weight;

            if responding.contains(&candidate.node_id) {
                weighted_coverage += weight;
            }
        }

        if total_weight > 0.0 {
            (response_ratio + weighted_coverage / total_weight) / 2.0
        } else {
            response_ratio
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Chunk;

    #[test]
    fn test_result_aggregation() {
        let config = Config::default();
        let router = Arc::new(QueryRouter::new(768, &config.routing));
        let coordinator = QueryCoordinator::new(None, None, router, None, config);

        // Create test results from multiple nodes
        let chunk1 = Chunk {
            metadata: crate::types::ChunkMetadata::new("c1".to_string(), "d1".to_string()),
            content: "content 1".to_string(),
            token_count: 2,
        };
        let chunk2 = Chunk {
            metadata: crate::types::ChunkMetadata::new("c2".to_string(), "d1".to_string()),
            content: "content 2".to_string(),
            token_count: 2,
        };

        let node1_results = vec![
            SearchResult::new(chunk1.clone(), 0.9),
            SearchResult::new(chunk2.clone(), 0.7),
        ];
        let node2_results = vec![
            SearchResult::new(chunk2.clone(), 0.85),
            SearchResult::new(chunk1.clone(), 0.65),
        ];

        let all_results = vec![
            ("node1".to_string(), node1_results),
            ("node2".to_string(), node2_results),
        ];

        let aggregated = coordinator.aggregate_results(all_results, 10);

        // Both chunks should be present
        assert_eq!(aggregated.len(), 2);
    }
}
