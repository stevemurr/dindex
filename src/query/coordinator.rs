//! Query coordinator for distributed search

use crate::config::Config;
use crate::embedding::EmbeddingEngine;
use crate::network::NetworkHandle;
use crate::retrieval::{
    reciprocal_rank_fusion, HybridRetriever, RrfConfig, SimpleReranker,
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
        let mut all_results: Vec<(NodeId, Vec<SearchResult>)> =
            Vec::with_capacity(plan.candidate_nodes.len() + 1);
        let mut responding_nodes = Vec::with_capacity(plan.candidate_nodes.len() + 1);
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
            // Get candidate peer IDs from semantic routing
            let mut peer_ids: Vec<PeerId> = plan
                .candidate_nodes
                .iter()
                .filter_map(|c| c.node_id.parse().ok())
                .collect();

            // If no candidates from semantic routing, fall back to all connected peers
            if peer_ids.is_empty() {
                match network.get_peers().await {
                    Ok(peers) => {
                        peer_ids = peers.into_iter().map(|p| p.peer_id).collect();
                        if !peer_ids.is_empty() {
                            debug!(
                                "No semantic routing candidates, querying all {} connected peers",
                                peer_ids.len()
                            );
                        }
                    }
                    Err(e) => {
                        warn!("Failed to get connected peers: {}", e);
                    }
                }
            }

            if !peer_ids.is_empty() {
                match network
                    .query(peer_ids.clone(), query.clone(), Some(embedding.clone()), plan.timeout)
                    .await
                {
                    Ok(responses) => {
                        for response in responses {
                            let node_id = response
                                .responder_peer
                                .clone()
                                .unwrap_or_else(|| response.request_id.clone());
                            responding_nodes.push(node_id.clone());
                            all_results.push((node_id, response.results));
                        }
                    }
                    Err(e) => {
                        warn!("Remote query failed: {}", e);
                        for peer_id in &peer_ids {
                            timed_out_nodes.push(peer_id.to_string());
                        }
                    }
                }

                // Adaptive fan-out: if initial results are low quality, expand to more peers
                let initial_quality = self.estimate_quality(
                    &responding_nodes, &timed_out_nodes, &plan, &all_results,
                );
                let total_results: usize = all_results.iter().map(|(_, r)| r.len()).sum();
                let avg_score = if total_results > 0 {
                    all_results.iter()
                        .flat_map(|(_, r)| r.iter().map(|s| s.relevance_score))
                        .sum::<f32>() / total_results as f32
                } else {
                    0.0
                };

                if initial_quality < self.config.retrieval.fanout_quality_threshold
                    || total_results < self.config.retrieval.fanout_min_results
                    || avg_score < self.config.retrieval.fanout_score_threshold
                {
                    if let Ok(all_peers) = network.get_peers().await {
                        let already_queried: std::collections::HashSet<PeerId> =
                            peer_ids.iter().cloned().collect();
                        let tier2_peers: Vec<PeerId> = all_peers
                            .into_iter()
                            .map(|p| p.peer_id)
                            .filter(|p| !already_queried.contains(p))
                            .take(self.config.retrieval.max_fanout_peers)
                            .collect();

                        if !tier2_peers.is_empty() {
                            let tier2_timeout = plan.timeout.mul_f32(
                                self.config.retrieval.fanout_timeout_fraction,
                            );
                            debug!(
                                "Adaptive fan-out: expanding to {} additional peers (quality={:.2}, results={}, avg_score={:.2})",
                                tier2_peers.len(), initial_quality, total_results, avg_score
                            );
                            match network
                                .query(tier2_peers.clone(), query.clone(), Some(embedding.clone()), tier2_timeout)
                                .await
                            {
                                Ok(responses) => {
                                    for response in responses {
                                        let node_id = response
                                            .responder_peer
                                            .clone()
                                            .unwrap_or_else(|| response.request_id.clone());
                                        responding_nodes.push(node_id.clone());
                                        all_results.push((node_id, response.results));
                                    }
                                }
                                Err(e) => {
                                    warn!("Adaptive fan-out query failed: {}", e);
                                    for peer_id in tier2_peers {
                                        timed_out_nodes.push(peer_id.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Calculate quality estimate (before aggregate_results consumes all_results)
        let quality = self.estimate_quality(&responding_nodes, &timed_out_nodes, &plan, &all_results);

        // Aggregate results using RRF
        let aggregated = self.aggregate_results(all_results, query.top_k);

        // Rerank results
        let mut final_results = aggregated;
        SimpleReranker::rerank(&query.text, &mut final_results);
        final_results.truncate(query.top_k);

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
    fn create_plan(&self, _query: &Query, embedding: &Embedding) -> Result<QueryPlan> {
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
            .map(|(_node_id, results)| {
                results
                    .iter()
                    .enumerate()
                    .map(|(rank, r)| crate::retrieval::RankedResult {
                        chunk_id: r.chunk.metadata.chunk_id.clone(),
                        rank: rank + 1,
                        original_score: r.relevance_score,
                        method: crate::retrieval::RetrievalMethod::Dense,
                    })
                    .collect()
            })
            .collect();

        let rrf_config = RrfConfig {
            k: self.config.retrieval.rrf_k,
        };
        let fused = reciprocal_rank_fusion(&ranked_lists, &rrf_config);

        // Map back to SearchResult, keeping the highest-scored version of each chunk
        let mut chunk_map: HashMap<String, SearchResult> = HashMap::new();
        for (_, results) in node_results {
            for result in results {
                chunk_map
                    .entry(result.chunk.metadata.chunk_id.clone())
                    .and_modify(|existing| {
                        if result.relevance_score > existing.relevance_score {
                            *existing = result.clone();
                        }
                    })
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
                    result.matched_by = f.contributing_methods.clone();
                    result
                })
            })
            .collect()
    }

    /// Estimate query result quality using response coverage, result scores, and sufficiency.
    ///
    /// Quality is a weighted combination of:
    /// - Response coverage (40%): fraction of expected peers that responded, weighted by rank
    /// - Average relevance score (40%): mean score of all returned results
    /// - Result sufficiency (20%): total_results / expected (capped at 1.0)
    fn estimate_quality(
        &self,
        responding: &[NodeId],
        _timed_out: &[NodeId],
        plan: &QueryPlan,
        results: &[(NodeId, Vec<SearchResult>)],
    ) -> f32 {
        let total_expected = plan.candidate_nodes.len() + if plan.query_local { 1 } else { 0 };
        if total_expected == 0 {
            return 1.0;
        }

        // 1. Response coverage (weighted by candidate rank)
        let responded = responding.len();
        let response_ratio = responded as f32 / total_expected as f32;

        let mut weighted_coverage = 0.0;
        let mut total_weight = 0.0;

        for (i, candidate) in plan.candidate_nodes.iter().enumerate() {
            let weight = 1.0 / (i as f32 + 1.0); // Higher weight for top candidates
            total_weight += weight;

            if responding.contains(&candidate.node_id) {
                weighted_coverage += weight;
            }
        }

        let coverage_score = if total_weight > 0.0 {
            (response_ratio + weighted_coverage / total_weight) / 2.0
        } else {
            response_ratio
        };

        // 2. Average relevance score of returned results
        let total_results: usize = results.iter().map(|(_, r)| r.len()).sum();
        let avg_score = if total_results > 0 {
            results.iter()
                .flat_map(|(_, r)| r.iter().map(|s| s.relevance_score))
                .sum::<f32>() / total_results as f32
        } else {
            0.0
        };

        // 3. Result sufficiency: did we get enough results?
        let expected_results = plan.candidate_nodes.len().max(1); // at least 1
        let sufficiency = (total_results as f32 / expected_results as f32).min(1.0);

        // Weighted combination
        coverage_score * 0.4 + avg_score * 0.4 + sufficiency * 0.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Chunk;

    fn make_coordinator() -> QueryCoordinator {
        let config = Config::default();
        let router = Arc::new(QueryRouter::new(768, &config.routing));
        QueryCoordinator::new(None, None, router, None, config)
    }

    #[test]
    fn test_result_aggregation() {
        let coordinator = make_coordinator();

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

    fn make_test_results(scores: &[f32]) -> Vec<(NodeId, Vec<SearchResult>)> {
        let results: Vec<SearchResult> = scores.iter().map(|&score| {
            let chunk = Chunk {
                metadata: crate::types::ChunkMetadata::new(
                    format!("c_{}", score),
                    "d1".to_string(),
                ),
                content: "test".to_string(),
                token_count: 1,
            };
            SearchResult::new(chunk, score)
        }).collect();
        vec![("node1".to_string(), results)]
    }

    #[test]
    fn test_quality_estimate_full_response() {
        let coordinator = make_coordinator();
        let plan = QueryPlan {
            embedding: vec![0.0; 768],
            candidate_nodes: vec![
                CandidateNode { node_id: "node1".to_string(), similarity: 0.9, matching_centroids: vec![0] },
                CandidateNode { node_id: "node2".to_string(), similarity: 0.8, matching_centroids: vec![1] },
            ],
            query_local: false,
            timeout: Duration::from_secs(10),
        };

        let responding = vec!["node1".to_string(), "node2".to_string()];
        let timed_out: Vec<NodeId> = vec![];
        let results = make_test_results(&[0.9, 0.8]);
        let quality = coordinator.estimate_quality(&responding, &timed_out, &plan, &results);
        assert!(quality >= 0.7, "full response with good scores should have high quality, got {}", quality);
    }

    #[test]
    fn test_quality_estimate_no_responses() {
        let coordinator = make_coordinator();
        let plan = QueryPlan {
            embedding: vec![0.0; 768],
            candidate_nodes: vec![
                CandidateNode { node_id: "node1".to_string(), similarity: 0.9, matching_centroids: vec![0] },
                CandidateNode { node_id: "node2".to_string(), similarity: 0.8, matching_centroids: vec![1] },
            ],
            query_local: false,
            timeout: Duration::from_secs(10),
        };

        let responding: Vec<NodeId> = vec![];
        let timed_out = vec!["node1".to_string(), "node2".to_string()];
        let no_results: Vec<(NodeId, Vec<SearchResult>)> = vec![];
        let quality = coordinator.estimate_quality(&responding, &timed_out, &plan, &no_results);
        assert!(quality < coordinator.config.retrieval.fanout_quality_threshold,
            "no responses should trigger fan-out, got quality {}", quality);
    }

    #[test]
    fn test_quality_estimate_partial_response() {
        let coordinator = make_coordinator();
        let plan = QueryPlan {
            embedding: vec![0.0; 768],
            candidate_nodes: vec![
                CandidateNode { node_id: "node1".to_string(), similarity: 0.9, matching_centroids: vec![0] },
                CandidateNode { node_id: "node2".to_string(), similarity: 0.8, matching_centroids: vec![1] },
                CandidateNode { node_id: "node3".to_string(), similarity: 0.7, matching_centroids: vec![2] },
            ],
            query_local: false,
            timeout: Duration::from_secs(10),
        };

        // Only top candidate responded
        let responding = vec!["node1".to_string()];
        let timed_out = vec!["node2".to_string(), "node3".to_string()];
        let results = make_test_results(&[0.5]);
        let quality = coordinator.estimate_quality(&responding, &timed_out, &plan, &results);
        assert!(quality > 0.0 && quality < 1.0,
            "partial response should have intermediate quality, got {}", quality);
    }

    #[test]
    fn test_quality_estimate_high_scores_boost_quality() {
        let coordinator = make_coordinator();
        let plan = QueryPlan {
            embedding: vec![0.0; 768],
            candidate_nodes: vec![
                CandidateNode { node_id: "node1".to_string(), similarity: 0.9, matching_centroids: vec![0] },
            ],
            query_local: false,
            timeout: Duration::from_secs(10),
        };

        let responding = vec!["node1".to_string()];
        let timed_out: Vec<NodeId> = vec![];

        let high_results = make_test_results(&[0.95, 0.90, 0.85]);
        let low_results = make_test_results(&[0.1, 0.05]);

        let high_quality = coordinator.estimate_quality(&responding, &timed_out, &plan, &high_results);
        let low_quality = coordinator.estimate_quality(&responding, &timed_out, &plan, &low_results);

        assert!(high_quality > low_quality,
            "high scoring results should produce higher quality: {} vs {}", high_quality, low_quality);
    }

    #[test]
    fn test_aggregation_keeps_highest_score() {
        let coordinator = make_coordinator();

        let chunk = Chunk {
            metadata: crate::types::ChunkMetadata::new("c1".to_string(), "d1".to_string()),
            content: "content 1".to_string(),
            token_count: 2,
        };

        // Same chunk from two nodes with different scores
        let node1_results = vec![SearchResult::new(chunk.clone(), 0.5)];
        let node2_results = vec![SearchResult::new(chunk.clone(), 0.9)];

        let all_results = vec![
            ("node1".to_string(), node1_results),
            ("node2".to_string(), node2_results),
        ];

        let aggregated = coordinator.aggregate_results(all_results, 10);
        assert_eq!(aggregated.len(), 1);
    }
}
