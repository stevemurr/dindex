//! Query coordinator for distributed search

use crate::config::Config;
use crate::embedding::EmbeddingEngine;
use crate::network::NetworkHandle;
use crate::retrieval::HybridRetriever;
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

            // If no candidates from semantic routing, fall back to a random subset
            // of connected peers (capped at max_fanout_peers) instead of all peers
            if peer_ids.is_empty() {
                match network.get_peers().await {
                    Ok(peers) => {
                        let mut fallback_peers: Vec<PeerId> =
                            peers.into_iter().map(|p| p.peer_id).collect();
                        if !fallback_peers.is_empty() {
                            use rand::seq::SliceRandom;
                            fallback_peers.shuffle(&mut rand::thread_rng());
                            fallback_peers
                                .truncate(self.config.retrieval.max_fanout_peers);
                            debug!(
                                "No semantic routing candidates, querying random subset of {} connected peers",
                                fallback_peers.len()
                            );
                            peer_ids = fallback_peers;
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

        // Aggregate results using score-based merge (no double-RRF)
        let final_results = self.aggregate_results(all_results, query.top_k);

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

    /// Aggregate results from multiple nodes using score-based merge.
    ///
    /// Each node's results already have RRF-fused scores from their local hybrid
    /// retriever. Rather than re-applying RRF (which destroys score meaning by
    /// treating rank-1 results identically regardless of score), we take the max
    /// score per chunk_id across all nodes. This preserves the quality signal from
    /// each node's retrieval pipeline and keeps `matched_by` attribution intact.
    fn aggregate_results(
        &self,
        node_results: Vec<(NodeId, Vec<SearchResult>)>,
        top_k: usize,
    ) -> Vec<SearchResult> {
        if node_results.is_empty() {
            return Vec::new();
        }

        // Keep the highest-scored version of each chunk across all nodes.
        // This preserves matched_by and node_id from the best-scoring source.
        let mut best: HashMap<String, SearchResult> = HashMap::new();
        for (node_id, results) in node_results {
            for mut result in results {
                // Tag with source node if not already set
                if result.node_id.is_none() {
                    result.node_id = Some(node_id.clone());
                }
                best.entry(result.chunk.metadata.chunk_id.clone())
                    .and_modify(|existing| {
                        if result.relevance_score > existing.relevance_score {
                            *existing = result.clone();
                        }
                    })
                    .or_insert(result);
            }
        }

        let mut merged: Vec<SearchResult> = best.into_values().collect();
        merged.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
        merged.truncate(top_k);
        merged
    }

    /// Estimate query result quality using response coverage, result scores, confidence, and sufficiency.
    ///
    /// Quality is a weighted combination of:
    /// - Response coverage (30%): fraction of expected peers that responded, weighted by rank
    /// - Average relevance score (30%): mean score of all returned results
    /// - Score confidence (20%): 1 - coefficient of variation; high variance = uncertain results
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
        let scores: Vec<f32> = results
            .iter()
            .flat_map(|(_, r)| r.iter().map(|s| s.relevance_score))
            .collect();
        let total_results = scores.len();
        let avg_score = if total_results > 0 {
            scores.iter().sum::<f32>() / total_results as f32
        } else {
            0.0
        };

        // 3. Score confidence: 1 - coefficient of variation (clamped to [0, 1])
        // High variance among result scores suggests uncertain/mixed quality results.
        let confidence = if total_results > 1 && avg_score > 0.0 {
            let variance = scores
                .iter()
                .map(|s| (s - avg_score).powi(2))
                .sum::<f32>()
                / total_results as f32;
            let std_dev = variance.sqrt();
            let cv = std_dev / avg_score; // coefficient of variation
            (1.0 - cv).clamp(0.0, 1.0)
        } else if total_results == 1 {
            1.0 // single result has perfect confidence
        } else {
            0.0 // no results = no confidence
        };

        // 4. Result sufficiency: did we get enough results?
        let expected_results = plan.candidate_nodes.len().max(1); // at least 1
        let sufficiency = (total_results as f32 / expected_results as f32).min(1.0);

        // Weighted combination
        coverage_score * 0.3 + avg_score * 0.3 + confidence * 0.2 + sufficiency * 0.2
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
        // Score-based merge keeps the highest score (0.9), not a re-fused score
        assert!(
            (aggregated[0].relevance_score - 0.9).abs() < 1e-6,
            "should keep max score, got {}",
            aggregated[0].relevance_score,
        );
    }

    #[test]
    fn test_aggregation_preserves_matched_by() {
        let coordinator = make_coordinator();

        let chunk = Chunk {
            metadata: crate::types::ChunkMetadata::new("c1".to_string(), "d1".to_string()),
            content: "content 1".to_string(),
            token_count: 2,
        };

        let mut result = SearchResult::new(chunk, 0.8);
        result.matched_by = vec![
            crate::retrieval::RetrievalMethod::Dense,
            crate::retrieval::RetrievalMethod::Bm25,
        ];

        let all_results = vec![("node1".to_string(), vec![result])];
        let aggregated = coordinator.aggregate_results(all_results, 10);

        assert_eq!(aggregated.len(), 1);
        // matched_by from the remote node should be preserved, not overwritten
        assert_eq!(aggregated[0].matched_by.len(), 2);
        assert!(aggregated[0].matched_by.contains(&crate::retrieval::RetrievalMethod::Dense));
        assert!(aggregated[0].matched_by.contains(&crate::retrieval::RetrievalMethod::Bm25));
    }

    #[test]
    fn test_aggregation_tags_node_id() {
        let coordinator = make_coordinator();

        let chunk = Chunk {
            metadata: crate::types::ChunkMetadata::new("c1".to_string(), "d1".to_string()),
            content: "content 1".to_string(),
            token_count: 2,
        };

        let result = SearchResult::new(chunk, 0.8);
        let all_results = vec![("remote-node".to_string(), vec![result])];
        let aggregated = coordinator.aggregate_results(all_results, 10);

        assert_eq!(aggregated[0].node_id, Some("remote-node".to_string()));
    }

    #[test]
    fn test_aggregation_sorts_by_score() {
        let coordinator = make_coordinator();

        let chunk_a = Chunk {
            metadata: crate::types::ChunkMetadata::new("ca".to_string(), "d1".to_string()),
            content: "low score".to_string(),
            token_count: 2,
        };
        let chunk_b = Chunk {
            metadata: crate::types::ChunkMetadata::new("cb".to_string(), "d1".to_string()),
            content: "high score".to_string(),
            token_count: 2,
        };

        let all_results = vec![(
            "node1".to_string(),
            vec![
                SearchResult::new(chunk_a, 0.3),
                SearchResult::new(chunk_b, 0.9),
            ],
        )];

        let aggregated = coordinator.aggregate_results(all_results, 10);
        assert_eq!(aggregated.len(), 2);
        assert_eq!(aggregated[0].chunk.metadata.chunk_id, "cb");
        assert_eq!(aggregated[1].chunk.metadata.chunk_id, "ca");
    }

    #[test]
    fn test_quality_confidence_penalizes_high_variance() {
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

        // Consistent high scores -> high confidence
        let consistent = make_test_results(&[0.9, 0.85, 0.88]);
        let q_consistent = coordinator.estimate_quality(&responding, &timed_out, &plan, &consistent);

        // Wildly varying scores -> low confidence
        let varied = make_test_results(&[0.95, 0.1, 0.5]);
        let q_varied = coordinator.estimate_quality(&responding, &timed_out, &plan, &varied);

        assert!(
            q_consistent > q_varied,
            "consistent scores ({}) should yield higher quality than varied scores ({})",
            q_consistent, q_varied
        );
    }
}
