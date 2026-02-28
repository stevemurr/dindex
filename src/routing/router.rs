//! Semantic query router
//!
//! Routes queries to relevant nodes based on content centroids and LSH

use super::{BandedBloomFilter, CentroidGenerator, LshIndex};
use crate::config::RoutingConfig;
use crate::embedding::cosine_similarity;
use crate::types::{Embedding, LshSignature, NodeAdvertisement, NodeCentroid, NodeId};

use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info};

/// Query router for finding relevant nodes
pub struct QueryRouter {
    /// Known node advertisements
    node_ads: RwLock<HashMap<NodeId, NodeAdvertisement>>,
    /// LSH index for fast filtering
    lsh_index: LshIndex,
    /// Configuration
    config: RoutingConfig,
}

/// Candidate node for a query
#[derive(Debug, Clone)]
pub struct CandidateNode {
    pub node_id: NodeId,
    pub similarity: f32,
    pub matching_centroids: Vec<u32>,
}

impl QueryRouter {
    /// Create a new query router
    pub fn new(dimensions: usize, config: &RoutingConfig) -> Self {
        Self {
            node_ads: RwLock::new(HashMap::new()),
            lsh_index: LshIndex::new(dimensions, config.lsh_bits, 42),
            config: config.clone(),
        }
    }

    /// Register a node advertisement
    pub fn register_node(&self, advertisement: NodeAdvertisement) {
        info!(
            "Registering node {} with {} centroids",
            advertisement.node_id,
            advertisement.centroids.len()
        );
        self.node_ads
            .write()
            .insert(advertisement.node_id.clone(), advertisement);
    }

    /// Find candidate nodes for a query embedding
    pub fn find_candidates(
        &self,
        query_embedding: &Embedding,
        query_lsh: Option<&LshSignature>,
    ) -> Vec<CandidateNode> {
        let node_ads = self.node_ads.read();

        if node_ads.is_empty() {
            return Vec::new();
        }

        let mut candidates: Vec<CandidateNode> = Vec::with_capacity(node_ads.len());

        // Truncate query embedding to match centroid dimensions if needed
        let query_dims = query_embedding.len();

        for (node_id, advertisement) in node_ads.iter() {
            // LSH banded bloom filter pre-check
            // Uses banding technique: if ANY band of query LSH matches a band in the filter,
            // the node might have relevant content. This correctly handles semantic similarity
            // where similar vectors have similar (not identical) LSH signatures.
            if let (Some(lsh), Some(bloom_bytes)) = (query_lsh, &advertisement.lsh_bloom_filter) {
                if let Some(bloom) = BandedBloomFilter::from_bytes(bloom_bytes) {
                    if !bloom.might_contain_similar(&lsh.bits, lsh.num_bits) {
                        // No bands match - skip this node (definitely no similar content)
                        debug!("Skipping node {} - no LSH band matches", node_id);
                        continue;
                    }
                }
            }

            // Compare with centroids
            let mut max_similarity = 0.0f32;
            let mut matching_centroids = Vec::new();

            for centroid in &advertisement.centroids {
                // Handle dimension mismatch
                let similarity = if centroid.embedding.len() == query_dims {
                    cosine_similarity(query_embedding, &centroid.embedding)
                } else if centroid.embedding.len() < query_dims {
                    // Truncate query to match centroid
                    let truncated: Vec<f32> = query_embedding
                        .iter()
                        .take(centroid.embedding.len())
                        .copied()
                        .collect();
                    cosine_similarity(&truncated, &centroid.embedding)
                } else {
                    // Truncate centroid to match query
                    let truncated: Vec<f32> = centroid
                        .embedding
                        .iter()
                        .take(query_dims)
                        .copied()
                        .collect();
                    cosine_similarity(query_embedding, &truncated)
                };

                if similarity > max_similarity {
                    max_similarity = similarity;
                }

                // Track centroids above threshold
                if similarity > self.config.centroid_similarity_threshold {
                    matching_centroids.push(centroid.centroid_id);
                }
            }

            candidates.push(CandidateNode {
                node_id: node_id.clone(),
                similarity: max_similarity,
                matching_centroids,
            });
        }

        // Sort by similarity
        candidates.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));

        // Return top candidates
        let top_k = self.config.candidate_nodes.min(candidates.len());
        candidates.truncate(top_k);

        debug!(
            "Found {} candidate nodes for query",
            candidates.len()
        );

        candidates
    }

    /// Generate LSH signature for a query
    pub fn hash_query(&self, embedding: &Embedding) -> LshSignature {
        self.lsh_index.hash(embedding)
    }

    /// Save node advertisements to disk for persistence across restarts
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let ads = self.node_ads.read();
        let data = serde_json::to_vec(&*ads)?;
        std::fs::write(path, data)?;
        info!("Saved {} node advertisements to {}", ads.len(), path.display());
        Ok(())
    }

    /// Load node advertisements from disk
    pub fn load(&self, path: &Path) -> anyhow::Result<usize> {
        if !path.exists() {
            return Ok(0);
        }
        let data = std::fs::read(path)?;
        let loaded: HashMap<NodeId, NodeAdvertisement> = serde_json::from_slice(&data)?;
        let count = loaded.len();
        *self.node_ads.write() = loaded;
        info!("Loaded {} node advertisements from {}", count, path.display());
        Ok(count)
    }

    /// Get the number of registered nodes
    pub fn node_count(&self) -> usize {
        self.node_ads.read().len()
    }
}

/// Advertisement builder for creating node advertisements
pub struct AdvertisementBuilder {
    node_id: NodeId,
    centroids: Vec<NodeCentroid>,
    lsh_signatures: Vec<LshSignature>,
    total_chunks: usize,
}

impl AdvertisementBuilder {
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            centroids: Vec::new(),
            lsh_signatures: Vec::new(),
            total_chunks: 0,
        }
    }

    /// Generate centroids from embeddings
    pub fn with_centroids(
        mut self,
        embeddings: &[Embedding],
        num_centroids: usize,
        truncate_dims: Option<usize>,
    ) -> Self {
        let generator = CentroidGenerator::new(num_centroids);
        let mut centroids = generator.generate(embeddings);

        // Truncate if requested
        if let Some(dims) = truncate_dims {
            centroids = super::truncate_centroids(&centroids, dims);
        }

        self.centroids = centroids;
        self.total_chunks = embeddings.len();
        self
    }

    /// Add LSH signatures for bloom filter
    pub fn with_lsh(mut self, signatures: Vec<LshSignature>) -> Self {
        self.lsh_signatures = signatures;
        self
    }

    /// Build the advertisement
    pub fn build(self, lsh_bits: usize, num_bands: usize, fp_rate: f64) -> NodeAdvertisement {
        // Create banded bloom filter from LSH signatures
        // Uses LSH banding technique for proper semantic filtering
        let bloom_filter = if !self.lsh_signatures.is_empty() {
            let mut bloom = BandedBloomFilter::new(
                self.lsh_signatures.len(),
                lsh_bits,
                num_bands,
                fp_rate,
            );
            for sig in &self.lsh_signatures {
                bloom.insert(&sig.bits, sig.num_bits);
            }
            Some(bloom.to_bytes())
        } else {
            None
        };

        NodeAdvertisement {
            node_id: self.node_id,
            centroids: self.centroids,
            lsh_bloom_filter: bloom_filter,
            total_chunks: self.total_chunks,
            last_updated: chrono::Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_routing_config() -> RoutingConfig {
        RoutingConfig {
            num_centroids: 10,
            lsh_bits: 64,
            lsh_num_hashes: 4,
            bloom_bits_per_item: 10,
            candidate_nodes: 3,
            lsh_num_bands: 8,
            centroid_similarity_threshold: 0.5,
            bloom_false_positive_rate: 0.01,
        }
    }

    #[test]
    fn test_query_router() {
        let config = test_routing_config();
        let router = QueryRouter::new(64, &config);

        // Create a test advertisement
        let embeddings: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                (0..64)
                    .map(|j| ((i * 64 + j) as f32 / 1000.0).sin())
                    .collect()
            })
            .collect();

        let ad = AdvertisementBuilder::new("node1".to_string())
            .with_centroids(&embeddings, 5, None)
            .build(config.lsh_bits, config.lsh_num_bands, config.bloom_false_positive_rate);

        router.register_node(ad);

        // Query
        let query: Vec<f32> = (0..64).map(|i| (i as f32 / 1000.0).sin()).collect();
        let candidates = router.find_candidates(&query, None);

        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].node_id, "node1");
    }

    #[test]
    fn test_find_candidates_no_registered_nodes() {
        let config = test_routing_config();
        let router = QueryRouter::new(64, &config);

        // Query with no nodes registered
        let query: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0).sin()).collect();
        let candidates = router.find_candidates(&query, None);

        assert!(
            candidates.is_empty(),
            "Should return empty when no nodes are registered"
        );
    }

    #[test]
    fn test_find_candidates_with_bloom_filter_prefilter() {
        let config = RoutingConfig {
            num_centroids: 10,
            lsh_bits: 128,
            lsh_num_hashes: 4,
            bloom_bits_per_item: 10,
            candidate_nodes: 5,
            ..Default::default()
        };

        let router = QueryRouter::new(64, &config);

        // Create embeddings and their LSH signatures for a node
        let embeddings: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                (0..64)
                    .map(|j| ((i * 64 + j) as f32 / 1000.0).sin())
                    .collect()
            })
            .collect();

        // Generate LSH signatures for each embedding
        let lsh_sigs: Vec<LshSignature> = embeddings
            .iter()
            .map(|emb| router.hash_query(emb))
            .collect();

        // Build advertisement with LSH bloom filter
        let ad = AdvertisementBuilder::new("node_with_bloom".to_string())
            .with_centroids(&embeddings, 5, None)
            .with_lsh(lsh_sigs)
            .build(config.lsh_bits, config.lsh_num_bands, config.bloom_false_positive_rate);

        router.register_node(ad);

        // Query with a similar embedding (should match via bloom filter)
        let query: Vec<f32> = (0..64).map(|i| (i as f32 / 1000.0).sin()).collect();
        let query_lsh = router.hash_query(&query);
        let candidates = router.find_candidates(&query, Some(&query_lsh));

        assert!(
            !candidates.is_empty(),
            "Should find the node when bloom filter matches"
        );
        assert_eq!(candidates[0].node_id, "node_with_bloom");

        // Query with a very different embedding (may or may not be filtered)
        // We just verify it doesn't panic
        let different_query: Vec<f32> = (0..64)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let different_lsh = router.hash_query(&different_query);
        let _candidates = router.find_candidates(&different_query, Some(&different_lsh));
        // Result may or may not be empty depending on false positive rate
    }

    #[test]
    fn test_save_load_roundtrip() {
        let config = test_routing_config();
        let router = QueryRouter::new(64, &config);

        // Register some nodes
        let embeddings: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..64).map(|j| ((i * 64 + j) as f32 / 1000.0).sin()).collect())
            .collect();
        let ad1 = AdvertisementBuilder::new("node_a".to_string())
            .with_centroids(&embeddings, 3, None)
            .build(config.lsh_bits, config.lsh_num_bands, config.bloom_false_positive_rate);
        let ad2 = AdvertisementBuilder::new("node_b".to_string())
            .with_centroids(&embeddings, 3, None)
            .build(config.lsh_bits, config.lsh_num_bands, config.bloom_false_positive_rate);
        router.register_node(ad1);
        router.register_node(ad2);
        assert_eq!(router.node_count(), 2);

        // Save
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ads.json");
        router.save(&path).unwrap();

        // Load into a fresh router
        let router2 = QueryRouter::new(64, &config);
        let loaded = router2.load(&path).unwrap();
        assert_eq!(loaded, 2);
        assert_eq!(router2.node_count(), 2);

        // Verify candidates still work
        let query: Vec<f32> = (0..64).map(|i| (i as f32 / 1000.0).sin()).collect();
        let candidates = router2.find_candidates(&query, None);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_load_nonexistent_returns_zero() {
        let config = test_routing_config();
        let router = QueryRouter::new(64, &config);
        let count = router.load(std::path::Path::new("/tmp/does_not_exist_dindex_test.json")).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_register_node_overwrites() {
        let config = test_routing_config();
        let router = QueryRouter::new(64, &config);

        let embeddings1: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..64).map(|j| ((i * 64 + j) as f32 / 500.0).sin()).collect())
            .collect();
        let embeddings2: Vec<Vec<f32>> = (0..15)
            .map(|i| (0..64).map(|j| ((i * 64 + j) as f32 / 1000.0).cos()).collect())
            .collect();

        let ad1 = AdvertisementBuilder::new("node_x".to_string())
            .with_centroids(&embeddings1, 2, None)
            .build(config.lsh_bits, config.lsh_num_bands, config.bloom_false_positive_rate);
        router.register_node(ad1);
        assert_eq!(router.node_count(), 1);

        // Re-register with different data — should overwrite, not duplicate
        let ad2 = AdvertisementBuilder::new("node_x".to_string())
            .with_centroids(&embeddings2, 3, None)
            .build(config.lsh_bits, config.lsh_num_bands, config.bloom_false_positive_rate);
        router.register_node(ad2);
        assert_eq!(router.node_count(), 1);
    }

    #[test]
    fn test_dimension_mismatch_handling() {
        let config = test_routing_config();
        let router = QueryRouter::new(64, &config);

        // Register a node with 32-dim centroids
        let small_embeddings: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                (0..32)
                    .map(|j| ((i * 32 + j) as f32 / 500.0).sin())
                    .collect()
            })
            .collect();

        let ad = AdvertisementBuilder::new("small_node".to_string())
            .with_centroids(&small_embeddings, 3, None)
            .build(config.lsh_bits, config.lsh_num_bands, config.bloom_false_positive_rate);

        router.register_node(ad);

        // Query with 64-dim embedding (larger than centroid dims)
        let query_64: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0).sin()).collect();
        let candidates = router.find_candidates(&query_64, None);

        // Should still return results (truncation handles the mismatch)
        assert!(
            !candidates.is_empty(),
            "Should handle query dimension > centroid dimension"
        );

        // Register a node with 128-dim centroids
        let large_embeddings: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                (0..128)
                    .map(|j| ((i * 128 + j) as f32 / 1000.0).sin())
                    .collect()
            })
            .collect();

        let ad2 = AdvertisementBuilder::new("large_node".to_string())
            .with_centroids(&large_embeddings, 3, None)
            .build(config.lsh_bits, config.lsh_num_bands, config.bloom_false_positive_rate);

        router.register_node(ad2);

        // Query with 64-dim embedding (smaller than centroid dims)
        let candidates2 = router.find_candidates(&query_64, None);

        // Should still return results (centroid truncation handles the mismatch)
        assert!(
            candidates2.len() >= 2,
            "Should find both nodes despite different dimensions"
        );
    }
}
