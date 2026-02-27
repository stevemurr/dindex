//! Network protocol messages

use crate::types::{NodeAdvertisement, Query, SearchResult};
use serde::{Deserialize, Serialize};

/// Protocol version
pub const PROTOCOL_VERSION: &str = "/dindex/1.0.0";

/// Query protocol
pub const QUERY_PROTOCOL: &str = "/dindex/query/1.0.0";

/// Advertisement protocol
pub const ADVERTISEMENT_PROTOCOL: &str = "/dindex/advertisement/1.0.0";

/// Network message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMessage {
    /// Query request to a node
    QueryRequest(QueryRequest),
    /// Query response from a node
    QueryResponse(QueryResponse),
    /// Node advertisement (centroids, bloom filter)
    Advertisement(NodeAdvertisement),
}

/// Query request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Unique request ID
    pub request_id: String,
    /// The query
    pub query: Query,
    /// Pre-computed query embedding (optional, to save compute on target)
    pub query_embedding: Option<Vec<f32>>,
    /// Originating peer (for routing response back)
    pub origin_peer: Option<String>,
    /// Target peers for this query. If non-empty, only these peers should execute the query.
    /// GossipSub broadcasts to all subscribers, so receivers use this list to filter.
    #[serde(default)]
    pub target_peers: Vec<String>,
}

impl QueryRequest {
    pub fn new(query: Query) -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            query,
            query_embedding: None,
            origin_peer: None,
            target_peers: Vec::new(),
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.query_embedding = Some(embedding);
        self
    }

    pub fn with_origin(mut self, peer_id: String) -> Self {
        self.origin_peer = Some(peer_id);
        self
    }

    pub fn with_target_peers(mut self, peers: Vec<String>) -> Self {
        self.target_peers = peers;
        self
    }
}

/// Query response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Matching request ID
    pub request_id: String,
    /// Search results
    pub results: Vec<SearchResult>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Whether results were truncated
    pub truncated: bool,
    /// Responding peer ID
    pub responder_peer: Option<String>,
}

impl QueryResponse {
    pub fn new(request_id: String, results: Vec<SearchResult>) -> Self {
        Self {
            request_id,
            results,
            processing_time_ms: 0,
            truncated: false,
            responder_peer: None,
        }
    }

    pub fn with_timing(mut self, processing_time_ms: u64) -> Self {
        self.processing_time_ms = processing_time_ms;
        self
    }

    pub fn with_responder(mut self, peer_id: String) -> Self {
        self.responder_peer = Some(peer_id);
        self
    }
}

/// GossipSub topics
pub mod topics {
    /// Topic for node advertisements
    pub const ADVERTISEMENTS: &str = "dindex/advertisements";
    /// Topic for query requests
    pub const QUERIES: &str = "dindex/queries";
    /// Topic for query responses
    pub const QUERY_RESPONSES: &str = "dindex/query-responses";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        Chunk, ChunkMetadata, NodeAdvertisement, NodeCentroid, Query, SearchResult,
    };
    use chrono::Utc;

    // ====================================================================
    // QueryRequest tests
    // ====================================================================

    #[test]
    fn test_query_request_new() {
        let query = Query::new("test query", 5);
        let req = QueryRequest::new(query);
        assert!(!req.request_id.is_empty(), "request_id should be auto-generated");
        assert_eq!(req.query.text, "test query");
        assert_eq!(req.query.top_k, 5);
        assert!(req.query_embedding.is_none());
        assert!(req.origin_peer.is_none());
        assert!(req.target_peers.is_empty());
    }

    #[test]
    fn test_query_request_with_embedding() {
        let query = Query::new("embed query", 3);
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let req = QueryRequest::new(query).with_embedding(embedding.clone());
        assert_eq!(req.query_embedding, Some(embedding));
    }

    #[test]
    fn test_query_request_with_origin() {
        let query = Query::new("origin query", 10);
        let req = QueryRequest::new(query).with_origin("peer-abc".to_string());
        assert_eq!(req.origin_peer, Some("peer-abc".to_string()));
    }

    #[test]
    fn test_query_request_builder_chaining() {
        let query = Query::new("chained", 7);
        let req = QueryRequest::new(query)
            .with_embedding(vec![1.0, 2.0])
            .with_origin("peer-xyz".to_string());
        assert_eq!(req.query.text, "chained");
        assert_eq!(req.query_embedding, Some(vec![1.0, 2.0]));
        assert_eq!(req.origin_peer, Some("peer-xyz".to_string()));
    }

    #[test]
    fn test_query_request_with_target_peers() {
        let query = Query::new("targeted", 5);
        let peers = vec!["peer-a".to_string(), "peer-b".to_string()];
        let req = QueryRequest::new(query).with_target_peers(peers.clone());
        assert_eq!(req.target_peers, peers);
    }

    #[test]
    fn test_target_peers_filtering_drops_non_targeted() {
        let query = Query::new("targeted query", 5);
        let req = QueryRequest::new(query)
            .with_target_peers(vec!["peer-a".to_string(), "peer-b".to_string()]);

        // Simulate receiver-side check: local peer is "peer-c" (not in target list)
        let local_peer_id = "peer-c".to_string();
        let should_process = req.target_peers.is_empty()
            || req.target_peers.contains(&local_peer_id);
        assert!(!should_process, "non-targeted peer should drop the query");
    }

    #[test]
    fn test_target_peers_filtering_accepts_targeted() {
        let query = Query::new("targeted query", 5);
        let req = QueryRequest::new(query)
            .with_target_peers(vec!["peer-a".to_string(), "peer-b".to_string()]);

        // Simulate receiver-side check: local peer is "peer-a" (in target list)
        let local_peer_id = "peer-a".to_string();
        let should_process = req.target_peers.is_empty()
            || req.target_peers.contains(&local_peer_id);
        assert!(should_process, "targeted peer should process the query");
    }

    #[test]
    fn test_target_peers_empty_means_broadcast() {
        let query = Query::new("broadcast query", 5);
        let req = QueryRequest::new(query);

        // Empty target_peers = broadcast, all peers should process
        let local_peer_id = "any-peer".to_string();
        let should_process = req.target_peers.is_empty()
            || req.target_peers.contains(&local_peer_id);
        assert!(should_process, "empty target_peers means broadcast to all");
    }

    #[test]
    fn test_target_peers_serialization_roundtrip() {
        let query = Query::new("ser target", 3);
        let peers = vec!["peer-x".to_string(), "peer-y".to_string()];
        let req = QueryRequest::new(query).with_target_peers(peers.clone());

        let serialized = serde_json::to_string(&req).expect("serialize");
        let deserialized: QueryRequest = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(deserialized.target_peers, peers);
    }

    #[test]
    fn test_query_request_serialization_roundtrip() {
        let query = Query::new("roundtrip query", 10);
        let req = QueryRequest::new(query)
            .with_embedding(vec![0.5, -0.3, 1.0])
            .with_origin("peer-1".to_string());

        let serialized = serde_json::to_string(&req).expect("serialize QueryRequest");
        let deserialized: QueryRequest =
            serde_json::from_str(&serialized).expect("deserialize QueryRequest");

        assert_eq!(deserialized.request_id, req.request_id);
        assert_eq!(deserialized.query.text, "roundtrip query");
        assert_eq!(deserialized.query.top_k, 10);
        assert_eq!(deserialized.query_embedding, Some(vec![0.5, -0.3, 1.0]));
        assert_eq!(deserialized.origin_peer, Some("peer-1".to_string()));
    }

    #[test]
    fn test_query_request_bincode_roundtrip() {
        let query = Query::new("bincode query", 3);
        let req = QueryRequest::new(query).with_embedding(vec![1.0, 2.0, 3.0]);

        let encoded = bincode::serialize(&req).expect("bincode serialize QueryRequest");
        let decoded: QueryRequest =
            bincode::deserialize(&encoded).expect("bincode deserialize QueryRequest");

        assert_eq!(decoded.request_id, req.request_id);
        assert_eq!(decoded.query.text, "bincode query");
        assert_eq!(decoded.query_embedding, Some(vec![1.0, 2.0, 3.0]));
    }

    // ====================================================================
    // QueryResponse tests
    // ====================================================================

    fn make_test_search_result(score: f32) -> SearchResult {
        let chunk = Chunk {
            metadata: ChunkMetadata::new("chunk-1".to_string(), "doc-1".to_string()),
            content: "test content".to_string(),
            token_count: 2,
        };
        SearchResult::new(chunk, score)
    }

    #[test]
    fn test_query_response_new() {
        let results = vec![make_test_search_result(0.9)];
        let resp = QueryResponse::new("req-123".to_string(), results);
        assert_eq!(resp.request_id, "req-123");
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.processing_time_ms, 0);
        assert!(!resp.truncated);
        assert!(resp.responder_peer.is_none());
    }

    #[test]
    fn test_query_response_with_timing() {
        let resp = QueryResponse::new("req-1".to_string(), vec![]).with_timing(42);
        assert_eq!(resp.processing_time_ms, 42);
    }

    #[test]
    fn test_query_response_with_responder() {
        let resp =
            QueryResponse::new("req-2".to_string(), vec![]).with_responder("node-A".to_string());
        assert_eq!(resp.responder_peer, Some("node-A".to_string()));
    }

    #[test]
    fn test_query_response_builder_chaining() {
        let results = vec![make_test_search_result(0.8), make_test_search_result(0.6)];
        let resp = QueryResponse::new("req-chain".to_string(), results)
            .with_timing(150)
            .with_responder("node-B".to_string());

        assert_eq!(resp.request_id, "req-chain");
        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.processing_time_ms, 150);
        assert_eq!(resp.responder_peer, Some("node-B".to_string()));
    }

    #[test]
    fn test_query_response_serialization_roundtrip() {
        let results = vec![make_test_search_result(0.75)];
        let resp = QueryResponse::new("req-ser".to_string(), results)
            .with_timing(99)
            .with_responder("node-C".to_string());

        let serialized = serde_json::to_string(&resp).expect("serialize QueryResponse");
        let deserialized: QueryResponse =
            serde_json::from_str(&serialized).expect("deserialize QueryResponse");

        assert_eq!(deserialized.request_id, "req-ser");
        assert_eq!(deserialized.results.len(), 1);
        assert_eq!(deserialized.processing_time_ms, 99);
        assert_eq!(deserialized.responder_peer, Some("node-C".to_string()));
    }

    #[test]
    fn test_query_response_bincode_roundtrip() {
        let resp = QueryResponse::new("req-bin".to_string(), vec![]).with_timing(200);

        let encoded = bincode::serialize(&resp).expect("bincode serialize QueryResponse");
        let decoded: QueryResponse =
            bincode::deserialize(&encoded).expect("bincode deserialize QueryResponse");

        assert_eq!(decoded.request_id, "req-bin");
        assert_eq!(decoded.processing_time_ms, 200);
        assert!(decoded.results.is_empty());
    }

    // ====================================================================
    // NetworkMessage tests
    // ====================================================================

    #[test]
    fn test_network_message_query_request_roundtrip() {
        let query = Query::new("network msg query", 5);
        let req = QueryRequest::new(query);
        let msg = NetworkMessage::QueryRequest(req);

        let serialized = serde_json::to_string(&msg).expect("serialize NetworkMessage");
        let deserialized: NetworkMessage =
            serde_json::from_str(&serialized).expect("deserialize NetworkMessage");

        match deserialized {
            NetworkMessage::QueryRequest(r) => {
                assert_eq!(r.query.text, "network msg query");
                assert_eq!(r.query.top_k, 5);
            }
            _ => panic!("Expected QueryRequest variant"),
        }
    }

    #[test]
    fn test_network_message_query_response_roundtrip() {
        let resp = QueryResponse::new("resp-id".to_string(), vec![]).with_timing(50);
        let msg = NetworkMessage::QueryResponse(resp);

        let serialized = serde_json::to_string(&msg).expect("serialize NetworkMessage");
        let deserialized: NetworkMessage =
            serde_json::from_str(&serialized).expect("deserialize NetworkMessage");

        match deserialized {
            NetworkMessage::QueryResponse(r) => {
                assert_eq!(r.request_id, "resp-id");
                assert_eq!(r.processing_time_ms, 50);
            }
            _ => panic!("Expected QueryResponse variant"),
        }
    }

    #[test]
    fn test_network_message_advertisement_roundtrip() {
        let advert = NodeAdvertisement {
            node_id: "node-adv".to_string(),
            centroids: vec![NodeCentroid {
                centroid_id: 1,
                embedding: vec![0.1, 0.2, 0.3],
                chunk_count: 100,
            }],
            lsh_bloom_filter: Some(vec![0xFF, 0x00, 0xAB]),
            total_chunks: 500,
            last_updated: Utc::now(),
        };
        let msg = NetworkMessage::Advertisement(advert);

        let serialized = serde_json::to_string(&msg).expect("serialize NetworkMessage");
        let deserialized: NetworkMessage =
            serde_json::from_str(&serialized).expect("deserialize NetworkMessage");

        match deserialized {
            NetworkMessage::Advertisement(a) => {
                assert_eq!(a.node_id, "node-adv");
                assert_eq!(a.centroids.len(), 1);
                assert_eq!(a.centroids[0].centroid_id, 1);
                assert_eq!(a.centroids[0].chunk_count, 100);
                assert_eq!(a.total_chunks, 500);
                assert_eq!(a.lsh_bloom_filter, Some(vec![0xFF, 0x00, 0xAB]));
            }
            _ => panic!("Expected Advertisement variant"),
        }
    }

    #[test]
    fn test_network_message_all_variants_bincode() {
        // QueryRequest variant
        let msg1 = NetworkMessage::QueryRequest(QueryRequest::new(Query::new("q", 1)));
        let enc1 = bincode::serialize(&msg1).expect("bincode serialize QueryRequest variant");
        let dec1: NetworkMessage = bincode::deserialize(&enc1).expect("bincode deserialize");
        assert!(matches!(dec1, NetworkMessage::QueryRequest(_)));

        // QueryResponse variant
        let msg2 = NetworkMessage::QueryResponse(QueryResponse::new("r".to_string(), vec![]));
        let enc2 = bincode::serialize(&msg2).expect("bincode serialize QueryResponse variant");
        let dec2: NetworkMessage = bincode::deserialize(&enc2).expect("bincode deserialize");
        assert!(matches!(dec2, NetworkMessage::QueryResponse(_)));

        // Advertisement variant
        let msg3 = NetworkMessage::Advertisement(NodeAdvertisement {
            node_id: "n".to_string(),
            centroids: vec![],
            lsh_bloom_filter: None,
            total_chunks: 0,
            last_updated: Utc::now(),
        });
        let enc3 = bincode::serialize(&msg3).expect("bincode serialize Advertisement variant");
        let dec3: NetworkMessage = bincode::deserialize(&enc3).expect("bincode deserialize");
        assert!(matches!(dec3, NetworkMessage::Advertisement(_)));
    }

    // ====================================================================
    // Protocol constants tests
    // ====================================================================

    #[test]
    fn test_protocol_constants() {
        assert_eq!(PROTOCOL_VERSION, "/dindex/1.0.0");
        assert_eq!(QUERY_PROTOCOL, "/dindex/query/1.0.0");
        assert_eq!(ADVERTISEMENT_PROTOCOL, "/dindex/advertisement/1.0.0");
    }

    #[test]
    fn test_topic_constants() {
        assert_eq!(topics::ADVERTISEMENTS, "dindex/advertisements");
        assert_eq!(topics::QUERIES, "dindex/queries");
        assert_eq!(topics::QUERY_RESPONSES, "dindex/query-responses");
    }
}
