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
    /// Ping/keepalive
    Ping,
    /// Pong response
    Pong,
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
}

impl QueryRequest {
    pub fn new(query: Query) -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            query,
            query_embedding: None,
            origin_peer: None,
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
