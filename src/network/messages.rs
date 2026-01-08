//! Network protocol messages

use crate::types::{ChunkId, NodeAdvertisement, Query, SearchResult};
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
    /// Request for specific chunks by ID
    ChunkRequest(ChunkRequest),
    /// Response with chunk data
    ChunkResponse(ChunkResponse),
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
}

impl QueryRequest {
    pub fn new(query: Query) -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            query,
            query_embedding: None,
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.query_embedding = Some(embedding);
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
}

impl QueryResponse {
    pub fn new(request_id: String, results: Vec<SearchResult>) -> Self {
        Self {
            request_id,
            results,
            processing_time_ms: 0,
            truncated: false,
        }
    }
}

/// Request for specific chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRequest {
    pub request_id: String,
    pub chunk_ids: Vec<ChunkId>,
}

/// Response with chunk data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResponse {
    pub request_id: String,
    pub chunks: Vec<crate::types::Chunk>,
}

/// GossipSub topics
pub mod topics {
    /// Topic for node advertisements
    pub const ADVERTISEMENTS: &str = "dindex/advertisements";
    /// Topic for index updates
    pub const INDEX_UPDATES: &str = "dindex/index-updates";
}
