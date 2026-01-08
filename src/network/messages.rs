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
    /// Batch of URLs for scraping exchange
    UrlBatch(UrlBatchMessage),
    /// SimHash query for content deduplication
    SimHashQuery(SimHashQueryMessage),
    /// SimHash query response
    SimHashResponse(SimHashResponseMessage),
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

/// Batch of URLs to exchange between scraping nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlBatchMessage {
    /// Unique batch ID
    pub batch_id: String,
    /// Source peer
    pub from_peer: String,
    /// URLs grouped by target hostname (hostname, url_string, priority, depth)
    pub urls: Vec<ScrapingUrl>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// A URL being exchanged for scraping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapingUrl {
    /// Target hostname
    pub hostname: String,
    /// Full URL
    pub url: String,
    /// Priority score
    pub priority: f32,
    /// Depth from seed
    pub depth: u8,
}

impl UrlBatchMessage {
    pub fn new(from_peer: String) -> Self {
        Self {
            batch_id: uuid::Uuid::new_v4().to_string(),
            from_peer,
            urls: Vec::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn add_url(&mut self, hostname: String, url: String, priority: f32, depth: u8) {
        self.urls.push(ScrapingUrl {
            hostname,
            url,
            priority,
            depth,
        });
    }

    pub fn len(&self) -> usize {
        self.urls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.urls.is_empty()
    }
}

/// SimHash query for content deduplication across the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimHashQueryMessage {
    /// Unique query ID
    pub query_id: String,
    /// The SimHash value to check
    pub simhash: u64,
    /// Maximum Hamming distance for near-duplicates
    pub max_distance: u32,
    /// Requesting peer
    pub from_peer: String,
}

impl SimHashQueryMessage {
    pub fn new(simhash: u64, max_distance: u32, from_peer: String) -> Self {
        Self {
            query_id: uuid::Uuid::new_v4().to_string(),
            simhash,
            max_distance,
            from_peer,
        }
    }
}

/// Response to a SimHash query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimHashResponseMessage {
    /// Matching query ID
    pub query_id: String,
    /// Whether a duplicate was found
    pub is_duplicate: bool,
    /// The existing document ID if duplicate
    pub existing_document_id: Option<String>,
    /// The existing SimHash if found
    pub existing_simhash: Option<u64>,
    /// Hamming distance to existing
    pub hamming_distance: Option<u32>,
    /// Responding peer
    pub responder_peer: String,
}

impl SimHashResponseMessage {
    pub fn not_found(query_id: String, responder_peer: String) -> Self {
        Self {
            query_id,
            is_duplicate: false,
            existing_document_id: None,
            existing_simhash: None,
            hamming_distance: None,
            responder_peer,
        }
    }

    pub fn found(
        query_id: String,
        document_id: String,
        simhash: u64,
        distance: u32,
        responder_peer: String,
    ) -> Self {
        Self {
            query_id,
            is_duplicate: true,
            existing_document_id: Some(document_id),
            existing_simhash: Some(simhash),
            hamming_distance: Some(distance),
            responder_peer,
        }
    }
}

/// GossipSub topics
pub mod topics {
    /// Topic for node advertisements
    pub const ADVERTISEMENTS: &str = "dindex/advertisements";
    /// Topic for index updates
    pub const INDEX_UPDATES: &str = "dindex/index-updates";
    /// Topic for query requests
    pub const QUERIES: &str = "dindex/queries";
    /// Topic for query responses
    pub const QUERY_RESPONSES: &str = "dindex/query-responses";
    /// Topic for URL exchange between scraping nodes
    pub const URL_EXCHANGE: &str = "dindex/url-exchange";
    /// Topic for SimHash deduplication queries
    pub const SIMHASH_QUERIES: &str = "dindex/simhash-queries";
}
