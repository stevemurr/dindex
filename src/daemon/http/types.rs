//! HTTP API Request/Response Types
//!
//! JSON-serializable types for the HTTP API.

use serde::{Deserialize, Serialize};

use crate::daemon::protocol::ScrapeOptions;
use crate::retrieval::RetrievalMethod;

/// Search request body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// The search query text
    pub query: String,
    /// Number of results to return (default: 10)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Optional filters
    #[serde(default)]
    pub filters: Option<SearchFilters>,
}

fn default_top_k() -> usize {
    10
}

/// Optional search filters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchFilters {
    /// Filter by source URL prefix
    pub source_url_prefix: Option<String>,
    /// Filter where metadata key exactly equals value
    pub metadata_equals: Option<std::collections::HashMap<String, String>>,
    /// Filter where metadata key's value is in the provided list
    pub metadata_contains: Option<std::collections::HashMap<String, Vec<String>>>,
}

/// A matching chunk within a grouped search result (JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingChunkJson {
    /// Chunk identifier
    pub chunk_id: String,
    /// The chunk content text
    pub content: String,
    /// Relevance score (0.0 to 1.0)
    pub relevance_score: f32,
    /// Which retrieval methods matched
    pub matched_by: Vec<RetrievalMethod>,
    /// Section hierarchy in the document
    pub section_hierarchy: Vec<String>,
    /// Position in document (0.0 to 1.0)
    pub position_in_doc: f32,
    /// Best-matching sentence snippet for citations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

/// Search results grouped by document (JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupedSearchResultJson {
    /// Parent document identifier
    pub document_id: String,
    /// Source URL if available
    pub source_url: Option<String>,
    /// Source title if available
    pub source_title: Option<String>,
    /// Maximum relevance score among all matching chunks
    pub relevance_score: f32,
    /// 1-based citation index linking to the citations array
    pub citation_index: usize,
    /// Matching chunks sorted by score descending
    pub chunks: Vec<MatchingChunkJson>,
}

/// A citation entry bundling source metadata and the best snippet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// 1-based citation index
    pub index: usize,
    /// Source title if available
    pub source_title: Option<String>,
    /// Source URL if available
    pub source_url: Option<String>,
    /// Best snippet from the top-scoring chunk
    pub snippet: Option<String>,
}

/// Search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results grouped by document
    pub results: Vec<GroupedSearchResultJson>,
    /// Citation entries for each grouped result
    pub citations: Vec<Citation>,
    /// Total number of unique documents matched
    pub total_documents: usize,
    /// Total number of matching chunks across all documents
    pub total_chunks: usize,
    /// Query execution time in milliseconds
    pub query_time_ms: u64,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Whether the service is healthy
    pub healthy: bool,
    /// Service version
    pub version: String,
}

/// Daemon status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    /// Whether the daemon is running
    pub running: bool,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Memory usage in MB
    pub memory_mb: u64,
    /// Number of active background jobs
    pub active_jobs: usize,
    /// Number of pending writes
    pub pending_writes: usize,
}

/// Index statistics response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsResponse {
    /// Total number of documents
    pub total_documents: usize,
    /// Total number of chunks
    pub total_chunks: usize,
    /// Vector index size in bytes
    pub vector_index_size_bytes: u64,
    /// BM25 index size in bytes
    pub bm25_index_size_bytes: u64,
    /// Total storage size in bytes
    pub storage_size_bytes: u64,
}

/// Index documents request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRequest {
    /// Documents to index
    pub documents: Vec<DocumentJson>,
}

/// Document for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentJson {
    /// Document content
    pub content: String,
    /// Optional document title
    pub title: Option<String>,
    /// Optional source URL
    pub url: Option<String>,
    /// Optional custom metadata (key-value pairs)
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

/// Index response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResponse {
    /// Number of documents indexed
    pub documents_indexed: usize,
    /// Number of chunks created
    pub chunks_created: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

/// Commit response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResponse {
    /// Whether the commit was successful
    pub success: bool,
}

/// Delete documents request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    /// Document IDs to delete
    pub document_ids: Vec<String>,
}

/// Delete response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    /// Number of documents deleted
    pub documents_deleted: usize,
    /// Number of chunks deleted
    pub chunks_deleted: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

/// Clear index response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearResponse {
    /// Number of chunks deleted
    pub chunks_deleted: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error code
    pub code: String,
    /// Human-readable error message
    pub message: String,
}

// ============ Scrape Types ============

/// Scrape request body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapeRequest {
    /// URLs to start scraping from
    pub urls: Vec<String>,
    /// Scrape options
    #[serde(default)]
    pub options: ScrapeOptions,
}

/// Response when a job is started
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStartedResponse {
    /// The job ID
    pub job_id: String,
    /// Human-readable message
    pub message: String,
}

/// Response for job progress queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobProgressResponse {
    /// The job ID
    pub job_id: String,
    /// Current stage of the job
    pub stage: String,
    /// Current progress count
    pub current: u64,
    /// Total count (if known)
    pub total: Option<u64>,
    /// Processing rate (items/sec)
    pub rate: Option<f64>,
    /// Estimated time remaining in seconds
    pub eta_seconds: Option<u64>,
}

/// Response for job cancellation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobCancelResponse {
    /// Whether the cancellation was successful
    pub success: bool,
    /// Human-readable message
    pub message: String,
}

impl ErrorResponse {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new("INTERNAL_ERROR", message)
    }

    pub fn unauthorized() -> Self {
        Self::new("UNAUTHORIZED", "Invalid or missing API key")
    }
}
