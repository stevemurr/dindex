//! HTTP API Request/Response Types
//!
//! JSON-serializable types for the HTTP API.

use serde::{Deserialize, Serialize};

use crate::types::{Chunk, ChunkMetadata};

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

/// A single search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultJson {
    /// The matched chunk
    pub chunk: ChunkJson,
    /// Relevance score (0.0 to 1.0)
    pub relevance_score: f32,
    /// Which retrieval methods matched
    pub matched_by: Vec<String>,
}

/// Chunk data for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkJson {
    /// The chunk content text
    pub content: String,
    /// Chunk metadata
    pub metadata: ChunkMetadataJson,
}

/// Chunk metadata for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadataJson {
    /// Unique chunk identifier
    pub chunk_id: String,
    /// Parent document identifier
    pub document_id: String,
    /// Source URL if available
    pub source_url: Option<String>,
    /// Source title if available
    pub source_title: Option<String>,
}

impl From<&Chunk> for ChunkJson {
    fn from(chunk: &Chunk) -> Self {
        Self {
            content: chunk.content.clone(),
            metadata: ChunkMetadataJson {
                chunk_id: chunk.metadata.chunk_id.clone(),
                document_id: chunk.metadata.document_id.clone(),
                source_url: chunk.metadata.source_url.clone(),
                source_title: chunk.metadata.source_title.clone(),
            },
        }
    }
}

impl From<&ChunkMetadata> for ChunkMetadataJson {
    fn from(meta: &ChunkMetadata) -> Self {
        Self {
            chunk_id: meta.chunk_id.clone(),
            document_id: meta.document_id.clone(),
            source_url: meta.source_url.clone(),
            source_title: meta.source_title.clone(),
        }
    }
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
    pub matched_by: Vec<String>,
    /// Section hierarchy in the document
    pub section_hierarchy: Vec<String>,
    /// Position in document (0.0 to 1.0)
    pub position_in_doc: f32,
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
    /// Matching chunks sorted by score descending
    pub chunks: Vec<MatchingChunkJson>,
}

/// Search response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results grouped by document
    pub results: Vec<GroupedSearchResultJson>,
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

/// Scrape options for HTTP API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapeOptionsJson {
    /// Maximum crawl depth (0 = current page only)
    #[serde(default = "default_max_depth")]
    pub max_depth: u8,
    /// Stay on the same domain
    #[serde(default = "default_stay_on_domain")]
    pub stay_on_domain: bool,
    /// Delay between requests in milliseconds
    #[serde(default = "default_delay_ms")]
    pub delay_ms: u64,
    /// Maximum number of pages to scrape
    #[serde(default = "default_max_pages")]
    pub max_pages: usize,
}

fn default_max_depth() -> u8 {
    2
}

fn default_stay_on_domain() -> bool {
    true
}

fn default_delay_ms() -> u64 {
    1000
}

fn default_max_pages() -> usize {
    100
}

impl Default for ScrapeOptionsJson {
    fn default() -> Self {
        Self {
            max_depth: default_max_depth(),
            stay_on_domain: default_stay_on_domain(),
            delay_ms: default_delay_ms(),
            max_pages: default_max_pages(),
        }
    }
}

/// Scrape request body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapeRequest {
    /// URLs to start scraping from
    pub urls: Vec<String>,
    /// Scrape options
    #[serde(default)]
    pub options: ScrapeOptionsJson,
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
