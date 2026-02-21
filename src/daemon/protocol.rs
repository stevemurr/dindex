//! IPC Protocol Types
//!
//! Defines the request/response types for daemon-client communication.
//! Uses length-prefixed binary protocol with bincode serialization.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::{Chunk, ChunkMetadata, QueryFilters, SearchResult};

/// Output format for search results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    JsonPretty,
}

/// Source for bulk import operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportSource {
    WikimediaXml { path: String },
    Zim { path: String },
    Warc { path: String },
}

/// Options for import operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportOptions {
    pub batch_size: usize,
    pub deduplicate: bool,
    pub min_content_length: usize,
    pub max_documents: Option<usize>,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            batch_size: 100,
            deduplicate: true,
            min_content_length: 100,
            max_documents: None,
        }
    }
}

/// Options for scrape operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapeOptions {
    pub max_depth: u8,
    pub stay_on_domain: bool,
    pub delay_ms: u64,
    pub max_pages: usize,
}

impl Default for ScrapeOptions {
    fn default() -> Self {
        Self {
            max_depth: 2,
            stay_on_domain: false,
            delay_ms: 1000,
            max_pages: 100,
        }
    }
}

/// Payload for streaming chunks during indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkPayload {
    pub content: String,
    pub metadata: ChunkMetadata,
    pub embedding: Option<Vec<f32>>,
}

impl From<Chunk> for ChunkPayload {
    fn from(chunk: Chunk) -> Self {
        Self {
            content: chunk.content,
            metadata: chunk.metadata,
            embedding: None,
        }
    }
}

/// Request types sent from client to daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    // ============ Query Operations ============
    /// Search the index
    Search {
        query: String,
        top_k: usize,
        format: OutputFormat,
        filters: Option<QueryFilters>,
    },

    // ============ Write Operations ============
    /// Start a new document indexing stream
    IndexDocuments { stream_id: Uuid },

    /// Send a chunk for indexing
    IndexChunk { stream_id: Uuid, chunk: ChunkPayload },

    /// Complete the indexing stream and commit
    IndexComplete { stream_id: Uuid },

    // ============ Import Operations ============
    /// Start a bulk import job
    ImportStart {
        source: ImportSource,
        options: ImportOptions,
    },

    /// Cancel a running import job
    ImportCancel { job_id: Uuid },

    /// Get progress of a job
    JobProgress { job_id: Uuid },

    // ============ Scrape Operations ============
    /// Start a web scraping job
    ScrapeStart {
        urls: Vec<String>,
        options: ScrapeOptions,
    },

    /// Cancel a running scrape job
    ScrapeCancel { job_id: Uuid },

    // ============ Management Operations ============
    /// Check if daemon is running (ping)
    Ping,

    /// Get daemon status
    Status,

    /// Get index statistics
    Stats,

    /// Force a commit of pending writes
    ForceCommit,

    /// Graceful shutdown
    Shutdown,
}

/// Progress information for long-running jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    pub job_id: Uuid,
    pub stage: String,
    pub current: u64,
    pub total: Option<u64>,
    pub rate: Option<f64>,
    pub eta_seconds: Option<u64>,
}

/// Statistics for completed jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStats {
    pub documents_processed: usize,
    pub chunks_indexed: usize,
    pub duration_ms: u64,
    pub errors: usize,
}

/// Current daemon status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub uptime_seconds: u64,
    pub memory_mb: u64,
    pub active_jobs: usize,
    pub pending_writes: usize,
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub vector_index_size_bytes: u64,
    pub bm25_index_size_bytes: u64,
    pub storage_size_bytes: u64,
}

/// Error codes for response errors
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ErrorCode {
    InternalError,
    SearchFailed,
    IndexFailed,
    ImportFailed,
    ScrapeFailed,
    JobNotFound,
    InvalidRequest,
    ShuttingDown,
    StreamNotFound,
}

/// Response types sent from daemon to client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    // ============ Query Results ============
    /// Search results
    SearchResults {
        results: Vec<SearchResult>,
        query_time_ms: u64,
    },

    // ============ Streaming Acknowledgments ============
    /// Stream is ready to receive chunks
    StreamReady { stream_id: Uuid },

    /// Acknowledgment of received chunk
    ChunkAck { stream_id: Uuid, count: usize },

    // ============ Job Status ============
    /// Job has started
    JobStarted { job_id: Uuid },

    /// Job progress update
    JobProgress { job_id: Uuid, progress: Progress },

    /// Job completed successfully
    JobComplete { job_id: Uuid, stats: JobStats },

    /// Job failed
    JobFailed { job_id: Uuid, error: String },

    // ============ Status Responses ============
    /// Ping response
    Pong,

    /// Daemon status
    Status(DaemonStatus),

    /// Index statistics
    Stats(IndexStats),

    // ============ Generic Responses ============
    /// Operation succeeded
    Ok,

    /// Operation failed with error
    Error { code: ErrorCode, message: String },
}

impl Response {
    /// Create an error response
    pub fn error(code: ErrorCode, message: impl Into<String>) -> Self {
        Self::Error {
            code,
            message: message.into(),
        }
    }

    /// Create an internal error response
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::error(ErrorCode::InternalError, message)
    }
}

/// Wire format for messages (length-prefixed)
pub const MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024; // 64MB max message

/// Encode a message to bytes with length prefix
pub fn encode_message<T: Serialize>(msg: &T) -> anyhow::Result<Vec<u8>> {
    let payload = bincode::serialize(msg)?;
    if payload.len() > MAX_MESSAGE_SIZE {
        anyhow::bail!("Message too large: {} bytes", payload.len());
    }
    let len = (payload.len() as u32).to_le_bytes();
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len);
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Decode a message from bytes (after length prefix is read)
pub fn decode_message<T: for<'de> Deserialize<'de>>(data: &[u8]) -> anyhow::Result<T> {
    Ok(bincode::deserialize(data)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let req = Request::Search {
            query: "test query".to_string(),
            top_k: 10,
            format: OutputFormat::Json,
            filters: None,
        };

        let encoded = encode_message(&req).unwrap();
        assert!(encoded.len() > 4); // At least length prefix

        // Skip length prefix and decode
        let decoded: Request = decode_message(&encoded[4..]).unwrap();
        match decoded {
            Request::Search { query, top_k, .. } => {
                assert_eq!(query, "test query");
                assert_eq!(top_k, 10);
            }
            _ => panic!("Wrong request type"),
        }
    }

    #[test]
    fn test_response_serialization() {
        let resp = Response::Ok;
        let encoded = encode_message(&resp).unwrap();
        let decoded: Response = decode_message(&encoded[4..]).unwrap();
        assert!(matches!(decoded, Response::Ok));
    }
}
