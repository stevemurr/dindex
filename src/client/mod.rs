//! Client Module
//!
//! Provides client-side IPC communication with the daemon.
//! CLI commands use this module to send requests and receive responses.

pub mod connection;

pub use connection::DaemonClient;

use crate::daemon::protocol::{
    ChunkPayload, DaemonStatus, ImportOptions, ImportSource, IndexStats, JobStats, OutputFormat,
    Progress, Request, Response, ScrapeOptions,
};
use crate::types::{Chunk, SearchResult};

use anyhow::Result;
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur when communicating with the daemon
#[derive(Debug, Error)]
pub enum ClientError {
    #[error("Daemon is not running. Start it with: dindex daemon start")]
    DaemonNotRunning,

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Request failed: {0}")]
    RequestFailed(String),

    #[error("Unexpected response from daemon")]
    UnexpectedResponse,

    #[error("Search failed: {0}")]
    SearchFailed(String),

    #[error("Index operation failed: {0}")]
    IndexFailed(String),

    #[error("Import operation failed: {0}")]
    ImportFailed(String),

    #[error("Scrape operation failed: {0}")]
    ScrapeFailed(String),

    #[error("Job not found: {0}")]
    JobNotFound(Uuid),

    #[error("Daemon error: {0}")]
    DaemonError(String),
}

/// Search the index via daemon
pub async fn search(
    query: &str,
    top_k: usize,
    format: OutputFormat,
) -> Result<Vec<SearchResult>, ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client
        .send(Request::Search {
            query: query.to_string(),
            top_k,
            format,
            filters: None,
        })
        .await?;

    match response {
        Response::SearchResults { results, .. } => Ok(results),
        Response::Error { message, .. } => Err(ClientError::SearchFailed(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Get daemon status
pub async fn status() -> Result<DaemonStatus, ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client.send(Request::Status).await?;

    match response {
        Response::Status(status) => Ok(status),
        Response::Error { message, .. } => Err(ClientError::DaemonError(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Get index statistics
pub async fn stats() -> Result<IndexStats, ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client.send(Request::Stats).await?;

    match response {
        Response::Stats(stats) => Ok(stats),
        Response::Error { message, .. } => Err(ClientError::DaemonError(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Ping the daemon to check if it's running
pub async fn ping() -> Result<bool, ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client.send(Request::Ping).await?;

    Ok(matches!(response, Response::Pong))
}

/// Request daemon shutdown
pub async fn shutdown() -> Result<(), ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client.send(Request::Shutdown).await?;

    match response {
        Response::Ok => Ok(()),
        Response::Error { message, .. } => Err(ClientError::DaemonError(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Force a commit of pending writes
pub async fn force_commit() -> Result<(), ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client.send(Request::ForceCommit).await?;

    match response {
        Response::Ok => Ok(()),
        Response::Error { message, .. } => Err(ClientError::DaemonError(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Index chunks via daemon
/// Returns job stats upon completion
pub async fn index_chunks(chunks: Vec<Chunk>) -> Result<JobStats, ClientError> {
    let mut client = DaemonClient::connect().await?;
    let stream_id = Uuid::new_v4();

    // Start stream
    let response = client
        .send(Request::IndexDocuments { stream_id })
        .await?;

    match response {
        Response::StreamReady { stream_id: sid } => {
            if sid != stream_id {
                return Err(ClientError::UnexpectedResponse);
            }
        }
        Response::Error { message, .. } => return Err(ClientError::IndexFailed(message)),
        _ => return Err(ClientError::UnexpectedResponse),
    }

    // Send chunks
    for chunk in chunks {
        let payload = ChunkPayload::from(chunk);
        let response = client
            .send(Request::IndexChunk {
                stream_id,
                chunk: payload,
            })
            .await?;

        match response {
            Response::ChunkAck { .. } => {}
            Response::Error { message, .. } => return Err(ClientError::IndexFailed(message)),
            _ => return Err(ClientError::UnexpectedResponse),
        }
    }

    // Complete and wait for commit
    let response = client.send(Request::IndexComplete { stream_id }).await?;

    match response {
        Response::JobComplete { stats, .. } => Ok(stats),
        Response::Error { message, .. } => Err(ClientError::IndexFailed(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Start an import job
/// Returns job_id for tracking progress
pub async fn start_import(
    source: ImportSource,
    options: ImportOptions,
) -> Result<Uuid, ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client
        .send(Request::ImportStart { source, options })
        .await?;

    match response {
        Response::JobStarted { job_id } => Ok(job_id),
        Response::Error { message, .. } => Err(ClientError::ImportFailed(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Cancel an import job
pub async fn cancel_import(job_id: Uuid) -> Result<(), ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client.send(Request::ImportCancel { job_id }).await?;

    match response {
        Response::Ok => Ok(()),
        Response::Error { message, .. } => Err(ClientError::DaemonError(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Start a scrape job
/// Returns job_id for tracking progress
pub async fn start_scrape(urls: Vec<String>, options: ScrapeOptions) -> Result<Uuid, ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client
        .send(Request::ScrapeStart { urls, options })
        .await?;

    match response {
        Response::JobStarted { job_id } => Ok(job_id),
        Response::Error { message, .. } => Err(ClientError::ScrapeFailed(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Cancel a scrape job
pub async fn cancel_scrape(job_id: Uuid) -> Result<(), ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client.send(Request::ScrapeCancel { job_id }).await?;

    match response {
        Response::Ok => Ok(()),
        Response::Error { message, .. } => Err(ClientError::DaemonError(message)),
        _ => Err(ClientError::UnexpectedResponse),
    }
}

/// Get progress of a job
pub async fn job_progress(job_id: Uuid) -> Result<Progress, ClientError> {
    let mut client = DaemonClient::connect().await?;

    let response = client.send(Request::JobProgress { job_id }).await?;

    match response {
        Response::JobProgress { progress, .. } => Ok(progress),
        Response::Error { code, message } => {
            if code == crate::daemon::protocol::ErrorCode::JobNotFound {
                Err(ClientError::JobNotFound(job_id))
            } else {
                Err(ClientError::DaemonError(message))
            }
        }
        _ => Err(ClientError::UnexpectedResponse),
    }
}
