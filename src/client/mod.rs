//! Client Module
//!
//! Provides client-side IPC communication with the daemon.
//! CLI commands use this module to send requests and receive responses.

pub mod connection;

pub use connection::DaemonClient;

use crate::daemon::protocol::{
    DaemonStatus, IndexStats, OutputFormat, Request, Response,
};
use crate::types::SearchResult;

use anyhow::Result;
use thiserror::Error;

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
