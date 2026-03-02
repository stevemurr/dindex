//! HTTP API Request Handlers
//!
//! Handlers that map HTTP requests to RequestHandler operations.

mod cluster;
mod index;
mod jobs;
mod search;
mod system;

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::daemon::handler::RequestHandler;
use crate::daemon::http::types::ErrorResponse;
use crate::daemon::protocol::ErrorCode;

/// Convert an IPC error to an HTTP error response.
/// Maps specific error codes to appropriate HTTP status codes.
pub(crate) fn ipc_error(code: ErrorCode, message: String) -> Response {
    let status = match code {
        ErrorCode::JobNotFound => StatusCode::NOT_FOUND,
        ErrorCode::InvalidRequest => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(ErrorResponse::new(format!("{:?}", code), message))).into_response()
}

/// Standard response for unexpected IPC response types.
pub(crate) fn unexpected_response() -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse::internal_error("Unexpected response type")),
    )
        .into_response()
}

/// Return a BAD_REQUEST error response.
pub(crate) fn bad_request(code: impl Into<String>, message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse::new(code, message)),
    )
        .into_response()
}

/// Maximum allowed query length (10KB)
const MAX_QUERY_LENGTH: usize = 10_000;

/// Maximum allowed document content size (10MB)
const MAX_DOCUMENT_SIZE: usize = 10 * 1024 * 1024;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub handler: Arc<RequestHandler>,
}

// Re-export all handlers
pub use cluster::cluster_documents;
pub use index::{clear_index, commit, delete_documents, index_documents};
pub use jobs::{cancel_job, get_job_progress, job_events_sse, start_scrape};
pub use search::search;
pub use system::{health, prometheus_metrics, stats, status};
