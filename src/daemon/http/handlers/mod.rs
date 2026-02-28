//! HTTP API Request Handlers
//!
//! Handlers that map HTTP requests to RequestHandler operations.

mod index;
mod jobs;
mod search;
mod system;

use std::sync::Arc;

use crate::daemon::handler::RequestHandler;

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
pub use index::{clear_index, commit, delete_documents, index_documents};
pub use jobs::{cancel_job, get_job_progress, job_events_sse, start_scrape};
pub use search::search;
pub use system::{health, prometheus_metrics, stats, status};
