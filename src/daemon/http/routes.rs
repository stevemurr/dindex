//! HTTP API Route Definitions
//!
//! Defines the REST API routes for dindex.

use axum::{
    middleware,
    routing::{delete, get, post},
    Router,
};

use super::auth::{auth_middleware, AuthState};
use super::handlers::{self, AppState};

/// Create the API router with all routes
pub fn create_router(app_state: AppState, auth_state: AuthState) -> Router {
    // API v1 routes
    let api_v1 = Router::new()
        // Health check (no auth required)
        .route("/health", get(handlers::health))
        // Protected routes
        .route("/search", post(handlers::search))
        .route("/status", get(handlers::status))
        .route("/stats", get(handlers::stats))
        .route("/index", post(handlers::index_documents))
        .route("/index/commit", post(handlers::commit))
        .route("/index/clear", post(handlers::clear_index))
        .route("/documents", delete(handlers::delete_documents))
        // Scrape routes
        .route("/scrape", post(handlers::start_scrape))
        .route("/jobs/:job_id", get(handlers::get_job_progress))
        .route("/jobs/:job_id/events", get(handlers::job_events_sse))
        .route("/jobs/:job_id/cancel", post(handlers::cancel_job))
        .layer(middleware::from_fn_with_state(
            auth_state.clone(),
            auth_middleware,
        ))
        .with_state(app_state);

    // Mount under /api/v1
    Router::new().nest("/api/v1", api_v1)
}
