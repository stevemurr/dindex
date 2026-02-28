//! System handlers: health, metrics, status, stats

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};

use super::AppState;
use crate::daemon::http::types::*;
use crate::daemon::protocol::{Request, Response as IpcResponse};

/// Health check endpoint
pub async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        healthy: true,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Prometheus metrics endpoint
pub async fn prometheus_metrics(State(state): State<AppState>) -> impl IntoResponse {
    state.handler.metrics().update_memory_usage();
    let body = state.handler.metrics().to_prometheus();
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

/// Status endpoint
pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    match state.handler.handle(Request::Status).await {
        IpcResponse::Status(status) => (
            StatusCode::OK,
            Json(StatusResponse {
                running: status.running,
                uptime_seconds: status.uptime_seconds,
                memory_mb: status.memory_mb,
                active_jobs: status.active_jobs,
                pending_writes: status.pending_writes,
            }),
        )
            .into_response(),
        IpcResponse::Error { code, message } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(format!("{:?}", code), message)),
        )
            .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::internal_error("Unexpected response type")),
        )
            .into_response(),
    }
}

/// Stats endpoint
pub async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    match state.handler.handle(Request::Stats).await {
        IpcResponse::Stats(stats) => (
            StatusCode::OK,
            Json(StatsResponse {
                total_documents: stats.total_documents,
                total_chunks: stats.total_chunks,
                vector_index_size_bytes: stats.vector_index_size_bytes,
                bm25_index_size_bytes: stats.bm25_index_size_bytes,
                storage_size_bytes: stats.storage_size_bytes,
            }),
        )
            .into_response(),
        IpcResponse::Error { code, message } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(format!("{:?}", code), message)),
        )
            .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::internal_error("Unexpected response type")),
        )
            .into_response(),
    }
}
