//! HTTP API Request Handlers
//!
//! Handlers that map HTTP requests to RequestHandler operations.

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error};
use uuid::Uuid;

use crate::daemon::handler::RequestHandler;
use crate::daemon::protocol::{self, OutputFormat, Request, Response as IpcResponse};
use crate::types::Document;

use super::types::*;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub handler: Arc<RequestHandler>,
}

/// Health check endpoint
pub async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        healthy: true,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Search endpoint
pub async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> impl IntoResponse {
    debug!("HTTP search request: query={}, top_k={}", request.query, request.top_k);

    let ipc_request = Request::Search {
        query: request.query,
        top_k: request.top_k,
        format: OutputFormat::Json,
    };

    match state.handler.handle(ipc_request).await {
        IpcResponse::SearchResults { results, query_time_ms } => {
            let json_results: Vec<SearchResultJson> = results
                .iter()
                .map(|r| SearchResultJson {
                    chunk: ChunkJson::from(&r.chunk),
                    relevance_score: r.relevance_score,
                    matched_by: r.matched_by.clone(),
                })
                .collect();

            (
                StatusCode::OK,
                Json(SearchResponse {
                    results: json_results,
                    query_time_ms,
                }),
            )
                .into_response()
        }
        IpcResponse::Error { code, message } => {
            error!("Search failed: {:?} - {}", code, message);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(format!("{:?}", code), message)),
            )
                .into_response()
        }
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::internal_error("Unexpected response type")),
        )
            .into_response(),
    }
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

/// Index documents endpoint
pub async fn index_documents(
    State(state): State<AppState>,
    Json(request): Json<IndexRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let stream_id = Uuid::new_v4();

    debug!("HTTP index request: {} documents", request.documents.len());

    // Start the indexing stream
    match state.handler.handle(Request::IndexDocuments { stream_id }).await {
        IpcResponse::StreamReady { .. } => {}
        IpcResponse::Error { code, message } => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(format!("{:?}", code), message)),
            )
                .into_response();
        }
        _ => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::internal_error("Failed to start indexing stream")),
            )
                .into_response();
        }
    }

    // Send each document as chunks
    // Note: In a real implementation, we'd chunk the documents properly.
    // For now, we send each document as a single chunk.
    for doc in &request.documents {
        let document = Document::new(&doc.content)
            .with_title(doc.title.clone().unwrap_or_default())
            .with_url(doc.url.clone().unwrap_or_default());

        let chunk = crate::types::Chunk {
            metadata: crate::types::ChunkMetadata::new(
                Uuid::new_v4().to_string(),
                document.id.clone(),
            ),
            content: document.content.clone(),
            token_count: 0,
        };

        // Update metadata with document info
        let mut chunk = chunk;
        chunk.metadata.source_title = doc.title.clone();
        chunk.metadata.source_url = doc.url.clone();

        let payload = protocol::ChunkPayload {
            content: chunk.content.clone(),
            metadata: chunk.metadata.clone(),
            embedding: None,
        };

        match state
            .handler
            .handle(Request::IndexChunk {
                stream_id,
                chunk: payload,
            })
            .await
        {
            IpcResponse::ChunkAck { .. } => {}
            IpcResponse::Error { code, message } => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new(format!("{:?}", code), message)),
                )
                    .into_response();
            }
            _ => {}
        }
    }

    // Complete the stream
    match state.handler.handle(Request::IndexComplete { stream_id }).await {
        IpcResponse::JobComplete { stats, .. } => {
            let duration_ms = start.elapsed().as_millis() as u64;
            (
                StatusCode::OK,
                Json(IndexResponse {
                    documents_indexed: request.documents.len(),
                    chunks_created: stats.chunks_indexed,
                    duration_ms,
                }),
            )
                .into_response()
        }
        IpcResponse::Error { code, message } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(format!("{:?}", code), message)),
        )
            .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::internal_error("Failed to complete indexing")),
        )
            .into_response(),
    }
}

/// Commit endpoint
pub async fn commit(State(state): State<AppState>) -> impl IntoResponse {
    match state.handler.handle(Request::ForceCommit).await {
        IpcResponse::Ok => (StatusCode::OK, Json(CommitResponse { success: true })).into_response(),
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
