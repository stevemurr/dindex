//! Index document handlers: index, delete, clear, commit

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use std::time::Instant;
use tracing::{debug, error};
use uuid::Uuid;

use super::{AppState, MAX_DOCUMENT_SIZE};
use super::super::types::*;
use crate::daemon::protocol::{self, Request, Response as IpcResponse};
use crate::types::Document;

/// Index documents endpoint
pub async fn index_documents(
    State(state): State<AppState>,
    Json(request): Json<IndexRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let stream_id = Uuid::new_v4();

    debug!("HTTP index request: {} documents", request.documents.len());

    // Validate document content sizes
    for (i, doc) in request.documents.iter().enumerate() {
        if doc.content.len() > MAX_DOCUMENT_SIZE {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "DOCUMENT_TOO_LARGE".to_string(),
                    format!(
                        "Document {} content length {} exceeds maximum allowed size of {} bytes",
                        i,
                        doc.content.len(),
                        MAX_DOCUMENT_SIZE
                    ),
                )),
            )
                .into_response();
        }
    }

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
    for doc in &request.documents {
        let mut document = Document::new(&doc.content)
            .with_title(doc.title.clone().unwrap_or_default())
            .with_url(doc.url.clone().unwrap_or_default());

        // Copy metadata from document request
        if let Some(ref metadata) = doc.metadata {
            document.metadata = metadata.clone();
        }

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
        // Copy extra metadata from document
        chunk.metadata.extra = document.metadata.clone();

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

/// Delete documents endpoint
pub async fn delete_documents(
    State(state): State<AppState>,
    Json(request): Json<DeleteRequest>,
) -> impl IntoResponse {
    let start = Instant::now();

    if request.document_ids.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "INVALID_REQUEST".to_string(),
                "document_ids must not be empty".to_string(),
            )),
        )
            .into_response();
    }

    debug!("HTTP delete request: {} documents", request.document_ids.len());

    let ipc_request = Request::DeleteDocuments {
        document_ids: request.document_ids,
    };

    match state.handler.handle(ipc_request).await {
        IpcResponse::DeleteComplete {
            documents_deleted,
            chunks_deleted,
        } => {
            let duration_ms = start.elapsed().as_millis() as u64;
            (
                StatusCode::OK,
                Json(DeleteResponse {
                    documents_deleted,
                    chunks_deleted,
                    duration_ms,
                }),
            )
                .into_response()
        }
        IpcResponse::Error { code, message } => {
            error!("Delete failed: {:?} - {}", code, message);
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

/// Clear all index entries endpoint
pub async fn clear_index(State(state): State<AppState>) -> impl IntoResponse {
    let start = Instant::now();

    debug!("HTTP clear index request");

    match state.handler.handle(Request::ClearIndex).await {
        IpcResponse::ClearComplete { chunks_deleted } => {
            let duration_ms = start.elapsed().as_millis() as u64;
            (
                StatusCode::OK,
                Json(ClearResponse {
                    chunks_deleted,
                    duration_ms,
                }),
            )
                .into_response()
        }
        IpcResponse::Error { code, message } => {
            error!("Clear failed: {:?} - {}", code, message);
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
