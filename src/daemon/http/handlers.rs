//! HTTP API Request Handlers
//!
//! Handlers that map HTTP requests to RequestHandler operations.

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Json,
};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as _;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use axum::extract::Path;

use crate::daemon::handler::RequestHandler;
use crate::daemon::protocol::{self, OutputFormat, ProgressStage, Request, Response as IpcResponse};
use crate::types::{Document, GroupedSearchResult};

use super::types::*;

/// Maximum allowed query length (10KB)
const MAX_QUERY_LENGTH: usize = 10_000;

/// Maximum allowed document content size (10MB)
const MAX_DOCUMENT_SIZE: usize = 10 * 1024 * 1024;

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

/// Search endpoint
pub async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> impl IntoResponse {
    // Validate query length
    if request.query.len() > MAX_QUERY_LENGTH {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "QUERY_TOO_LARGE".to_string(),
                format!(
                    "Query length {} exceeds maximum allowed length of {} bytes",
                    request.query.len(),
                    MAX_QUERY_LENGTH
                ),
            )),
        )
            .into_response();
    }

    debug!("HTTP search request: query={}, top_k={}", request.query, request.top_k);

    // Convert HTTP filters to QueryFilters
    let filters = request.filters.map(|f| crate::types::QueryFilters {
        source_url_prefix: f.source_url_prefix,
        metadata_equals: f.metadata_equals,
        metadata_contains: f.metadata_contains,
        ..Default::default()
    });

    let query_text = request.query.clone();
    let ipc_request = Request::Search {
        query: request.query,
        top_k: request.top_k,
        format: OutputFormat::Json,
        filters,
    };

    match state.handler.handle(ipc_request).await {
        IpcResponse::SearchResults { results, query_time_ms } => {
            let mut grouped = GroupedSearchResult::from_results(results, request.top_k);

            // Generate snippets for each chunk
            for group in &mut grouped {
                for chunk in &mut group.chunks {
                    chunk.snippet = crate::retrieval::extract_snippet(&query_text, &chunk.content, 200);
                }
            }

            let total_documents = grouped.len();
            let total_chunks: usize = grouped.iter().map(|g| g.chunks.len()).sum();

            let mut citations = Vec::with_capacity(grouped.len());
            let json_results: Vec<GroupedSearchResultJson> = grouped
                .into_iter()
                .map(|g| {
                    let top_snippet = g.chunks.first().and_then(|c| c.snippet.clone());
                    citations.push(Citation {
                        index: g.citation_index,
                        source_title: g.source_title.clone(),
                        source_url: g.source_url.clone(),
                        snippet: top_snippet,
                    });
                    GroupedSearchResultJson {
                        document_id: g.document_id,
                        source_url: g.source_url,
                        source_title: g.source_title,
                        relevance_score: g.relevance_score,
                        citation_index: g.citation_index,
                        chunks: g.chunks.into_iter().map(|c| MatchingChunkJson {
                            chunk_id: c.chunk_id,
                            content: c.content,
                            relevance_score: c.relevance_score,
                            matched_by: c.matched_by,
                            section_hierarchy: c.section_hierarchy,
                            position_in_doc: c.position_in_doc,
                            snippet: c.snippet,
                        }).collect(),
                    }
                })
                .collect();

            (
                StatusCode::OK,
                Json(SearchResponse {
                    results: json_results,
                    citations,
                    total_documents,
                    total_chunks,
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
    // Note: In a real implementation, we'd chunk the documents properly.
    // For now, we send each document as a single chunk.
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

// ============ Scrape Handlers ============

/// Start a web scrape job
pub async fn start_scrape(
    State(state): State<AppState>,
    Json(request): Json<ScrapeRequest>,
) -> impl IntoResponse {
    if request.urls.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "INVALID_REQUEST".to_string(),
                "At least one URL is required".to_string(),
            )),
        )
            .into_response();
    }

    debug!("HTTP scrape request: {} URLs, depth={}", request.urls.len(), request.options.max_depth);

    let ipc_request = Request::ScrapeStart {
        urls: request.urls,
        options: request.options,
    };

    match state.handler.handle(ipc_request).await {
        IpcResponse::JobStarted { job_id } => (
            StatusCode::OK,
            Json(JobStartedResponse {
                job_id: job_id.to_string(),
                message: "Scrape job started".to_string(),
            }),
        )
            .into_response(),
        IpcResponse::Error { code, message } => {
            error!("Scrape start failed: {:?} - {}", code, message);
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

/// Get job progress
pub async fn get_job_progress(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let uuid = match uuid::Uuid::parse_str(&job_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "INVALID_JOB_ID".to_string(),
                    "Invalid job ID format".to_string(),
                )),
            )
                .into_response();
        }
    };

    debug!("HTTP job progress request: {}", job_id);

    match state.handler.handle(Request::JobProgress { job_id: uuid }).await {
        IpcResponse::JobProgress { job_id, progress } => (
            StatusCode::OK,
            Json(JobProgressResponse {
                job_id: job_id.to_string(),
                stage: progress.stage,
                current: progress.current,
                total: progress.total,
                rate: progress.rate,
                eta_seconds: progress.eta_seconds,
            }),
        )
            .into_response(),
        IpcResponse::JobComplete { job_id, stats } => (
            StatusCode::OK,
            Json(JobProgressResponse {
                job_id: job_id.to_string(),
                stage: ProgressStage::Completed,
                current: stats.chunks_indexed as u64,
                total: Some(stats.chunks_indexed as u64),
                rate: None,
                eta_seconds: Some(0),
            }),
        )
            .into_response(),
        IpcResponse::JobFailed { job_id, error: _ } => (
            StatusCode::OK,
            Json(JobProgressResponse {
                job_id: job_id.to_string(),
                stage: ProgressStage::Failed,
                current: 0,
                total: None,
                rate: None,
                eta_seconds: None,
            }),
        )
            .into_response(),
        IpcResponse::Error { code, message } => {
            let status = if code == crate::daemon::protocol::ErrorCode::JobNotFound {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
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

/// Cancel a running job
pub async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let uuid = match uuid::Uuid::parse_str(&job_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "INVALID_JOB_ID".to_string(),
                    "Invalid job ID format".to_string(),
                )),
            )
                .into_response();
        }
    };

    debug!("HTTP job cancel request: {}", job_id);

    match state.handler.handle(Request::ScrapeCancel { job_id: uuid }).await {
        IpcResponse::Ok => (
            StatusCode::OK,
            Json(JobCancelResponse {
                success: true,
                message: "Job cancelled".to_string(),
            }),
        )
            .into_response(),
        IpcResponse::Error { code, message } => {
            let status = if code == crate::daemon::protocol::ErrorCode::JobNotFound {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(JobCancelResponse {
                    success: false,
                    message,
                }),
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

// ============ SSE Event Stream ============

/// SSE endpoint for real-time scrape job events
pub async fn job_events_sse(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let uuid = match uuid::Uuid::parse_str(&job_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "INVALID_JOB_ID".to_string(),
                    "Invalid job ID format".to_string(),
                )),
            )
                .into_response();
        }
    };

    let rx = match state.handler.subscribe_job_events(uuid) {
        Some(rx) => {
            info!("SSE client connected for job {}", job_id);
            rx
        }
        None => {
            warn!("SSE subscribe failed: job {} not found", job_id);
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse::new(
                    "JOB_NOT_FOUND".to_string(),
                    format!("Job {} not found or not a scrape job", job_id),
                )),
            )
                .into_response();
        }
    };

    let job_id_log = job_id.clone();
    let stream = BroadcastStream::new(rx)
        .filter_map(move |result| match result {
            Ok(event) => {
                let event_name = event.event_name().to_string();
                match serde_json::to_string(&event) {
                    Ok(json) => {
                        debug!("SSE streaming {} to client for job {}", event_name, job_id_log);
                        Some(Ok::<_, Infallible>(
                            Event::default().event(event_name).data(json),
                        ))
                    }
                    Err(e) => {
                        warn!("SSE serialization error for job {}: {}", job_id_log, e);
                        None
                    }
                }
            }
            Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                warn!("SSE client lagged for job {}: missed {} events", job_id_log, n);
                Some(Ok(
                    Event::default()
                        .event("lagged")
                        .data(format!(r#"{{"missed":{}}}"#, n)),
                ))
            }
        });

    Sse::new(stream)
        .keep_alive(KeepAlive::default().interval(Duration::from_secs(15)))
        .into_response()
}
