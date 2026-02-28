//! Job handlers: scrape, progress, cancel, SSE events

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use std::convert::Infallible;
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt as _;
use tracing::{debug, error, info, warn};

use super::AppState;
use super::super::types::*;
use crate::daemon::protocol::{ProgressStage, Request, Response as IpcResponse};

/// Parse a job ID string into a UUID, returning an error response on failure.
fn parse_job_id(job_id: &str) -> Result<uuid::Uuid, Response> {
    uuid::Uuid::parse_str(job_id).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "INVALID_JOB_ID".to_string(),
                "Invalid job ID format".to_string(),
            )),
        )
            .into_response()
    })
}

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
    let uuid = match parse_job_id(&job_id) {
        Ok(id) => id,
        Err(resp) => return resp,
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
    let uuid = match parse_job_id(&job_id) {
        Ok(id) => id,
        Err(resp) => return resp,
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

/// SSE endpoint for real-time scrape job events
pub async fn job_events_sse(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let uuid = match parse_job_id(&job_id) {
        Ok(id) => id,
        Err(resp) => return resp,
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
