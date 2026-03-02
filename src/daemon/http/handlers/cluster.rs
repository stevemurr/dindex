//! Cluster handler

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use tracing::{debug, error};

use super::AppState;
use crate::daemon::http::types::*;
use crate::daemon::protocol::{Request, Response as IpcResponse};

/// Maximum number of URLs allowed in a single cluster request
const MAX_CLUSTER_URLS: usize = 1000;

/// Cluster documents endpoint
pub async fn cluster_documents(
    State(state): State<AppState>,
    Json(request): Json<ClusterRequest>,
) -> impl IntoResponse {
    // Validate: non-empty URL list
    if request.document_urls.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "EMPTY_URLS",
                "document_urls must not be empty",
            )),
        )
            .into_response();
    }

    // Validate: max URL count
    if request.document_urls.len() > MAX_CLUSTER_URLS {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "TOO_MANY_URLS",
                format!(
                    "document_urls length {} exceeds maximum of {}",
                    request.document_urls.len(),
                    MAX_CLUSTER_URLS
                ),
            )),
        )
            .into_response();
    }

    // Validate: max_clusters >= 1
    if request.max_clusters < 1 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "INVALID_MAX_CLUSTERS",
                "max_clusters must be at least 1",
            )),
        )
            .into_response();
    }

    debug!(
        "HTTP cluster request: {} URLs, max_clusters={}",
        request.document_urls.len(),
        request.max_clusters
    );

    let ipc_request = Request::ClusterDocuments {
        document_urls: request.document_urls,
        max_clusters: request.max_clusters,
    };

    match state.handler.handle(ipc_request).await {
        IpcResponse::ClusterResults {
            clusters,
            documents,
            unmatched_urls,
            cluster_time_ms,
        } => {
            let cluster_json: Vec<ClusterJson> = clusters
                .into_iter()
                .map(|c| ClusterJson {
                    cluster_id: c.cluster_id,
                    label: c.label,
                    document_urls: c.document_urls,
                })
                .collect();

            let doc_json = if request.include_summaries {
                documents
                    .into_iter()
                    .map(|(url, info)| {
                        (
                            url,
                            ClusterDocumentJson {
                                title: info.title,
                                snippet: info.snippet,
                            },
                        )
                    })
                    .collect()
            } else {
                std::collections::HashMap::new()
            };

            (
                StatusCode::OK,
                Json(ClusterResponse {
                    clusters: cluster_json,
                    documents: doc_json,
                    unmatched_urls,
                    cluster_time_ms,
                }),
            )
                .into_response()
        }
        IpcResponse::Error { code, message } => {
            error!("Cluster failed: {:?} - {}", code, message);
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
