//! Search handler

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use tracing::{debug, error};

use super::{AppState, MAX_QUERY_LENGTH};
use crate::daemon::http::types::*;
use crate::daemon::protocol::{OutputFormat, Request, Response as IpcResponse};
use crate::types::GroupedSearchResult;

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
