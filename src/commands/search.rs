use anyhow::{Context, Result};
use dindex::{
    client::{self, ClientError},
    config::Config,
    daemon::OutputFormat,
    embedding::init_embedding_engine,
    index::IndexStack,
    types::{GroupedSearchResult, Query},
    util::truncate_for_display,
};
use std::sync::Arc;
use tracing::info;

pub async fn search_index(
    config: Config,
    query_text: String,
    top_k: usize,
    format: String,
) -> Result<()> {
    info!("Searching for: {}", query_text);

    // Convert format string to OutputFormat
    let output_format = match format.as_str() {
        "json" => OutputFormat::Json,
        "json-pretty" => OutputFormat::JsonPretty,
        _ => OutputFormat::Text,
    };

    // Try daemon first
    match client::search(&query_text, top_k, output_format.clone()).await {
        Ok(results) => {
            output_search_results(&results, &format, top_k, &query_text);
            return Ok(());
        }
        Err(ClientError::DaemonNotRunning) => {
            // Fallback to direct access
            tracing::debug!("Daemon not running, using direct access");
        }
        Err(e) => {
            tracing::warn!("Daemon search failed: {}, falling back to direct access", e);
        }
    }

    // Direct access fallback
    // Initialize embedding engine
    let engine = init_embedding_engine(&config)
        .context("Failed to initialize embedding engine")?;

    let stack = IndexStack::open(&config)?;
    let retriever = Arc::new(stack.retriever(None, config.retrieval.clone()));

    // Create query
    let query = Query::new(&query_text, top_k);

    // Generate query embedding using the embedding engine
    let query_embedding = engine
        .embed(&query_text)
        .context("Failed to embed query")?;

    // Execute search
    let results = retriever.search(&query, Some(&query_embedding))?;

    output_search_results(&results, &format, top_k, &query_text);
    Ok(())
}

/// Output search results in the requested format, grouped by document
fn output_search_results(results: &[dindex::types::SearchResult], format: &str, top_k: usize, query: &str) {
    let mut grouped = GroupedSearchResult::from_results(results.to_vec(), top_k);

    // Generate snippets for each chunk
    for group in &mut grouped {
        for chunk in &mut group.chunks {
            chunk.snippet = dindex::retrieval::extract_snippet(query, &chunk.content, 200);
        }
    }

    let total_chunks: usize = grouped.iter().map(|g| g.chunks.len()).sum();

    match format {
        "json" | "json-pretty" => {
            // Build a response with citations
            let citations: Vec<serde_json::Value> = grouped.iter().map(|g| {
                let top_snippet = g.chunks.first().and_then(|c| c.snippet.clone());
                serde_json::json!({
                    "index": g.citation_index,
                    "source_title": g.source_title,
                    "source_url": g.source_url,
                    "snippet": top_snippet,
                })
            }).collect();

            let response = serde_json::json!({
                "results": grouped,
                "citations": citations,
                "total_documents": grouped.len(),
                "total_chunks": total_chunks,
            });

            match serde_json::to_string_pretty(&response) {
                Ok(json) => println!("{}", json),
                Err(e) => eprintln!("Failed to serialize results: {}", e),
            }
        }
        _ => {
            println!(
                "\nSearch Results ({} documents, {} chunks):\n",
                grouped.len(),
                total_chunks,
            );
            for group in &grouped {
                println!(
                    "[{}] [Score: {:.4}] {}",
                    group.citation_index,
                    group.relevance_score,
                    group.source_title.as_deref().unwrap_or("(untitled)"),
                );
                if let Some(url) = &group.source_url {
                    println!("   URL: {}", url);
                }
                println!("   Document: {}", group.document_id);
                println!("   Chunks ({}):", group.chunks.len());
                for (j, chunk) in group.chunks.iter().enumerate() {
                    // Prefer snippet over truncated content
                    let display_text = chunk.snippet.as_deref()
                        .unwrap_or_else(|| &chunk.content);
                    println!(
                        "     {}. [Score: {:.4}] {}",
                        j + 1,
                        chunk.relevance_score,
                        truncate_for_display(display_text, 200),
                    );
                    if !chunk.matched_by.is_empty() {
                        let methods: Vec<&str> = chunk.matched_by.iter().map(|m| m.as_str()).collect();
                        println!("        Matched by: {:?}", methods);
                    }
                }
                println!();
            }
        }
    }
}

