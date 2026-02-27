use anyhow::{Context, Result};
use dindex::{
    client::{self, ClientError},
    config::Config,
    daemon::OutputFormat,
    embedding::init_embedding_engine,
    index::ChunkStorage,
    retrieval::{Bm25Index, HybridRetriever},
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
            output_search_results(&results, &format, top_k);
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

    let index_path = config.node.data_dir.join("vector.index");
    let vector_index = if index_path.exists() {
        Arc::new(dindex::index::VectorIndex::load(&index_path, &config.index)?)
    } else {
        Arc::new(dindex::index::VectorIndex::new(config.embedding.dimensions, &config.index)?)
    };

    let bm25_path = config.node.data_dir.join("bm25");
    let bm25_index = Arc::new(Bm25Index::new(&bm25_path)?);

    let chunk_storage = Arc::new(ChunkStorage::load(&config.node.data_dir)?);

    let retriever = Arc::new(HybridRetriever::new(
        vector_index,
        bm25_index,
        chunk_storage,
        None,
        config.retrieval.clone(),
    ));

    // Create query
    let query = Query::new(&query_text, top_k);

    // Generate query embedding using the embedding engine
    let query_embedding = engine
        .embed(&query_text)
        .context("Failed to embed query")?;

    // Execute search
    let results = retriever.search(&query, Some(&query_embedding))?;

    output_search_results(&results, &format, top_k);
    Ok(())
}

/// Output search results in the requested format, grouped by document
fn output_search_results(results: &[dindex::types::SearchResult], format: &str, top_k: usize) {
    let grouped = GroupedSearchResult::from_results(results.to_vec(), top_k);
    let total_chunks: usize = grouped.iter().map(|g| g.chunks.len()).sum();

    match format {
        "json" | "json-pretty" => {
            let json = serde_json::to_string_pretty(&grouped).unwrap_or_default();
            println!("{}", json);
        }
        _ => {
            println!(
                "\nSearch Results ({} documents, {} chunks):\n",
                grouped.len(),
                total_chunks,
            );
            for (i, group) in grouped.iter().enumerate() {
                println!(
                    "{}. [Score: {:.4}] {}",
                    i + 1,
                    group.relevance_score,
                    group.source_title.as_deref().unwrap_or("(untitled)"),
                );
                if let Some(url) = &group.source_url {
                    println!("   URL: {}", url);
                }
                println!("   Document: {}", group.document_id);
                println!("   Chunks ({}):", group.chunks.len());
                for (j, chunk) in group.chunks.iter().enumerate() {
                    println!(
                        "     {}. [Score: {:.4}] {}...",
                        j + 1,
                        chunk.relevance_score,
                        truncate_for_display(&chunk.content, 200),
                    );
                    if !chunk.matched_by.is_empty() {
                        println!("        Matched by: {:?}", chunk.matched_by);
                    }
                }
                println!();
            }
        }
    }
}

