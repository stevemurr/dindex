use anyhow::{Context, Result};
use dindex::{
    client::{self, ClientError},
    config::Config,
    daemon::OutputFormat,
    embedding::init_embedding_engine,
    index::ChunkStorage,
    retrieval::{Bm25Index, HybridRetriever},
    types::Query,
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
            output_search_results(&results, &format);
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

    output_search_results(&results, &format);
    Ok(())
}

/// Output search results in the requested format
fn output_search_results(results: &[dindex::types::SearchResult], format: &str) {
    match format {
        "json" => {
            let json = serde_json::to_string_pretty(results).unwrap_or_default();
            println!("{}", json);
        }
        _ => {
            println!("\nSearch Results ({} found):\n", results.len());
            for (i, result) in results.iter().enumerate() {
                println!("{}. [Score: {:.4}]", i + 1, result.relevance_score);
                println!("   ID: {}", result.chunk.metadata.chunk_id);
                if let Some(title) = &result.chunk.metadata.source_title {
                    println!("   Title: {}", title);
                }
                println!("   Content: {}...", truncate_for_display(&result.chunk.content, 200));
                println!("   Matched by: {:?}", result.matched_by);
                println!();
            }
        }
    }
}

