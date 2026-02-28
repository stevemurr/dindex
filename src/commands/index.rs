use anyhow::{Context, Result};
use dindex::{
    chunking::TextSplitter,
    client::{self, ClientError},
    config::Config,
    content::{extract_from_bytes_with_url, extract_from_path, ContentType},
    embedding::init_embedding_engine,
    index::IndexStack,
    types::Document,
};
use std::path::PathBuf;
use std::time::Duration;
use tracing::info;
use walkdir;

pub async fn index_document(
    config: Config,
    path: PathBuf,
    title: Option<String>,
    url: Option<String>,
) -> Result<()> {
    let path_str = path.to_string_lossy();

    // Check if path is a URL
    let (content, extracted_title, source_url) = if path_str.starts_with("http://")
        || path_str.starts_with("https://")
    {
        info!("Fetching URL: {}", path_str);

        // Fetch content from URL
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("dindex/0.1")
            .build()?;

        let response = client.get(path_str.as_ref()).send().await?;

        // Get content type from headers or URL
        let content_type_header = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        let content_type = if !content_type_header.is_empty() {
            ContentType::from_mime(content_type_header)
        } else {
            ContentType::from_url(&path_str)
        };

        let bytes = response.bytes().await?;
        info!(
            "Fetched {} bytes, content type: {:?}",
            bytes.len(),
            content_type
        );

        let extracted = extract_from_bytes_with_url(&bytes, content_type, Some(&path_str))?;
        (extracted.content, extracted.title, Some(path_str.to_string()))
    } else if path.is_file() {
        info!("Indexing file: {}", path.display());

        let extracted = extract_from_path(&path)?;
        (extracted.content, extracted.title, url)
    } else if path.is_dir() {
        info!("Indexing directory: {}", path.display());

        // Index all files in directory
        let mut combined = String::new();
        for entry in walkdir::WalkDir::new(&path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            match extract_from_path(entry.path()) {
                Ok(extracted) => {
                    combined.push_str(&extracted.content);
                    combined.push_str("\n\n");
                }
                Err(e) => {
                    tracing::warn!("Failed to extract {}: {}", entry.path().display(), e);
                }
            }
        }
        (combined, None, url)
    } else {
        anyhow::bail!("Path does not exist: {}", path.display());
    };

    // Create document
    let mut doc = Document::new(content);
    if let Some(t) = title {
        doc = doc.with_title(t);
    } else if let Some(t) = extracted_title {
        doc = doc.with_title(t);
    } else {
        doc = doc.with_title(path.file_name().unwrap_or_default().to_string_lossy());
    }
    if let Some(u) = source_url {
        doc = doc.with_url(u);
    }

    // Split into chunks
    let splitter = TextSplitter::new(config.chunking.clone());
    let chunks = splitter.split_document(&doc);

    info!("Created {} chunks", chunks.len());

    // Try daemon first
    match client::index_chunks(chunks.clone()).await {
        Ok(stats) => {
            info!(
                "Indexed {} chunks via daemon in {}ms",
                stats.chunks_indexed, stats.duration_ms
            );
            return Ok(());
        }
        Err(ClientError::DaemonNotRunning) => {
            // Fallback to direct access
            tracing::debug!("Daemon not running, using direct access");
        }
        Err(e) => {
            tracing::warn!("Daemon indexing failed: {}, falling back to direct access", e);
        }
    }

    // Direct access fallback
    info!("Using direct index access (daemon not available)");

    // Initialize embedding engine
    let engine = init_embedding_engine(&config)
        .context("Failed to initialize embedding engine")?;

    let stack = IndexStack::create(&config.node.data_dir, config.embedding.dimensions, &config.index)?;
    let indexer = stack.indexer();

    // Extract texts for batch embedding
    let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

    // Generate real embeddings using the embedding engine
    let embeddings = engine
        .embed_batch(&texts)
        .context("Failed to generate embeddings")?;

    // Pair chunks with embeddings
    let chunks_with_embeddings: Vec<_> = chunks
        .into_iter()
        .zip(embeddings.into_iter())
        .collect();

    // Index chunks
    let keys = indexer.index_batch(&chunks_with_embeddings)?;
    info!("Indexed {} chunks with keys {:?}", keys.len(), &keys[..keys.len().min(5)]);

    // Save index
    stack.save_vector_index(&config.node.data_dir)?;
    indexer.save()?;

    info!("Index saved to {}", config.node.data_dir.display());

    Ok(())
}
