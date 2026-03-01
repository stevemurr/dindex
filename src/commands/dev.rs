use anyhow::{Context, Result};
use dindex::{
    client,
    config::Config,
    daemon,
    index::ChunkStorage,
    retrieval::Bm25Index,
};
use std::time::Duration;

use crate::DevAction;

/// Handle dev commands
pub async fn handle_dev(config: Config, action: DevAction) -> Result<()> {
    match action {
        DevAction::ResetIndex { force } => {
            reset_index(config, force).await
        }
        DevAction::RebuildBm25 => {
            rebuild_bm25(config).await
        }
    }
}

/// Reset the local index by deleting all index files
async fn reset_index(config: Config, force: bool) -> Result<()> {
    let data_dir = &config.node.data_dir;

    if !force {
        println!("This will delete all indexed content in: {}", data_dir.display());
        println!("Are you sure? [y/N] ");

        use std::io::{self, BufRead};
        let stdin = io::stdin();
        let mut line = String::new();
        stdin.lock().read_line(&mut line)?;

        if !line.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    // Stop daemon if running
    if daemon::is_daemon_running(data_dir) {
        println!("Stopping daemon...");
        let _ = client::shutdown().await;
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    // Files and directories to delete for a full reset
    let items_to_delete = [
        // Index data
        "bm25",                         // BM25/Tantivy index directory
        "chunks.sled",                  // Chunk storage (sled database)
        "document_registry.sled",       // Document registry (sled database)
        "vector.index",                 // HNSW vector index
        "vector.index.mappings.json",   // Vector index key mappings
        // Legacy storage (for migration cleanup)
        "chunks.json",                  // Old chunk storage format
        // Daemon files
        "dindex.err",                   // Daemon error log
        "dindex.log",                   // Daemon log
        "dindex.pid",                   // Daemon PID file
    ];

    let mut deleted = 0;
    for item in &items_to_delete {
        let path = data_dir.join(item);
        if path.exists() {
            if path.is_dir() {
                std::fs::remove_dir_all(&path)
                    .with_context(|| format!("Failed to remove directory: {}", path.display()))?;
            } else {
                std::fs::remove_file(&path)
                    .with_context(|| format!("Failed to remove file: {}", path.display()))?;
            }
            println!("  Removed: {}", item);
            deleted += 1;
        }
    }

    if deleted == 0 {
        println!("No index files found to delete.");
    } else {
        println!("\nIndex reset complete. Removed {} items.", deleted);
    }

    Ok(())
}

/// Rebuild BM25 index from existing chunk storage
async fn rebuild_bm25(config: Config) -> Result<()> {
    let data_dir = &config.node.data_dir;

    // Open chunk storage (ChunkStorage::new joins "chunks.sled" internally)
    let chunks_sled = data_dir.join("chunks.sled");
    if !chunks_sled.exists() {
        anyhow::bail!("No chunk storage found at {}", chunks_sled.display());
    }
    let storage = ChunkStorage::new(data_dir)
        .context("Failed to open chunk storage")?;

    let chunk_ids = storage.chunk_ids();
    let total = chunk_ids.len();
    if total == 0 {
        println!("No chunks in storage. Nothing to rebuild.");
        return Ok(());
    }
    println!("Found {} chunks in storage", total);

    // Delete existing BM25 index and create fresh
    let bm25_path = data_dir.join("bm25");
    if bm25_path.exists() {
        std::fs::remove_dir_all(&bm25_path)
            .context("Failed to remove old BM25 index")?;
        println!("Removed old BM25 index");
    }

    let bm25 = Bm25Index::new(&bm25_path)
        .context("Failed to create BM25 index")?;

    // Add all chunks in batches
    let batch_size = 500;
    let mut added = 0;
    for batch_start in (0..total).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(total);
        let batch_ids = &chunk_ids[batch_start..batch_end];
        let stored = storage.get_batch(batch_ids);

        for s in &stored {
            bm25.add(&s.chunk)?;
            added += 1;
        }
        bm25.commit()?;
        print!("\r  Indexed {}/{} chunks", added, total);
    }
    println!();

    println!("BM25 rebuild complete: {} chunks indexed", added);
    Ok(())
}
