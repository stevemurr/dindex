use anyhow::{Context, Result};
use dindex::{
    client,
    config::Config,
    daemon,
};
use std::time::Duration;

use crate::DevAction;

/// Handle dev commands
pub async fn handle_dev(config: Config, action: DevAction) -> Result<()> {
    match action {
        DevAction::ResetIndex { force } => {
            reset_index(config, force).await
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
