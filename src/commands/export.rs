use anyhow::Result;
use dindex::{
    config::Config,
    index::ChunkStorage,
};
use std::path::PathBuf;
use tracing::info;

pub async fn export_index(config: Config, output: PathBuf, format: String) -> Result<()> {
    info!("Exporting index to: {}", output.display());

    let chunk_storage = ChunkStorage::load(&config.node.data_dir)?;
    let chunks = chunk_storage.all_embeddings();
    let chunk_count = chunks.len();

    match format.as_str() {
        "jsonl" => {
            use std::io::Write;
            let mut file = std::fs::File::create(&output)?;

            for (chunk_id, _embedding) in &chunks {
                if let Some(stored) = chunk_storage.get(chunk_id) {
                    let json = serde_json::to_string(&stored.chunk)?;
                    writeln!(file, "{}", json)?;
                }
            }
        }
        "json" => {
            let all_chunks: Vec<_> = chunks
                .iter()
                .filter_map(|(id, _)| chunk_storage.get(id).map(|s| s.chunk))
                .collect();
            let json = serde_json::to_string_pretty(&all_chunks)?;
            std::fs::write(&output, json)?;
        }
        _ => {
            anyhow::bail!("Unknown format: {}", format);
        }
    }

    println!("Exported {} chunks to {}", chunk_count, output.display());

    Ok(())
}
