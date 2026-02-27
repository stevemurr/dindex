use anyhow::Result;
use dindex::{
    client,
    config::Config,
    index::{ChunkStorage, DocumentRegistry},
    types::DocumentIdentity,
};
use tracing::info;

pub async fn show_stats(config: Config) -> Result<()> {
    info!("Loading index statistics...");

    // Try daemon first
    if let Ok(stats) = client::stats().await {
        println!("\nDIndex Statistics (via daemon):");
        println!("================================");
        println!("Data directory: {}", config.node.data_dir.display());
        println!("Total documents: {}", stats.total_documents);
        println!("Total chunks: {}", stats.total_chunks);
        println!("Vector index size: {} bytes", stats.vector_index_size_bytes);
        println!("BM25 index size: {} bytes", stats.bm25_index_size_bytes);
        println!("Storage size: {} bytes", stats.storage_size_bytes);
        println!("Embedding dimensions: {}", config.embedding.dimensions);
        println!("Model: {}", config.embedding.model_name);
        return Ok(());
    }

    // Fallback to direct access
    let chunk_storage = ChunkStorage::load(&config.node.data_dir)?;

    println!("\nDIndex Statistics:");
    println!("==================");
    println!("Data directory: {}", config.node.data_dir.display());
    println!("Total chunks: {}", chunk_storage.len());
    println!("Embedding dimensions: {}", config.embedding.dimensions);
    println!("Model: {}", config.embedding.model_name);

    // Check index files
    let index_path = config.node.data_dir.join("vector.index");
    if index_path.exists() {
        let metadata = std::fs::metadata(&index_path)?;
        println!("Vector index size: {} bytes", metadata.len());
    }

    Ok(())
}

pub async fn show_registry_stats(config: Config) -> Result<()> {
    // Try to load the registry
    let registry = match DocumentRegistry::load(
        &config.node.data_dir,
        config.dedup.simhash_distance_threshold,
    ) {
        Ok(r) => r,
        Err(e) => {
            println!("No document registry found: {}", e);
            println!("\nTo create a registry, import documents with:");
            println!("  dindex import <dump-file>");
            println!("Or migrate existing documents with:");
            println!("  dindex migrate-registry");
            return Ok(());
        }
    };

    let stats = registry.stats();

    println!("\nDocument Registry Statistics");
    println!("============================");
    println!("Total documents:     {}", stats.total_documents);
    println!("Total chunks:        {}", stats.total_chunks);
    println!("Total URLs:          {}", stats.total_urls);
    println!("SimHash buckets:     {}", stats.buckets_used);

    if !stats.source_counts.is_empty() {
        println!("\nDocuments by source:");
        for (source, count) in &stats.source_counts {
            println!("  {}: {}", source, count);
        }
    }

    println!("\nConfiguration:");
    println!("  Dedup enabled:     {}", config.dedup.enabled);
    println!("  SimHash threshold: {}", config.dedup.simhash_distance_threshold);
    println!("  Data directory:    {}", config.node.data_dir.display());

    Ok(())
}

pub async fn migrate_registry(config: Config, dry_run: bool, skip_existing: bool) -> Result<()> {
    info!("Migrating existing documents to unified registry...");

    // Load or create the registry
    let registry = DocumentRegistry::load(
        &config.node.data_dir,
        config.dedup.simhash_distance_threshold,
    )?;

    // Load chunk storage
    let chunk_storage = ChunkStorage::load(&config.node.data_dir)?;

    // Get all existing chunks
    let chunks = chunk_storage.chunk_ids();
    if chunks.is_empty() {
        println!("No existing chunks found to migrate.");
        return Ok(());
    }

    println!("Found {} existing chunks", chunks.len());

    // Group chunks by document ID
    let mut doc_chunks: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for chunk_id in &chunks {
        if let Some(stored) = chunk_storage.get(chunk_id) {
            let doc_id = &stored.chunk.metadata.document_id;
            doc_chunks
                .entry(doc_id.clone())
                .or_default()
                .push(chunk_id.clone());
        }
    }

    println!("Found {} unique documents", doc_chunks.len());

    if dry_run {
        println!("\nDry run - no changes will be made");
        println!("==================================");
    }

    let mut migrated = 0;
    let mut skipped = 0;
    let mut duplicates = 0;

    for (doc_id, chunk_ids) in &doc_chunks {
        // Get content from first chunk to compute identity
        let first_chunk = chunk_storage.get(&chunk_ids[0]);
        let Some(stored) = first_chunk else {
            continue;
        };

        // Collect full content from all chunks
        let mut content = String::new();
        for cid in chunk_ids {
            if let Some(sc) = chunk_storage.get(cid) {
                content.push_str(&sc.chunk.content);
                content.push(' ');
            }
        }

        // Compute identity
        let identity = DocumentIdentity::compute(&content);

        // Check if already in registry
        if skip_existing {
            if let Some(_existing) = registry.get(&identity.content_id) {
                skipped += 1;
                continue;
            }
        }

        // Check for duplicates
        let dup_check = registry.check_duplicate(&identity);
        match dup_check {
            dindex::index::DuplicateCheckResult::ExactMatch { entry } => {
                if !dry_run && !skip_existing {
                    // Update existing entry with this doc's chunks
                    registry.update_metadata(
                        &entry.content_id,
                        stored.chunk.metadata.source_url.clone(),
                        None,
                    );
                }
                duplicates += 1;
                continue;
            }
            dindex::index::DuplicateCheckResult::NearDuplicate { entry, .. } => {
                if !dry_run && !skip_existing {
                    registry.update_metadata(
                        &entry.content_id,
                        stored.chunk.metadata.source_url.clone(),
                        None,
                    );
                }
                duplicates += 1;
                continue;
            }
            dindex::index::DuplicateCheckResult::New => {
                // Register new document
                if !dry_run {
                    registry.register(
                        identity,
                        stored.chunk.metadata.source_title.clone(),
                        stored.chunk.metadata.source_url.clone(),
                        "migrated",
                        Some(("original_doc_id", doc_id.as_str())),
                        chunk_ids.clone(),
                    );
                }
                migrated += 1;
            }
        }
    }

    // Save registry
    if !dry_run {
        registry.save()?;
        println!("\nRegistry saved to {}", config.node.data_dir.display());
    }

    println!("\nMigration Summary");
    println!("=================");
    println!("Documents migrated:  {}", migrated);
    println!("Documents skipped:   {}", skipped);
    println!("Duplicates found:    {}", duplicates);
    println!("Total processed:     {}", doc_chunks.len());

    Ok(())
}
