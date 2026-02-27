use anyhow::{Context, Result};
use dindex::{
    config::Config,
    embedding::init_embedding_engine,
};
use tracing::info;

pub async fn download_model(mut config: Config, model: String) -> Result<()> {
    use dindex::embedding::model::ModelRegistry;

    info!("Downloading model: {}", model);

    // Validate model name
    if ModelRegistry::get(&model).is_none() {
        println!("Unknown model: {}", model);
        println!("\nAvailable models:");
        for name in ModelRegistry::list() {
            println!("  - {}", name);
        }
        println!("\nYou can also use any HuggingFace model ID directly (e.g., BAAI/bge-m3)");
        std::process::exit(1);
    }

    // Update config with the requested model
    config.embedding.model_name = model.clone();

    // Initialize the embedding engine - this triggers the download
    println!("Downloading model from HuggingFace Hub...");
    println!("(Models are cached in ~/.cache/huggingface/hub)");

    let _engine = init_embedding_engine(&config)
        .context("Failed to download/initialize model")?;

    println!("\nModel '{}' downloaded and ready!", model);

    Ok(())
}

pub async fn check_model(mut config: Config, model: String, download_if_missing: bool) -> Result<()> {
    use dindex::embedding::model::{check_model_cached_detailed, format_size, get_cached_model_size, ModelRegistry};

    // Validate model name
    if ModelRegistry::get(&model).is_none() {
        println!("Unknown model: {}", model);
        println!("\nAvailable models:");
        for name in ModelRegistry::list() {
            println!("  - {}", name);
        }
        println!("\nYou can also use any HuggingFace model ID directly (e.g., BAAI/bge-m3)");
        std::process::exit(1);
    }

    // Check if model is cached
    if let Some((cache_path, is_complete)) = check_model_cached_detailed(&model) {
        if is_complete {
            let size = get_cached_model_size(&model).unwrap_or(0);
            println!("Model '{}' is ready", model);
            println!("  Cache: {}", cache_path.display());
            println!("  Size:  {}", format_size(size));
            return Ok(());
        } else {
            println!("Model '{}' is partially downloaded (incomplete)", model);
            println!("  Cache: {}", cache_path.display());
            if !download_if_missing {
                println!("\nTo complete the download, run:");
                println!("  dindex check-model {}", model);
                println!("  # or");
                println!("  dindex download {}", model);
                std::process::exit(1);
            }
            println!("\nCompleting download...");
        }
    } else if !download_if_missing {
        println!("Model '{}' is NOT downloaded", model);
        println!("\nTo download, run:");
        println!("  dindex check-model {}", model);
        println!("  # or");
        println!("  dindex download {}", model);
        std::process::exit(1);
    } else {
        println!("Model '{}' not found in cache, downloading...", model);
    }

    // Download the model
    config.embedding.model_name = model.clone();

    println!("Downloading from HuggingFace Hub...");
    println!("(Models are cached in ~/.cache/huggingface/hub)");

    let _engine = init_embedding_engine(&config)
        .context("Failed to download/initialize model")?;

    let size = get_cached_model_size(&model).unwrap_or(0);
    println!("\nModel '{}' downloaded and ready!", model);
    println!("  Size: {}", format_size(size));

    Ok(())
}
