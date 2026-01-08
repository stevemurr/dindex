//! Model management utilities

use crate::config::EmbeddingConfig;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

/// Model registry with known models and their URLs
pub struct ModelRegistry;

/// Information about a known model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: &'static str,
    pub model_url: &'static str,
    pub tokenizer_url: &'static str,
    pub dimensions: usize,
    pub max_sequence_length: usize,
    pub supports_matryoshka: bool,
}

impl ModelRegistry {
    /// Get info for a known model
    pub fn get(model_name: &str) -> Option<ModelInfo> {
        match model_name {
            "nomic-embed-text-v1.5" => Some(ModelInfo {
                name: "nomic-embed-text-v1.5",
                model_url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx",
                tokenizer_url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json",
                dimensions: 768,
                max_sequence_length: 8192,
                supports_matryoshka: true,
            }),
            "e5-small-v2" => Some(ModelInfo {
                name: "e5-small-v2",
                model_url: "https://huggingface.co/intfloat/e5-small-v2/resolve/main/onnx/model.onnx",
                tokenizer_url: "https://huggingface.co/intfloat/e5-small-v2/resolve/main/tokenizer.json",
                dimensions: 384,
                max_sequence_length: 512,
                supports_matryoshka: false,
            }),
            "bge-base-en-v1.5" => Some(ModelInfo {
                name: "bge-base-en-v1.5",
                model_url: "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx",
                tokenizer_url: "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json",
                dimensions: 768,
                max_sequence_length: 512,
                supports_matryoshka: false,
            }),
            "all-MiniLM-L6-v2" => Some(ModelInfo {
                name: "all-MiniLM-L6-v2",
                model_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
                tokenizer_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
                dimensions: 384,
                max_sequence_length: 256,
                supports_matryoshka: false,
            }),
            _ => None,
        }
    }

    /// List all known models
    pub fn list() -> Vec<&'static str> {
        vec![
            "nomic-embed-text-v1.5",
            "e5-small-v2",
            "bge-base-en-v1.5",
            "all-MiniLM-L6-v2",
        ]
    }
}

/// Model manager for downloading and caching models
pub struct ModelManager {
    cache_dir: PathBuf,
    client: Client,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(cache_dir: impl AsRef<Path>) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)?;

        Ok(Self {
            cache_dir,
            client: Client::new(),
        })
    }

    /// Get paths for a model, downloading if necessary
    pub async fn ensure_model(&self, model_name: &str) -> Result<(PathBuf, PathBuf)> {
        let model_info = ModelRegistry::get(model_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown model: {}", model_name))?;

        let model_dir = self.cache_dir.join(model_name);
        fs::create_dir_all(&model_dir).await?;

        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Download model if not present
        if !model_path.exists() {
            info!("Downloading model: {}", model_name);
            self.download_file(model_info.model_url, &model_path).await?;
        }

        // Download tokenizer if not present
        if !tokenizer_path.exists() {
            info!("Downloading tokenizer for: {}", model_name);
            self.download_file(model_info.tokenizer_url, &tokenizer_path).await?;
        }

        Ok((model_path, tokenizer_path))
    }

    /// Download a file with progress bar
    async fn download_file(&self, url: &str, path: &Path) -> Result<()> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .context("Failed to start download")?;

        let total_size = response.content_length().unwrap_or(0);

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut file = fs::File::create(path).await?;
        let mut downloaded: u64 = 0;

        let mut stream = response.bytes_stream();
        use futures::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error downloading chunk")?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            pb.set_position(downloaded);
        }

        pb.finish_with_message("Download complete");
        Ok(())
    }

    /// Create an embedding config with downloaded model paths
    pub async fn create_config(&self, model_name: &str) -> Result<EmbeddingConfig> {
        let (model_path, tokenizer_path) = self.ensure_model(model_name).await?;
        let model_info = ModelRegistry::get(model_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown model: {}", model_name))?;

        Ok(EmbeddingConfig {
            model_name: model_name.to_string(),
            model_path: Some(model_path),
            tokenizer_path: Some(tokenizer_path),
            dimensions: model_info.dimensions,
            truncated_dimensions: if model_info.supports_matryoshka {
                256
            } else {
                model_info.dimensions
            },
            max_sequence_length: model_info.max_sequence_length,
            ..Default::default()
        })
    }
}
