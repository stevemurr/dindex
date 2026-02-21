//! Model management utilities
//!
//! Note: With embed_anything, model downloading is handled automatically
//! by the HuggingFace hub. This module provides metadata and compatibility.

use crate::config::EmbeddingConfig;
use anyhow::Result;
use tracing::info;

/// Model registry with known models and their metadata
pub struct ModelRegistry;

/// Information about a known model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Short model name
    pub name: String,
    /// HuggingFace model ID
    pub huggingface_id: String,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Maximum sequence length in tokens
    pub max_sequence_length: usize,
    /// Whether the model supports Matryoshka (variable dimension) embeddings
    pub supports_matryoshka: bool,
}

impl ModelRegistry {
    /// Get info for a known model
    pub fn get(model_name: &str) -> Option<ModelInfo> {
        match model_name {
            // BGE-M3 - Default model, excellent multilingual support
            "bge-m3" => Some(ModelInfo {
                name: "bge-m3".to_string(),
                huggingface_id: "BAAI/bge-m3".to_string(),
                dimensions: 1024,
                max_sequence_length: 8192,
                supports_matryoshka: true,
            }),
            // BGE models - English optimized
            "bge-large-en-v1.5" => Some(ModelInfo {
                name: "bge-large-en-v1.5".to_string(),
                huggingface_id: "BAAI/bge-large-en-v1.5".to_string(),
                dimensions: 1024,
                max_sequence_length: 512,
                supports_matryoshka: false,
            }),
            "bge-base-en-v1.5" => Some(ModelInfo {
                name: "bge-base-en-v1.5".to_string(),
                huggingface_id: "BAAI/bge-base-en-v1.5".to_string(),
                dimensions: 768,
                max_sequence_length: 512,
                supports_matryoshka: false,
            }),
            "bge-small-en-v1.5" => Some(ModelInfo {
                name: "bge-small-en-v1.5".to_string(),
                huggingface_id: "BAAI/bge-small-en-v1.5".to_string(),
                dimensions: 384,
                max_sequence_length: 512,
                supports_matryoshka: false,
            }),
            // E5 models
            "e5-large-v2" => Some(ModelInfo {
                name: "e5-large-v2".to_string(),
                huggingface_id: "intfloat/e5-large-v2".to_string(),
                dimensions: 1024,
                max_sequence_length: 512,
                supports_matryoshka: false,
            }),
            "e5-base-v2" => Some(ModelInfo {
                name: "e5-base-v2".to_string(),
                huggingface_id: "intfloat/e5-base-v2".to_string(),
                dimensions: 768,
                max_sequence_length: 512,
                supports_matryoshka: false,
            }),
            "e5-small-v2" => Some(ModelInfo {
                name: "e5-small-v2".to_string(),
                huggingface_id: "intfloat/e5-small-v2".to_string(),
                dimensions: 384,
                max_sequence_length: 512,
                supports_matryoshka: false,
            }),
            // Sentence Transformers
            "all-MiniLM-L6-v2" => Some(ModelInfo {
                name: "all-MiniLM-L6-v2".to_string(),
                huggingface_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                dimensions: 384,
                max_sequence_length: 256,
                supports_matryoshka: false,
            }),
            "all-MiniLM-L12-v2" => Some(ModelInfo {
                name: "all-MiniLM-L12-v2".to_string(),
                huggingface_id: "sentence-transformers/all-MiniLM-L12-v2".to_string(),
                dimensions: 384,
                max_sequence_length: 256,
                supports_matryoshka: false,
            }),
            // Jina models
            "jina-embeddings-v2-small-en" => Some(ModelInfo {
                name: "jina-embeddings-v2-small-en".to_string(),
                huggingface_id: "jinaai/jina-embeddings-v2-small-en".to_string(),
                dimensions: 512,
                max_sequence_length: 8192,
                supports_matryoshka: false,
            }),
            "jina-embeddings-v2-base-en" => Some(ModelInfo {
                name: "jina-embeddings-v2-base-en".to_string(),
                huggingface_id: "jinaai/jina-embeddings-v2-base-en".to_string(),
                dimensions: 768,
                max_sequence_length: 8192,
                supports_matryoshka: false,
            }),
            // Legacy nomic model
            "nomic-embed-text-v1.5" => Some(ModelInfo {
                name: "nomic-embed-text-v1.5".to_string(),
                huggingface_id: "nomic-ai/nomic-embed-text-v1.5".to_string(),
                dimensions: 768,
                max_sequence_length: 8192,
                supports_matryoshka: true,
            }),
            // Allow direct HuggingFace IDs
            name if name.contains('/') => {
                // Return a generic info for direct HF IDs
                // The actual dimensions will be determined at runtime
                Some(ModelInfo {
                    name: "custom".to_string(),
                    huggingface_id: name.to_string(),
                    dimensions: 768, // Default, will be overridden
                    max_sequence_length: 512,
                    supports_matryoshka: false,
                })
            }
            _ => None,
        }
    }

    /// List all known models
    pub fn list() -> Vec<&'static str> {
        vec![
            "bge-m3",
            "bge-large-en-v1.5",
            "bge-base-en-v1.5",
            "bge-small-en-v1.5",
            "e5-large-v2",
            "e5-base-v2",
            "e5-small-v2",
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "jina-embeddings-v2-small-en",
            "jina-embeddings-v2-base-en",
            "nomic-embed-text-v1.5",
        ]
    }

    /// Get recommended model for a use case
    pub fn recommended(use_case: &str) -> &'static str {
        match use_case {
            "multilingual" => "bge-m3",
            "english" => "bge-base-en-v1.5",
            "fast" | "small" => "all-MiniLM-L6-v2",
            "quality" | "best" => "bge-m3",
            "long-context" => "bge-m3",
            _ => "bge-m3",
        }
    }
}

/// Create an embedding config for a model
///
/// Note: With embed_anything, model downloading is automatic.
/// This function just creates the config with appropriate dimensions.
pub fn create_config(model_name: &str) -> Result<EmbeddingConfig> {
    let model_info = ModelRegistry::get(model_name)
        .ok_or_else(|| anyhow::anyhow!("Unknown model: {}", model_name))?;

    info!(
        "Creating config for model: {} ({} dims, {} max tokens)",
        model_name, model_info.dimensions, model_info.max_sequence_length
    );

    Ok(EmbeddingConfig {
        model_name: model_name.to_string(),
        model_path: None, // embed_anything handles this
        tokenizer_path: None, // embed_anything handles this
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

/// Check if a model is cached in HuggingFace hub
///
/// Returns (cache_path, is_complete) if found
/// A model is considered complete if it has model weights (safetensors or pytorch)
pub fn check_model_cached(model_name: &str) -> Option<std::path::PathBuf> {
    let (path, complete) = check_model_cached_detailed(model_name)?;
    if complete {
        Some(path)
    } else {
        None
    }
}

/// Detailed check for model cache status
///
/// Returns (cache_path, is_complete)
pub fn check_model_cached_detailed(model_name: &str) -> Option<(std::path::PathBuf, bool)> {
    let model_info = ModelRegistry::get(model_name)?;
    let hf_id = &model_info.huggingface_id;

    // HuggingFace cache structure: ~/.cache/huggingface/hub/models--{org}--{model}
    let home = std::env::var("HOME").ok()?;
    let cache_dir = std::path::PathBuf::from(home).join(".cache/huggingface/hub");

    // Convert "BAAI/bge-m3" to "models--BAAI--bge-m3"
    let model_dir_name = format!("models--{}", hf_id.replace('/', "--"));
    let model_path = cache_dir.join(&model_dir_name);

    if !model_path.exists() {
        return None;
    }

    // Check blobs directory for model weights
    let blobs_dir = model_path.join("blobs");
    let mut has_weights = false;

    if let Ok(entries) = std::fs::read_dir(&blobs_dir) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                let size = meta.len();
                // Model weights are typically > 100MB
                if size > 100_000_000 {
                    has_weights = true;
                    break;
                }
            }
        }
    }

    // Also check if snapshots exist
    let snapshots_dir = model_path.join("snapshots");
    let has_snapshots = snapshots_dir.exists()
        && snapshots_dir.read_dir().ok()?.next().is_some();

    if has_snapshots {
        Some((model_path, has_weights))
    } else {
        None
    }
}

/// Get the size of cached model files
pub fn get_cached_model_size(model_name: &str) -> Option<u64> {
    let cache_path = check_model_cached(model_name)?;

    fn dir_size(path: &std::path::Path) -> u64 {
        let mut size = 0;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    size += dir_size(&path);
                } else if let Ok(meta) = path.metadata() {
                    size += meta.len();
                }
            }
        }
        size
    }

    Some(dir_size(&cache_path))
}

/// Format bytes as human-readable size
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Print information about available models
pub fn print_models() {
    println!("Available embedding models:\n");
    println!("{:<30} {:>6} {:>8} {}", "Model", "Dims", "Max Seq", "Notes");
    println!("{}", "-".repeat(70));

    for name in ModelRegistry::list() {
        if let Some(info) = ModelRegistry::get(name) {
            let notes = if info.supports_matryoshka {
                "Matryoshka"
            } else if name.contains("bge-m3") {
                "Multilingual, recommended"
            } else {
                ""
            };
            println!(
                "{:<30} {:>6} {:>8} {}",
                name, info.dimensions, info.max_sequence_length, notes
            );
        }
    }

    println!("\nDefault: bge-m3 (1024 dimensions, multilingual, best quality)");
    println!("\nYou can also use any HuggingFace model ID directly, e.g.:");
    println!("  model_name = \"BAAI/bge-m3\"");
}
