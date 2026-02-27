//! Model management utilities
//!
//! Note: With embed_anything, model downloading is handled automatically
//! by the HuggingFace hub. This module provides metadata and compatibility.

/// Model registry with known models and their metadata
pub struct ModelRegistry;

/// Information about a known model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Short model name
    pub name: &'static str,
    /// HuggingFace model ID
    pub huggingface_id: &'static str,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Maximum sequence length in tokens
    pub max_sequence_length: usize,
    /// Whether the model supports Matryoshka (variable dimension) embeddings
    pub supports_matryoshka: bool,
}

/// All known models as a static data table
const MODELS: &[ModelInfo] = &[
    // BGE models - English optimized
    ModelInfo {
        name: "bge-large-en-v1.5",
        huggingface_id: "BAAI/bge-large-en-v1.5",
        dimensions: 1024,
        max_sequence_length: 512,
        supports_matryoshka: false,
    },
    ModelInfo {
        name: "bge-base-en-v1.5",
        huggingface_id: "BAAI/bge-base-en-v1.5",
        dimensions: 768,
        max_sequence_length: 512,
        supports_matryoshka: false,
    },
    ModelInfo {
        name: "bge-small-en-v1.5",
        huggingface_id: "BAAI/bge-small-en-v1.5",
        dimensions: 384,
        max_sequence_length: 512,
        supports_matryoshka: false,
    },
    // E5 models
    ModelInfo {
        name: "e5-large-v2",
        huggingface_id: "intfloat/e5-large-v2",
        dimensions: 1024,
        max_sequence_length: 512,
        supports_matryoshka: false,
    },
    ModelInfo {
        name: "e5-base-v2",
        huggingface_id: "intfloat/e5-base-v2",
        dimensions: 768,
        max_sequence_length: 512,
        supports_matryoshka: false,
    },
    ModelInfo {
        name: "e5-small-v2",
        huggingface_id: "intfloat/e5-small-v2",
        dimensions: 384,
        max_sequence_length: 512,
        supports_matryoshka: false,
    },
    // Sentence Transformers
    ModelInfo {
        name: "all-MiniLM-L6-v2",
        huggingface_id: "sentence-transformers/all-MiniLM-L6-v2",
        dimensions: 384,
        max_sequence_length: 256,
        supports_matryoshka: false,
    },
    ModelInfo {
        name: "all-MiniLM-L12-v2",
        huggingface_id: "sentence-transformers/all-MiniLM-L12-v2",
        dimensions: 384,
        max_sequence_length: 256,
        supports_matryoshka: false,
    },
    // Jina models
    ModelInfo {
        name: "jina-embeddings-v2-small-en",
        huggingface_id: "jinaai/jina-embeddings-v2-small-en",
        dimensions: 512,
        max_sequence_length: 8192,
        supports_matryoshka: false,
    },
    ModelInfo {
        name: "jina-embeddings-v2-base-en",
        huggingface_id: "jinaai/jina-embeddings-v2-base-en",
        dimensions: 768,
        max_sequence_length: 8192,
        supports_matryoshka: false,
    },
    // Legacy nomic model
    ModelInfo {
        name: "nomic-embed-text-v1.5",
        huggingface_id: "nomic-ai/nomic-embed-text-v1.5",
        dimensions: 768,
        max_sequence_length: 8192,
        supports_matryoshka: true,
    },
];

impl ModelRegistry {
    /// Get info for a known model by short name
    pub fn get(model_name: &str) -> Option<&'static ModelInfo> {
        MODELS.iter().find(|m| m.name == model_name)
    }

    /// Check if a model name is valid (either a known model or a direct HuggingFace ID)
    pub fn is_valid(model_name: &str) -> bool {
        Self::get(model_name).is_some() || model_name.contains('/')
    }

    /// List all known models
    pub fn list_models() -> &'static [ModelInfo] {
        MODELS
    }

    /// List all known model names (for backward compatibility)
    pub fn list() -> Vec<&'static str> {
        vec![
            "all-MiniLM-L6-v2",  // Default - fast and reliable
            "all-MiniLM-L12-v2",
            "bge-base-en-v1.5",
            "bge-large-en-v1.5",
            "bge-small-en-v1.5",
            "e5-base-v2",
            "e5-large-v2",
            "e5-small-v2",
            // Note: bge-m3 not supported (uses XLMRobertaModel)
            // Note: jina models may have compatibility issues
        ]
    }
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
    let hf_id = model_info.huggingface_id;

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
            } else if name == "all-MiniLM-L6-v2" {
                "Default, fast"
            } else {
                ""
            };
            println!(
                "{:<30} {:>6} {:>8} {}",
                name, info.dimensions, info.max_sequence_length, notes
            );
        }
    }

    println!("\nDefault: all-MiniLM-L6-v2 (384 dimensions, fast, English)");
    println!("\nYou can also use any HuggingFace model ID with BertModel architecture, e.g.:");
    println!("  model_name = \"BAAI/bge-base-en-v1.5\"");
    println!("\nNote: bge-m3 is NOT supported (uses XLMRobertaModel architecture)");
}
