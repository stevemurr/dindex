//! Configuration for DIndex

mod daemon;
mod embedding;
mod index;
mod logging;
mod network;
mod node;
mod scraping;

pub use daemon::{DaemonConfig, HttpConfig, MetricsConfig};
pub use embedding::{BackendConfig, EmbeddingConfig};
pub use index::{ChunkingConfig, IndexConfig, RetrievalConfig};
pub use logging::{LogFormat, LogLevel, LoggingConfig};
pub use network::RoutingConfig;
pub use node::NodeConfig;
pub use scraping::{BulkImportConfig, DedupConfig, ScrapingConfig};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Default user agent for all HTTP requests (scraping, politeness, fetching)
pub const DEFAULT_USER_AGENT: &str = "DecentralizedSearchBot/1.0 (+https://github.com/dindex)";

/// Main configuration for the DIndex node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Node configuration
    pub node: NodeConfig,
    /// Embedding model configuration
    pub embedding: EmbeddingConfig,
    /// Vector index configuration
    pub index: IndexConfig,
    /// Chunking configuration
    pub chunking: ChunkingConfig,
    /// Retrieval configuration
    pub retrieval: RetrievalConfig,
    /// Routing configuration
    pub routing: RoutingConfig,
    /// Scraping configuration
    #[serde(default)]
    pub scraping: ScrapingConfig,
    /// Bulk import configuration
    #[serde(default)]
    pub bulk_import: BulkImportConfig,
    /// Deduplication configuration
    #[serde(default)]
    pub dedup: DedupConfig,
    /// Daemon configuration
    #[serde(default)]
    pub daemon: DaemonConfig,
    /// HTTP API server configuration
    #[serde(default)]
    pub http: HttpConfig,
    /// Metrics configuration
    #[serde(default)]
    pub metrics: MetricsConfig,
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            node: NodeConfig::default(),
            embedding: EmbeddingConfig::default(),
            index: IndexConfig::default(),
            chunking: ChunkingConfig::default(),
            retrieval: RetrievalConfig::default(),
            routing: RoutingConfig::default(),
            scraping: ScrapingConfig::default(),
            bulk_import: BulkImportConfig::default(),
            dedup: DedupConfig::default(),
            daemon: DaemonConfig::default(),
            http: HttpConfig::default(),
            metrics: MetricsConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a TOML file.
    ///
    /// After deserializing, this validates all fields and resolves embedding
    /// model paths from the data directory so callers don't need to remember
    /// to call `resolve_paths` themselves.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read config file '{}': {}", path.display(), e))?;
        let mut config: Config = toml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file '{}': {}", path.display(), e))?;
        config.validate()?;
        config.embedding.resolve_paths(&config.node.data_dir);
        Ok(config)
    }

    /// Validate all configuration fields.
    ///
    /// Collects all validation errors and reports them together so the user
    /// can fix everything in one pass rather than playing whack-a-mole.
    pub fn validate(&self) -> Result<()> {
        let mut errors: Vec<String> = Vec::new();

        // Embedding validation
        if self.embedding.dimensions == 0 {
            errors.push("embedding dimensions must be positive".to_string());
        }
        if self.embedding.dimensions > 4096 {
            errors.push("embedding dimensions must be <= 4096".to_string());
        }

        // Chunking validation
        if self.chunking.chunk_size == 0 {
            errors.push("chunk_size must be positive".to_string());
        }
        if self.chunking.chunk_size > 8192 {
            errors.push("chunk_size must be <= 8192".to_string());
        }
        if self.chunking.overlap_fraction >= 1.0 {
            errors.push("overlap_fraction must be less than 1.0".to_string());
        }

        // Retrieval validation
        if self.retrieval.rrf_k == 0 {
            errors.push("rrf_k must be positive".to_string());
        }
        if self.retrieval.candidate_count == 0 {
            errors.push("candidate_count must be positive".to_string());
        }
        if self.retrieval.fanout_quality_threshold < 0.0
            || self.retrieval.fanout_quality_threshold > 1.0
        {
            errors.push("fanout_quality_threshold must be between 0.0 and 1.0".to_string());
        }
        if self.retrieval.fanout_min_results == 0 {
            errors.push("fanout_min_results must be positive".to_string());
        }
        if self.retrieval.max_fanout_peers == 0 {
            errors.push("max_fanout_peers must be positive".to_string());
        }
        if self.retrieval.fanout_score_threshold < 0.0
            || self.retrieval.fanout_score_threshold > 1.0
        {
            errors.push("fanout_score_threshold must be between 0.0 and 1.0".to_string());
        }
        if self.retrieval.fanout_timeout_fraction <= 0.0
            || self.retrieval.fanout_timeout_fraction > 1.0
        {
            errors.push("fanout_timeout_fraction must be between 0.0 (exclusive) and 1.0".to_string());
        }

        // Index validation
        if self.index.hnsw_ef_construction == 0 {
            errors.push("ef_construction must be positive".to_string());
        }
        if self.index.hnsw_ef_search == 0 {
            errors.push("ef_search must be positive".to_string());
        }
        if self.index.hnsw_m == 0 {
            errors.push("HNSW M parameter must be positive".to_string());
        }

        // Routing validation
        if self.routing.lsh_num_bands == 0 {
            errors.push("lsh_num_bands must be positive".to_string());
        }
        if self.routing.lsh_bits > 0 && self.routing.lsh_num_bands > 0
            && !self.routing.lsh_bits.is_multiple_of(self.routing.lsh_num_bands)
        {
            errors.push(format!(
                "lsh_bits ({}) must be divisible by lsh_num_bands ({})",
                self.routing.lsh_bits, self.routing.lsh_num_bands
            ));
        }
        if self.routing.centroid_similarity_threshold < 0.0
            || self.routing.centroid_similarity_threshold > 1.0
        {
            errors.push("centroid_similarity_threshold must be between 0.0 and 1.0".to_string());
        }
        if self.routing.bloom_false_positive_rate <= 0.0
            || self.routing.bloom_false_positive_rate >= 1.0
        {
            errors.push("bloom_false_positive_rate must be between 0.0 (exclusive) and 1.0 (exclusive)".to_string());
        }

        // HTTP config validation
        if self.http.enabled && !self.http.listen_addr.is_empty() {
            // Extract port from listen_addr (format: "host:port")
            if let Some(port_str) = self.http.listen_addr.rsplit(':').next() {
                if let Ok(port) = port_str.parse::<u32>() {
                    if port == 0 || port > 65535 {
                        errors.push(format!(
                            "HTTP listen port must be between 1 and 65535, got {}",
                            port
                        ));
                    }
                }
            }
        }

        // Node validation
        if self.node.data_dir.as_os_str().is_empty() {
            errors.push("data_dir must not be empty".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            anyhow::bail!(
                "Configuration validation failed:\n  - {}",
                errors.join("\n  - ")
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ========================================================================
    // Helper: build a valid default config for mutation-based testing
    // ========================================================================

    fn valid_config() -> Config {
        Config::default()
    }

    // ========================================================================
    // Config::validate – happy path
    // ========================================================================

    #[test]
    fn default_config_passes_validation() {
        let cfg = valid_config();
        assert!(cfg.validate().is_ok(), "default config should be valid");
    }

    // ========================================================================
    // Config::validate – embedding dimension errors
    // ========================================================================

    #[test]
    fn validate_rejects_zero_embedding_dimensions() {
        let mut cfg = valid_config();
        cfg.embedding.dimensions = 0;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("embedding dimensions must be positive"),
            "unexpected error message: {}",
            err
        );
    }

    #[test]
    fn validate_rejects_oversized_embedding_dimensions() {
        let mut cfg = valid_config();
        cfg.embedding.dimensions = 5000;
        let err = cfg.validate().unwrap_err();
        assert!(
            err.to_string().contains("embedding dimensions must be <= 4096"),
            "unexpected error message: {}",
            err
        );
    }

    #[test]
    fn validate_accepts_max_embedding_dimensions() {
        let mut cfg = valid_config();
        cfg.embedding.dimensions = 4096;
        assert!(cfg.validate().is_ok());
    }

    // ========================================================================
    // Config::validate – chunking errors
    // ========================================================================

    #[test]
    fn validate_rejects_zero_chunk_size() {
        let mut cfg = valid_config();
        cfg.chunking.chunk_size = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("chunk_size must be positive"));
    }

    #[test]
    fn validate_rejects_oversized_chunk_size() {
        let mut cfg = valid_config();
        cfg.chunking.chunk_size = 10000;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("chunk_size must be <= 8192"));
    }

    #[test]
    fn validate_accepts_max_chunk_size() {
        let mut cfg = valid_config();
        cfg.chunking.chunk_size = 8192;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_overlap_fraction_gte_one() {
        let mut cfg = valid_config();
        cfg.chunking.overlap_fraction = 1.0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("overlap_fraction must be less than 1.0"));
    }

    #[test]
    fn validate_rejects_overlap_fraction_greater_than_one() {
        let mut cfg = valid_config();
        cfg.chunking.overlap_fraction = 1.5;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("overlap_fraction must be less than 1.0"));
    }

    #[test]
    fn validate_accepts_zero_overlap_fraction() {
        let mut cfg = valid_config();
        cfg.chunking.overlap_fraction = 0.0;
        assert!(cfg.validate().is_ok());
    }

    // ========================================================================
    // Config::validate – retrieval errors
    // ========================================================================

    #[test]
    fn validate_rejects_zero_rrf_k() {
        let mut cfg = valid_config();
        cfg.retrieval.rrf_k = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("rrf_k must be positive"));
    }

    #[test]
    fn validate_rejects_zero_candidate_count() {
        let mut cfg = valid_config();
        cfg.retrieval.candidate_count = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("candidate_count must be positive"));
    }

    // ========================================================================
    // Config::validate – HNSW index errors
    // ========================================================================

    #[test]
    fn validate_rejects_hnsw_m_zero() {
        let mut cfg = valid_config();
        cfg.index.hnsw_m = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("HNSW M parameter must be positive"));
    }

    #[test]
    fn validate_rejects_hnsw_ef_construction_zero() {
        let mut cfg = valid_config();
        cfg.index.hnsw_ef_construction = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("ef_construction must be positive"));
    }

    #[test]
    fn validate_rejects_hnsw_ef_search_zero() {
        let mut cfg = valid_config();
        cfg.index.hnsw_ef_search = 0;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("ef_search must be positive"));
    }

    // ========================================================================
    // Config::validate – HTTP port errors
    // ========================================================================

    #[test]
    fn validate_rejects_http_port_zero() {
        let mut cfg = valid_config();
        cfg.http.enabled = true;
        cfg.http.listen_addr = "0.0.0.0:0".to_string();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("HTTP listen port must be between 1 and 65535"));
    }

    #[test]
    fn validate_rejects_http_port_too_large() {
        let mut cfg = valid_config();
        cfg.http.enabled = true;
        cfg.http.listen_addr = "0.0.0.0:70000".to_string();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("HTTP listen port must be between 1 and 65535"));
    }

    #[test]
    fn validate_accepts_valid_http_port() {
        let mut cfg = valid_config();
        cfg.http.enabled = true;
        cfg.http.listen_addr = "127.0.0.1:8080".to_string();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_skips_http_port_check_when_disabled() {
        let mut cfg = valid_config();
        cfg.http.enabled = false;
        cfg.http.listen_addr = "0.0.0.0:0".to_string();
        // Port validation is only performed when HTTP is enabled
        assert!(cfg.validate().is_ok());
    }

    // ========================================================================
    // Config::validate – node data_dir
    // ========================================================================

    #[test]
    fn validate_rejects_empty_data_dir() {
        let mut cfg = valid_config();
        cfg.node.data_dir = PathBuf::from("");
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("data_dir must not be empty"));
    }

    // ========================================================================
    // Config::validate – multiple errors collected
    // ========================================================================

    #[test]
    fn validate_collects_multiple_errors() {
        let mut cfg = valid_config();
        cfg.embedding.dimensions = 0;
        cfg.chunking.chunk_size = 0;
        cfg.index.hnsw_m = 0;
        let err = cfg.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("embedding dimensions must be positive"));
        assert!(msg.contains("chunk_size must be positive"));
        assert!(msg.contains("HNSW M parameter must be positive"));
    }

    // ========================================================================
    // Default implementations – spot-check important values
    // ========================================================================

    #[test]
    fn default_embedding_config_values() {
        let emb = EmbeddingConfig::default();
        assert_eq!(emb.model_name, "all-MiniLM-L6-v2");
        assert_eq!(emb.dimensions, 384);
        assert_eq!(emb.truncated_dimensions, 384);
        assert_eq!(emb.max_sequence_length, 256);
        assert_eq!(emb.timeout_secs, 30);
        assert_eq!(emb.max_batch_size, 100);
        assert!(emb.backend.is_none());
        assert!(emb.endpoint.is_none());
        assert!(emb.model_path.is_none());
        assert!(emb.tokenizer_path.is_none());
        assert!(!emb.quantize_int8);
        assert_eq!(emb.gpu_device_id, 0);
    }

    #[test]
    fn default_index_config_values() {
        let idx = IndexConfig::default();
        assert_eq!(idx.hnsw_m, 16);
        assert_eq!(idx.hnsw_ef_construction, 200);
        assert_eq!(idx.hnsw_ef_search, 100);
        assert!(idx.memory_mapped);
        assert_eq!(idx.max_capacity, 1_000_000);
    }

    #[test]
    fn default_chunking_config_values() {
        let ch = ChunkingConfig::default();
        assert_eq!(ch.chunk_size, 512);
        assert!((ch.overlap_fraction - 0.15).abs() < f32::EPSILON);
        assert_eq!(ch.min_chunk_size, 50);
        assert_eq!(ch.max_chunk_size, 1024);
    }

    #[test]
    fn default_retrieval_config_values() {
        let ret = RetrievalConfig::default();
        assert!(ret.enable_dense);
        assert!(ret.enable_bm25);
        assert_eq!(ret.rrf_k, 60);
        assert_eq!(ret.candidate_count, 50);
        assert!(ret.enable_reranking);
        assert!(ret.reranker_model_path.is_none());
        assert!((ret.fanout_quality_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(ret.fanout_min_results, 3);
        assert_eq!(ret.max_fanout_peers, 10);
        assert!((ret.fanout_score_threshold - 0.3).abs() < f32::EPSILON);
        assert!((ret.fanout_timeout_fraction - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn default_routing_config_values() {
        let rt = RoutingConfig::default();
        assert_eq!(rt.num_centroids, 100);
        assert_eq!(rt.lsh_bits, 128);
        assert_eq!(rt.lsh_num_hashes, 8);
        assert_eq!(rt.bloom_bits_per_item, 10);
        assert_eq!(rt.candidate_nodes, 5);
        assert_eq!(rt.lsh_num_bands, 8);
        assert!((rt.centroid_similarity_threshold - 0.5).abs() < f32::EPSILON);
        assert!((rt.bloom_false_positive_rate - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn default_http_config_values() {
        let h = HttpConfig::default();
        assert!(!h.enabled);
        assert_eq!(h.listen_addr, "127.0.0.1:8080");
        assert!(h.api_keys.is_empty());
        assert!(!h.cors_enabled);
    }

    #[test]
    fn default_node_config_has_nonempty_data_dir() {
        let node = NodeConfig::default();
        assert!(!node.data_dir.as_os_str().is_empty());
        assert_eq!(node.replication_factor, 3);
        assert_eq!(node.query_timeout_secs, 10);
        assert!(node.enable_mdns);
    }

    // ========================================================================
    // EmbeddingConfig::resolve_backend
    // ========================================================================

    #[test]
    fn resolve_backend_http_with_endpoint() {
        let mut emb = EmbeddingConfig::default();
        emb.backend = Some("http".to_string());
        emb.endpoint = Some("https://api.openai.com/v1/embeddings".to_string());
        emb.model = Some("text-embedding-3-small".to_string());
        emb.api_key = Some("sk-test".to_string());
        emb.dimensions = 1536;

        let backend = emb.resolve_backend().expect("should produce Http backend");
        let BackendConfig::Http {
            endpoint,
            api_key,
            model,
            dimensions,
            timeout_secs,
            max_batch_size,
        } = backend;
        assert_eq!(endpoint, "https://api.openai.com/v1/embeddings");
        assert_eq!(api_key, Some("sk-test".to_string()));
        assert_eq!(model, "text-embedding-3-small");
        assert_eq!(dimensions, 1536);
        assert_eq!(timeout_secs, 30);
        assert_eq!(max_batch_size, 100);
    }

    #[test]
    fn resolve_backend_http_falls_back_to_model_name() {
        let mut emb = EmbeddingConfig::default();
        emb.backend = Some("http".to_string());
        emb.endpoint = Some("http://localhost:11434/v1/embeddings".to_string());
        // model is None, should fall back to model_name
        emb.model = None;

        let backend = emb.resolve_backend().expect("should produce Http backend");
        let BackendConfig::Http { model, .. } = backend;
        assert_eq!(model, "all-MiniLM-L6-v2");
    }

    #[test]
    fn resolve_backend_http_without_endpoint_returns_none() {
        let mut emb = EmbeddingConfig::default();
        emb.backend = Some("http".to_string());
        emb.endpoint = None;
        assert!(
            emb.resolve_backend().is_none(),
            "http backend without endpoint should return None"
        );
    }

    #[test]
    fn resolve_backend_none_when_no_backend_set() {
        let emb = EmbeddingConfig::default();
        assert!(emb.backend.is_none());
        assert!(
            emb.resolve_backend().is_none(),
            "default config with no backend should return None"
        );
    }

    #[test]
    fn resolve_backend_none_for_unknown_backend() {
        let mut emb = EmbeddingConfig::default();
        emb.backend = Some("grpc".to_string());
        assert!(
            emb.resolve_backend().is_none(),
            "unknown backend type should return None"
        );
    }

    // ========================================================================
    // EmbeddingConfig::resolve_paths
    // ========================================================================

    #[test]
    fn resolve_paths_does_not_set_paths_when_files_missing() {
        let mut emb = EmbeddingConfig::default();
        let tmp = tempfile::tempdir().unwrap();
        emb.resolve_paths(tmp.path());
        // No model files exist in the temp dir, so paths should remain None
        assert!(emb.model_path.is_none());
        assert!(emb.tokenizer_path.is_none());
    }

    #[test]
    fn resolve_paths_sets_paths_when_files_exist() {
        let mut emb = EmbeddingConfig::default();
        let tmp = tempfile::tempdir().unwrap();

        // Create the expected directory structure
        let model_dir = tmp.path().join("models").join(&emb.model_name);
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"fake model").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();

        emb.resolve_paths(tmp.path());
        assert_eq!(emb.model_path, Some(model_dir.join("model.onnx")));
        assert_eq!(emb.tokenizer_path, Some(model_dir.join("tokenizer.json")));
    }

    #[test]
    fn resolve_paths_does_not_overwrite_existing_paths() {
        let mut emb = EmbeddingConfig::default();
        let existing = PathBuf::from("/custom/model.onnx");
        emb.model_path = Some(existing.clone());
        emb.tokenizer_path = Some(PathBuf::from("/custom/tokenizer.json"));

        let tmp = tempfile::tempdir().unwrap();
        // Create files that would normally be picked up
        let model_dir = tmp.path().join("models").join(&emb.model_name);
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"fake").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();

        emb.resolve_paths(tmp.path());
        // Should keep the pre-existing custom paths
        assert_eq!(emb.model_path, Some(existing));
        assert_eq!(emb.tokenizer_path, Some(PathBuf::from("/custom/tokenizer.json")));
    }
}
