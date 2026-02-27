//! Configuration for DIndex

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ============================================================================
// Embedding Backend Configuration
// ============================================================================

/// Default timeout for HTTP backend requests
fn default_timeout() -> u64 {
    30
}

/// Default batch size for HTTP backend requests
fn default_batch_size() -> usize {
    100
}

/// Backend configuration for embedding providers
///
/// Supports OpenAI-compatible HTTP endpoints (OpenAI, Azure, LM Studio, vLLM, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "lowercase")]
pub enum BackendConfig {
    /// OpenAI-compatible HTTP endpoint
    ///
    /// Works with: OpenAI API, Azure OpenAI, LM Studio, vLLM,
    /// Ollama (OpenAI compat mode), text-embeddings-inference
    Http {
        /// API endpoint URL (e.g., "https://api.openai.com/v1/embeddings")
        endpoint: String,
        /// API key (optional, can also use OPENAI_API_KEY env var)
        #[serde(default)]
        api_key: Option<String>,
        /// Model name (e.g., "text-embedding-3-small")
        model: String,
        /// Embedding dimensions
        dimensions: usize,
        /// Request timeout in seconds
        #[serde(default = "default_timeout")]
        timeout_secs: u64,
        /// Maximum batch size for requests
        #[serde(default = "default_batch_size")]
        max_batch_size: usize,
    },
}

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

/// Node networking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Listen address for P2P connections
    pub listen_addr: String,
    /// Bootstrap peers to connect to
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,
    /// Data directory for persistence
    pub data_dir: PathBuf,
    /// Enable mDNS for local peer discovery
    pub enable_mdns: bool,
    /// DHT replication factor (k)
    pub replication_factor: usize,
    /// Query timeout in seconds
    pub query_timeout_secs: u64,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            listen_addr: "/ip4/0.0.0.0/udp/0/quic-v1".to_string(),
            bootstrap_peers: Vec::new(),
            data_dir: directories::ProjectDirs::from("", "", "dindex")
                .map(|d| d.data_dir().to_path_buf())
                .unwrap_or_else(|| PathBuf::from(".dindex")),
            enable_mdns: true,
            replication_factor: 3,
            query_timeout_secs: 10,
        }
    }
}

/// Embedding model configuration
///
/// Supports two configuration styles:
///
/// 1. **New style** (recommended): Use `backend = "http"` with endpoint fields
/// ```toml
/// [embedding]
/// backend = "http"
/// endpoint = "https://api.openai.com/v1/embeddings"
/// model = "text-embedding-3-small"
/// dimensions = 1536
/// ```
///
/// 2. **Legacy style**: Use the flat fields (backward compatible)
/// ```toml
/// [embedding]
/// model_name = "all-MiniLM-L6-v2"
/// dimensions = 384
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Backend type: "http" or "local" (default: "local" for legacy compat)
    #[serde(default)]
    pub backend: Option<String>,

    /// HTTP backend: API endpoint URL
    #[serde(default)]
    pub endpoint: Option<String>,
    /// HTTP backend: API key (optional, can also use OPENAI_API_KEY env var)
    #[serde(default)]
    pub api_key: Option<String>,
    /// HTTP backend: Model name for API requests (e.g., "bge-m3")
    #[serde(default)]
    pub model: Option<String>,
    /// HTTP backend: Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// HTTP backend: Maximum batch size for requests
    #[serde(default = "default_batch_size")]
    pub max_batch_size: usize,

    // ---- Legacy/local fields ----

    /// Model name for local backend (e.g., "all-MiniLM-L6-v2")
    #[serde(default = "default_model_name")]
    pub model_name: String,
    /// Path to model files (optional)
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    /// Path to tokenizer files (optional)
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,
    /// Embedding dimensions
    #[serde(default = "default_dimensions")]
    pub dimensions: usize,
    /// Truncated dimensions for Matryoshka (routing)
    #[serde(default = "default_dimensions")]
    pub truncated_dimensions: usize,
    /// Maximum sequence length
    #[serde(default = "default_max_seq_length")]
    pub max_sequence_length: usize,
    /// Enable INT8 quantization (deprecated)
    #[serde(default)]
    pub quantize_int8: bool,
    /// Number of threads for inference
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
    /// Use GPU acceleration
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
    /// GPU device ID
    #[serde(default)]
    pub gpu_device_id: usize,
}

fn default_model_name() -> String {
    "all-MiniLM-L6-v2".to_string()
}

fn default_dimensions() -> usize {
    384
}

fn default_max_seq_length() -> usize {
    256
}

fn default_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(8)
}

fn default_use_gpu() -> bool {
    false
}

impl EmbeddingConfig {
    /// Resolve the backend configuration from flat fields
    pub fn resolve_backend(&self) -> Option<BackendConfig> {
        match self.backend.as_deref() {
            Some("http") => {
                let endpoint = self.endpoint.clone()?;
                let model = self.model.clone().unwrap_or_else(|| self.model_name.clone());
                Some(BackendConfig::Http {
                    endpoint,
                    api_key: self.api_key.clone(),
                    model,
                    dimensions: self.dimensions,
                    timeout_secs: self.timeout_secs,
                    max_batch_size: self.max_batch_size,
                })
            }
            _ => None,
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: None,
            endpoint: None,
            api_key: None,
            model: None,
            timeout_secs: 30,
            max_batch_size: 100,
            model_name: "all-MiniLM-L6-v2".to_string(),
            model_path: None,
            tokenizer_path: None,
            dimensions: 384,
            truncated_dimensions: 384,
            max_sequence_length: 256,
            quantize_int8: false,
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
                .min(8),
            use_gpu: default_use_gpu(),
            gpu_device_id: 0,
        }
    }
}

impl EmbeddingConfig {
    /// Resolve model paths from the data directory if not explicitly set.
    /// Kept for backward compatibility with legacy local config layouts.
    pub fn resolve_paths(&mut self, data_dir: &std::path::Path) {
        if self.model_path.is_none() || self.tokenizer_path.is_none() {
            let model_dir = data_dir.join("models").join(&self.model_name);
            let model_file = model_dir.join("model.onnx");
            let tokenizer_file = model_dir.join("tokenizer.json");

            if model_file.exists() && self.model_path.is_none() {
                self.model_path = Some(model_file);
            }
            if tokenizer_file.exists() && self.tokenizer_path.is_none() {
                self.tokenizer_path = Some(tokenizer_file);
            }
        }
    }
}

/// Vector index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// HNSW M parameter (connections per layer)
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter
    pub hnsw_ef_construction: usize,
    /// HNSW ef_search parameter
    pub hnsw_ef_search: usize,
    /// Enable memory mapping
    pub memory_mapped: bool,
    /// Maximum index capacity
    pub max_capacity: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 100,
            memory_mapped: true,
            max_capacity: 1_000_000,
        }
    }
}

/// Chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Base chunk size in tokens
    pub chunk_size: usize,
    /// Overlap between chunks (as fraction, e.g., 0.15)
    pub overlap_fraction: f32,
    /// Minimum chunk size (won't create smaller chunks)
    pub min_chunk_size: usize,
    /// Maximum chunk size
    pub max_chunk_size: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap_fraction: 0.15,
            min_chunk_size: 50,
            max_chunk_size: 1024,
        }
    }
}

/// Retrieval configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Enable dense (vector) retrieval
    pub enable_dense: bool,
    /// Enable BM25 text retrieval
    pub enable_bm25: bool,
    /// RRF k parameter
    pub rrf_k: usize,
    /// Number of candidates to fetch before reranking
    pub candidate_count: usize,
    /// Enable cross-encoder reranking
    pub enable_reranking: bool,
    /// Reranker model path
    pub reranker_model_path: Option<PathBuf>,
    /// Minimum quality score before triggering adaptive fan-out (0.0-1.0)
    #[serde(default = "default_fanout_quality_threshold")]
    pub fanout_quality_threshold: f32,
    /// Minimum number of results before considering fan-out
    #[serde(default = "default_fanout_min_results")]
    pub fanout_min_results: usize,
    /// Maximum number of additional peers to query during fan-out
    #[serde(default = "default_max_fanout_peers")]
    pub max_fanout_peers: usize,
    /// Minimum average result score before triggering fan-out (0.0-1.0)
    #[serde(default = "default_fanout_score_threshold")]
    pub fanout_score_threshold: f32,
    /// Fraction of original timeout to use for tier-2 fan-out queries (0.0-1.0)
    #[serde(default = "default_fanout_timeout_fraction")]
    pub fanout_timeout_fraction: f32,
    /// Minimum score multiplier for aggregator pages (0.0-1.0).
    /// At aggregator_score=1.0, the result score is multiplied by this value.
    #[serde(default = "default_aggregator_min_multiplier")]
    pub aggregator_min_multiplier: f32,
}

fn default_fanout_quality_threshold() -> f32 {
    0.5
}

fn default_fanout_min_results() -> usize {
    3
}

fn default_max_fanout_peers() -> usize {
    10
}

fn default_fanout_score_threshold() -> f32 {
    0.3
}

fn default_fanout_timeout_fraction() -> f32 {
    0.5
}

fn default_aggregator_min_multiplier() -> f32 {
    0.5
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            enable_dense: true,
            enable_bm25: true,
            rrf_k: 60,
            candidate_count: 50,
            enable_reranking: true,
            reranker_model_path: None,
            fanout_quality_threshold: default_fanout_quality_threshold(),
            fanout_min_results: default_fanout_min_results(),
            max_fanout_peers: default_max_fanout_peers(),
            fanout_score_threshold: default_fanout_score_threshold(),
            fanout_timeout_fraction: default_fanout_timeout_fraction(),
            aggregator_min_multiplier: default_aggregator_min_multiplier(),
        }
    }
}

/// Routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Number of centroids per node
    pub num_centroids: usize,
    /// LSH signature bits
    pub lsh_bits: usize,
    /// Number of LSH hash functions
    pub lsh_num_hashes: usize,
    /// Bloom filter size (bits per item)
    pub bloom_bits_per_item: usize,
    /// Number of candidate nodes for queries
    pub candidate_nodes: usize,
    /// Number of LSH bands for banding technique
    #[serde(default = "default_lsh_num_bands")]
    pub lsh_num_bands: usize,
    /// Centroid similarity threshold for matching (0.0-1.0)
    #[serde(default = "default_centroid_similarity_threshold")]
    pub centroid_similarity_threshold: f32,
    /// Bloom filter false positive rate per band (0.0-1.0)
    #[serde(default = "default_bloom_false_positive_rate")]
    pub bloom_false_positive_rate: f64,
}

fn default_lsh_num_bands() -> usize {
    8
}

fn default_centroid_similarity_threshold() -> f32 {
    0.5
}

fn default_bloom_false_positive_rate() -> f64 {
    0.01
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            num_centroids: 100,
            lsh_bits: 128,
            lsh_num_hashes: 8,
            bloom_bits_per_item: 10,
            candidate_nodes: 5,
            lsh_num_bands: default_lsh_num_bands(),
            centroid_similarity_threshold: default_centroid_similarity_threshold(),
            bloom_false_positive_rate: default_bloom_false_positive_rate(),
        }
    }
}

/// Web scraping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapingConfig {
    /// Enable scraping functionality
    pub enabled: bool,
    /// Maximum concurrent fetches
    pub max_concurrent_fetches: usize,
    /// Maximum crawl depth from seed URLs
    pub max_depth: u8,
    /// Stay within seed domains only
    pub stay_on_domain: bool,
    /// URL patterns to include (simple substring matching)
    pub include_patterns: Vec<String>,
    /// URL patterns to exclude (simple substring matching)
    pub exclude_patterns: Vec<String>,
    /// Maximum pages to crawl per domain
    pub max_pages_per_domain: usize,
    /// Politeness delay between requests to same domain (milliseconds)
    pub politeness_delay_ms: u64,
    /// Default request timeout (seconds)
    pub request_timeout_secs: u64,
    /// User agent string
    pub user_agent: String,
    /// Enable headless browser for JS-heavy sites
    pub enable_js_rendering: bool,
    /// robots.txt cache TTL (seconds)
    pub robots_cache_ttl_secs: u64,
    /// URL deduplication bloom filter size
    pub url_bloom_size: usize,
    /// Content deduplication cache size
    pub content_cache_size: usize,
    /// SimHash maximum Hamming distance for near-duplicate detection
    pub simhash_distance_threshold: u32,
}

impl Default for ScrapingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_concurrent_fetches: 10,
            max_depth: 3,
            stay_on_domain: false,
            include_patterns: Vec::new(),
            exclude_patterns: vec![
                ".pdf".to_string(),
                ".jpg".to_string(),
                ".png".to_string(),
                ".gif".to_string(),
                ".zip".to_string(),
                ".tar".to_string(),
                "/login".to_string(),
                "/logout".to_string(),
                "/admin".to_string(),
            ],
            max_pages_per_domain: 1000,
            politeness_delay_ms: 1000,
            request_timeout_secs: 30,
            user_agent: "DecentralizedSearchBot/1.0 (+https://github.com/dindex)".to_string(),
            enable_js_rendering: false,
            robots_cache_ttl_secs: 86400, // 24 hours
            url_bloom_size: 10_000_000,
            content_cache_size: 100_000,
            simhash_distance_threshold: 3,
        }
    }
}

/// Bulk import configuration for offline dumps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkImportConfig {
    /// Default batch size for indexing
    pub batch_size: usize,
    /// Enable checkpointing for resume support
    pub enable_checkpoints: bool,
    /// Checkpoint directory
    pub checkpoint_dir: PathBuf,
    /// Default namespace filter for Wikipedia (0 = main articles only)
    pub wikipedia_namespaces: Vec<i32>,
    /// Minimum content length to import (skip very short articles)
    pub min_content_length: usize,
    /// Enable content deduplication
    pub deduplicate: bool,
    /// Checkpoint interval (every N documents)
    pub checkpoint_interval: usize,
}

impl Default for BulkImportConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            enable_checkpoints: true,
            checkpoint_dir: PathBuf::from(".dindex/checkpoints"),
            wikipedia_namespaces: vec![0], // Main namespace only
            min_content_length: 100,
            deduplicate: true,
            checkpoint_interval: 1000,
        }
    }
}

/// Deduplication configuration for unified document identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupConfig {
    /// Enable deduplication
    pub enabled: bool,
    /// Maximum SimHash Hamming distance for near-duplicate detection
    pub simhash_distance_threshold: u32,
    /// Normalize content before computing identity (lowercase, collapse whitespace)
    pub normalize_content: bool,
    /// Update existing documents when near-duplicates are found
    pub update_near_duplicates: bool,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            simhash_distance_threshold: 3,
            normalize_content: true,
            update_near_duplicates: true,
        }
    }
}

/// Daemon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Automatically start daemon if not running
    pub auto_start: bool,
    /// Socket path override (defaults to XDG_RUNTIME_DIR/dindex/dindex.sock)
    pub socket_path: Option<PathBuf>,
    /// Write pipeline batch size
    pub batch_size: usize,
    /// Commit interval in seconds
    pub commit_interval_secs: u64,
    /// Maximum pending writes before forcing commit
    pub max_pending_writes: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            auto_start: false,
            socket_path: None,
            batch_size: 100,
            commit_interval_secs: 30,
            max_pending_writes: 10000,
        }
    }
}

/// HTTP API server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// Enable HTTP API server
    pub enabled: bool,
    /// Listen address for HTTP server (e.g., "0.0.0.0:8080")
    pub listen_addr: String,
    /// API keys for authentication (empty = no auth required)
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// Enable CORS (useful for browser-based clients)
    pub cors_enabled: bool,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            listen_addr: "127.0.0.1:8080".to_string(),
            api_keys: Vec::new(),
            cors_enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

