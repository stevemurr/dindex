//! Configuration for DIndex

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

/// Default max sequence length for local backend
fn default_max_sequence_length() -> usize {
    512
}

/// Backend configuration for embedding providers
///
/// Supports two backend types:
/// - `http`: OpenAI-compatible HTTP endpoints (OpenAI, Azure, LM Studio, vLLM, etc.)
/// - `local`: Local inference using embed_anything (CPU/CUDA/Metal)
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
    /// Local inference using embed_anything (legacy)
    ///
    /// Uses embed_anything with candle backend for local inference.
    /// Supports CPU, CUDA (--features cuda), and Metal (--features metal).
    Local {
        /// Model name (e.g., "all-MiniLM-L6-v2", "bge-base-en-v1.5")
        model_name: String,
        /// Embedding dimensions
        dimensions: usize,
        /// Truncated dimensions for Matryoshka routing (optional)
        #[serde(default)]
        truncated_dimensions: Option<usize>,
        /// Maximum sequence length
        #[serde(default = "default_max_sequence_length")]
        max_sequence_length: usize,
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

/// Node networking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Listen address for P2P connections
    pub listen_addr: String,
    /// Bootstrap peers to connect to
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
/// 1. **New style** (recommended): Use the `backend` field with `BackendConfig`
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
    /// New-style backend configuration (takes precedence if set)
    #[serde(flatten)]
    pub backend: Option<BackendConfig>,

    // ---- Legacy fields (for backward compatibility) ----

    /// Model name (e.g., "bge-m3", "bge-base-en-v1.5", "all-MiniLM-L6-v2")
    /// Can also be a HuggingFace model ID (e.g., "BAAI/bge-m3")
    #[serde(default = "default_model_name")]
    pub model_name: String,
    /// Path to model files (optional, embed_anything downloads automatically)
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    /// Path to tokenizer files (optional, embed_anything handles internally)
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,
    /// Embedding dimensions (1024 for bge-m3, 768 for bge-base, 384 for MiniLM)
    #[serde(default = "default_dimensions")]
    pub dimensions: usize,
    /// Truncated dimensions for Matryoshka (routing)
    #[serde(default = "default_dimensions")]
    pub truncated_dimensions: usize,
    /// Maximum sequence length
    #[serde(default = "default_max_seq_length")]
    pub max_sequence_length: usize,
    /// Enable INT8 quantization (deprecated, handled by backend)
    #[serde(default)]
    pub quantize_int8: bool,
    /// Number of threads for inference
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
    /// Use GPU acceleration (CUDA or Metal based on platform)
    #[serde(default = "default_use_gpu")]
    pub use_gpu: bool,
    /// GPU device ID (for multi-GPU systems)
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
    num_cpus::get().min(8)
}

/// Default for use_gpu - true when cuda or metal feature is enabled
fn default_use_gpu() -> bool {
    cfg!(feature = "cuda") || cfg!(feature = "metal")
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: None, // Use legacy fields by default for backward compatibility
            model_name: "all-MiniLM-L6-v2".to_string(),
            model_path: None,
            tokenizer_path: None,
            dimensions: 384,  // all-MiniLM-L6-v2 dimensions
            truncated_dimensions: 384,  // No truncation for this model
            max_sequence_length: 256,  // all-MiniLM-L6-v2 max sequence
            quantize_int8: false,
            num_threads: num_cpus::get().min(8),
            use_gpu: default_use_gpu(),
            gpu_device_id: 0,
        }
    }
}

impl EmbeddingConfig {
    /// Resolve model paths from the data directory if not explicitly set.
    /// Note: embed_anything handles model downloading automatically,
    /// so this is primarily for backward compatibility.
    pub fn resolve_paths(&mut self, data_dir: &std::path::Path) {
        // embed_anything handles model caching in ~/.cache/huggingface/
        // This method is kept for backward compatibility but is no longer required
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
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            num_centroids: 100,
            lsh_bits: 128,
            lsh_num_hashes: 8,
            bloom_bits_per_item: 10,
            candidate_nodes: 5,
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

// Add num_cpus as a simple function since we're using it
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}
