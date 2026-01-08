//! Configuration for DIndex

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model name (e.g., "nomic-embed-text-v1.5", "e5-small-v2")
    pub model_name: String,
    /// Path to ONNX model file
    pub model_path: Option<PathBuf>,
    /// Path to tokenizer files
    pub tokenizer_path: Option<PathBuf>,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Truncated dimensions for Matryoshka (routing)
    pub truncated_dimensions: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Enable INT8 quantization
    pub quantize_int8: bool,
    /// Number of threads for inference
    pub num_threads: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "nomic-embed-text-v1.5".to_string(),
            model_path: None,
            tokenizer_path: None,
            dimensions: 768,
            truncated_dimensions: 256,
            max_sequence_length: 8192,
            quantize_int8: true,
            num_threads: num_cpus::get().min(8),
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

// Add num_cpus as a simple function since we're using it
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}
