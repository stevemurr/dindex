//! Scraping, bulk import, and deduplication configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::DEFAULT_USER_AGENT;

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
            user_agent: DEFAULT_USER_AGENT.to_string(),
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
