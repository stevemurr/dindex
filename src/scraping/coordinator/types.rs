//! Coordinator types: configuration, results, and statistics

use std::time::{Duration, Instant};
use url::Url;

use crate::scraping::{
    dedup::SimHash,
    extractor::{ExtractedContent, ExtractedMetadata, ExtractorConfig},
    fetcher::FetchConfig,
    politeness::PolitenessConfig,
    trap_detection::TrapDetectorConfig,
};

/// Configuration for the scraping coordinator
#[derive(Debug, Clone)]
pub struct ScrapingConfig {
    /// Enable scraping
    pub enabled: bool,
    /// Maximum concurrent fetches
    pub max_concurrent_fetches: usize,
    /// Maximum crawl depth
    pub max_depth: u8,
    /// Stay within seed domains only
    pub stay_on_domain: bool,
    /// URL patterns to include (regex)
    pub include_patterns: Vec<String>,
    /// URL patterns to exclude (regex)
    pub exclude_patterns: Vec<String>,
    /// Maximum pages per domain
    pub max_pages_per_domain: usize,
    /// Scraping interval (how often to check for new URLs)
    pub scrape_interval: Duration,
    /// Politeness configuration
    pub politeness: PolitenessConfig,
    /// Fetch configuration
    pub fetch: FetchConfig,
    /// Extractor configuration
    pub extractor: ExtractorConfig,
    /// Crawl trap detection configuration
    pub trap_detector: TrapDetectorConfig,
}

impl ScrapingConfig {
    /// Build a ScrapingConfig from the TOML ScrapingConfig + per-job overrides.
    ///
    /// Ensures consistent settings regardless of whether the scrape is initiated
    /// from the CLI command or the daemon job manager.
    pub fn from_config(
        config: &crate::config::ScrapingConfig,
        max_depth: u8,
        stay_on_domain: bool,
        delay_ms: u64,
        max_pages: usize,
    ) -> Self {
        Self {
            enabled: true,
            max_concurrent_fetches: config.max_concurrent_fetches,
            max_depth,
            stay_on_domain,
            include_patterns: config.include_patterns.clone(),
            exclude_patterns: config.exclude_patterns.clone(),
            max_pages_per_domain: max_pages,
            scrape_interval: Duration::from_millis(100),
            politeness: PolitenessConfig {
                user_agent: config.user_agent.clone(),
                default_delay: Duration::from_millis(delay_ms),
                min_delay: Duration::from_millis(delay_ms / 2),
                max_delay: Duration::from_secs(30),
                cache_size: 10_000,
                request_timeout: Duration::from_secs(config.request_timeout_secs),
            },
            fetch: FetchConfig {
                user_agent: config.user_agent.clone(),
                timeout: Duration::from_secs(config.request_timeout_secs),
                connect_timeout: Duration::from_secs(10),
                max_content_size: 10 * 1024 * 1024,
                max_redirects: 10,
                min_text_ratio: 0.1,
                enable_js_rendering: config.enable_js_rendering,
                connections_per_host: config.max_concurrent_fetches,
            },
            extractor: ExtractorConfig::default(),
            trap_detector: TrapDetectorConfig::default(),
        }
    }
}

impl Default for ScrapingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent_fetches: 10,
            max_depth: 3,
            stay_on_domain: false,
            include_patterns: Vec::new(),
            exclude_patterns: Vec::new(),
            max_pages_per_domain: 1000,
            scrape_interval: Duration::from_secs(1),
            politeness: PolitenessConfig::default(),
            fetch: FetchConfig::default(),
            extractor: ExtractorConfig::default(),
            trap_detector: TrapDetectorConfig::default(),
        }
    }
}

/// Successfully processed content from a URL
#[derive(Debug)]
pub struct ProcessedContent {
    pub content: ExtractedContent,
    pub metadata: ExtractedMetadata,
    pub simhash: SimHash,
}

/// Outcome of processing a single URL
#[derive(Debug)]
pub enum ProcessOutcome {
    Success(ProcessedContent),
    Failure { error: String },
}

/// Result of processing a single URL
#[derive(Debug)]
pub struct ProcessResult {
    /// The URL that was processed
    pub url: Url,
    /// Processing outcome (success with content, or failure with error)
    pub outcome: ProcessOutcome,
    /// URLs discovered on this page
    pub discovered_urls: Vec<Url>,
    /// Processing duration
    pub duration: Duration,
}

impl ProcessResult {
    /// Create a failure result with no discovered URLs
    pub(super) fn failure(url: &Url, error: impl Into<String>, start: Instant) -> Self {
        Self {
            url: url.clone(),
            outcome: ProcessOutcome::Failure { error: error.into() },
            discovered_urls: Vec::new(),
            duration: start.elapsed(),
        }
    }
}

/// Statistics from the scraping coordinator
#[derive(Debug, Clone, Default)]
pub struct ScrapingStats {
    /// Total URLs processed
    pub urls_processed: u64,
    /// Successful fetches
    pub successful_fetches: u64,
    /// Failed fetches
    pub failed_fetches: u64,
    /// Duplicate content skipped
    pub duplicates_skipped: u64,
    /// URLs discovered
    pub urls_discovered: u64,
    /// URLs sent to other nodes
    pub urls_exchanged: u64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Current queue size
    pub queue_size: usize,
    /// Domains being crawled
    pub active_domains: usize,
}

/// Capacity of the content deduplicator (max number of tracked documents)
pub(super) const CONTENT_DEDUP_CAPACITY: usize = 100_000;

/// Hamming distance threshold for near-duplicate detection
pub(super) const CONTENT_DEDUP_HAMMING_THRESHOLD: u32 = 3;
