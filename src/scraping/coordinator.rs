//! Scraping coordinator orchestrating the entire scraping pipeline
//!
//! Coordinates the flow from URL discovery through fetching, extraction,
//! deduplication, and indexing. Manages the scraping lifecycle and integrates
//! with the P2P network for distributed crawling.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use url::Url;

use super::{
    dedup::{ContentDeduplicator, SimHash, UrlDeduplicator},
    domain_assignment::{DomainAssignment, PeerId},
    extractor::{ContentExtractor, ExtractedContent, ExtractedMetadata, ExtractorConfig},
    fetcher::{extract_urls, FetchConfig, FetchEngine, FetchError, FetchResult},
    frontier::{ScoredUrl, UrlFrontier},
    politeness::{FetchDecision, PolitenessConfig, PolitenessController},
};

use crate::index::{DocumentRegistry, DuplicateCheckResult};
use crate::types::{Document, DocumentIdentity};

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
        }
    }
}

/// Result of processing a single URL
#[derive(Debug)]
pub struct ProcessResult {
    /// The URL that was processed
    pub url: Url,
    /// Whether processing was successful
    pub success: bool,
    /// Extracted content (if successful)
    pub content: Option<ExtractedContent>,
    /// Extracted metadata
    pub metadata: Option<ExtractedMetadata>,
    /// SimHash of the content
    pub simhash: Option<SimHash>,
    /// URLs discovered on this page
    pub discovered_urls: Vec<Url>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Processing duration
    pub duration: Duration,
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

/// Scraping coordinator managing the entire scraping pipeline
pub struct ScrapingCoordinator {
    /// Configuration
    config: ScrapingConfig,
    /// Local peer ID - stored for potential future distributed coordination.
    /// Currently passed to sub-components during initialization.
    #[allow(dead_code)]
    local_peer_id: PeerId,
    /// Domain assignment
    domain_assignment: Arc<RwLock<DomainAssignment>>,
    /// URL frontier
    frontier: Arc<RwLock<UrlFrontier>>,
    /// Politeness controller
    politeness: Arc<RwLock<PolitenessController>>,
    /// Fetch engine
    fetcher: Arc<RwLock<FetchEngine>>,
    /// Content extractor
    extractor: ContentExtractor,
    /// URL deduplicator
    url_dedup: Arc<RwLock<UrlDeduplicator>>,
    /// Content deduplicator (legacy, used if no registry is provided)
    content_dedup: Arc<RwLock<ContentDeduplicator>>,
    /// Document registry for unified deduplication (optional)
    document_registry: Option<Arc<DocumentRegistry>>,
    /// Statistics
    stats: Arc<RwLock<ScrapingStats>>,
    /// Seed domains (for stay_on_domain mode)
    seed_domains: HashSet<String>,
    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl ScrapingCoordinator {
    /// Create a new scraping coordinator
    pub fn new(config: ScrapingConfig, local_peer_id: PeerId) -> Result<Self, FetchError> {
        let mut domain_assignment = DomainAssignment::with_defaults();
        domain_assignment.set_local_peer(local_peer_id.clone());
        domain_assignment.on_node_join(local_peer_id.clone());

        // Create frontier with a copy of domain assignment
        let frontier = UrlFrontier::new(local_peer_id.clone(), domain_assignment.clone());

        let domain_assignment = Arc::new(RwLock::new(domain_assignment));

        let politeness = PolitenessController::new(config.politeness.clone());
        let fetcher = FetchEngine::new(config.fetch.clone())?;
        let extractor = ContentExtractor::new(config.extractor.clone());

        let url_dedup = UrlDeduplicator::default();
        let content_dedup = ContentDeduplicator::new(100_000, 3, local_peer_id.clone());

        Ok(Self {
            config,
            local_peer_id,
            domain_assignment,
            frontier: Arc::new(RwLock::new(frontier)),
            politeness: Arc::new(RwLock::new(politeness)),
            fetcher: Arc::new(RwLock::new(fetcher)),
            extractor,
            url_dedup: Arc::new(RwLock::new(url_dedup)),
            content_dedup: Arc::new(RwLock::new(content_dedup)),
            document_registry: None,
            stats: Arc::new(RwLock::new(ScrapingStats::default())),
            seed_domains: HashSet::new(),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Set the document registry for unified deduplication
    pub fn with_document_registry(mut self, registry: Arc<DocumentRegistry>) -> Self {
        self.document_registry = Some(registry);
        self
    }

    /// Add seed URLs to start crawling
    pub async fn add_seeds(&mut self, seeds: Vec<Url>) {
        // Track seed domains
        for url in &seeds {
            if let Some(domain) = url.host_str() {
                self.seed_domains.insert(domain.to_string());
            }
        }

        let mut frontier = self.frontier.write().await;
        frontier.add_seeds(seeds);
    }

    /// Process a single URL
    pub async fn process_url(&self, url: &Url) -> ProcessResult {
        let start = Instant::now();

        // Check URL deduplication
        {
            let mut url_dedup = self.url_dedup.write().await;
            if !url_dedup.is_new_url(url) {
                return ProcessResult {
                    url: url.clone(),
                    success: false,
                    content: None,
                    metadata: None,
                    simhash: None,
                    discovered_urls: Vec::new(),
                    error: Some("URL already seen".to_string()),
                    duration: start.elapsed(),
                };
            }
        }

        // Check politeness
        let fetch_decision = {
            let mut politeness = self.politeness.write().await;
            politeness.can_fetch(url).await
        };

        match fetch_decision {
            FetchDecision::Disallowed => {
                return ProcessResult {
                    url: url.clone(),
                    success: false,
                    content: None,
                    metadata: None,
                    simhash: None,
                    discovered_urls: Vec::new(),
                    error: Some("Disallowed by robots.txt".to_string()),
                    duration: start.elapsed(),
                };
            }
            FetchDecision::WaitFor(duration) => {
                tokio::time::sleep(duration).await;
            }
            FetchDecision::RateLimited(until) => {
                let now = Instant::now();
                if until > now {
                    tokio::time::sleep(until - now).await;
                }
            }
            FetchDecision::Allowed => {}
        }

        // Fetch the URL
        let fetch_result = {
            let mut fetcher = self.fetcher.write().await;
            fetcher.fetch(url).await
        };

        let hostname = url.host_str().unwrap_or_default();

        let response = match fetch_result {
            Ok(resp) => {
                let mut politeness = self.politeness.write().await;
                politeness.record_success(hostname);
                resp
            }
            Err(FetchError::Http(e)) if e.status().map(|s| s.as_u16()) == Some(429) => {
                let mut politeness = self.politeness.write().await;
                politeness.record_429(hostname, None);
                return ProcessResult {
                    url: url.clone(),
                    success: false,
                    content: None,
                    metadata: None,
                    simhash: None,
                    discovered_urls: Vec::new(),
                    error: Some("Rate limited (429)".to_string()),
                    duration: start.elapsed(),
                };
            }
            Err(e) => {
                let mut politeness = self.politeness.write().await;
                politeness.record_error(hostname);
                return ProcessResult {
                    url: url.clone(),
                    success: false,
                    content: None,
                    metadata: None,
                    simhash: None,
                    discovered_urls: Vec::new(),
                    error: Some(format!("Fetch error: {}", e)),
                    duration: start.elapsed(),
                };
            }
        };

        // Extract content
        let content = match self.extractor.extract(&response.body, url) {
            Ok(c) => c,
            Err(e) => {
                return ProcessResult {
                    url: url.clone(),
                    success: false,
                    content: None,
                    metadata: None,
                    simhash: None,
                    discovered_urls: extract_urls(&response),
                    error: Some(format!("Extraction error: {}", e)),
                    duration: start.elapsed(),
                };
            }
        };

        // Check content deduplication using registry if available, otherwise use legacy dedup
        let simhash = if let Some(ref registry) = self.document_registry {
            // Use unified document registry for deduplication
            let identity = DocumentIdentity::compute(&content.text_content);

            match registry.check_duplicate(&identity) {
                DuplicateCheckResult::ExactMatch { entry } => {
                    // Exact match - update URL mapping and skip
                    registry.update_metadata(&entry.content_id, Some(url.to_string()), None);

                    let mut stats = self.stats.write().await;
                    stats.duplicates_skipped += 1;

                    return ProcessResult {
                        url: url.clone(),
                        success: false,
                        content: None,
                        metadata: None,
                        simhash: None,
                        discovered_urls: Vec::new(),
                        error: Some(format!("Duplicate of {}", entry.content_id)),
                        duration: start.elapsed(),
                    };
                }
                DuplicateCheckResult::NearDuplicate { entry, hamming_distance } => {
                    // Near-duplicate - update URL mapping and skip
                    registry.update_metadata(&entry.content_id, Some(url.to_string()), None);

                    let mut stats = self.stats.write().await;
                    stats.duplicates_skipped += 1;

                    return ProcessResult {
                        url: url.clone(),
                        success: false,
                        content: None,
                        metadata: None,
                        simhash: None,
                        discovered_urls: Vec::new(),
                        error: Some(format!("Near-duplicate of {} (distance: {})", entry.content_id, hamming_distance)),
                        duration: start.elapsed(),
                    };
                }
                DuplicateCheckResult::New => {
                    // New content - will be registered after processing
                    SimHash(identity.simhash)
                }
            }
        } else {
            // Legacy deduplication using ContentDeduplicator
            let mut content_dedup = self.content_dedup.write().await;

            if let Some(existing_doc) = content_dedup.is_duplicate_local(&content.text_content) {
                let mut stats = self.stats.write().await;
                stats.duplicates_skipped += 1;

                return ProcessResult {
                    url: url.clone(),
                    success: false,
                    content: None,
                    metadata: None,
                    simhash: None,
                    discovered_urls: Vec::new(),
                    error: Some(format!("Duplicate of {}", existing_doc)),
                    duration: start.elapsed(),
                };
            }

            // Register new content
            let doc_id = format!("{}:{}", hostname, uuid::Uuid::new_v4());
            content_dedup.register(&content.text_content, doc_id)
        };

        // Extract metadata
        let metadata = self.extractor.extract_metadata(&response.body, url);

        // Extract URLs for further crawling
        let discovered_urls = self.filter_discovered_urls(&response, url);

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.urls_processed += 1;
            stats.successful_fetches += 1;
            stats.urls_discovered += discovered_urls.len() as u64;

            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let n = stats.urls_processed as f64;
            stats.avg_processing_time_ms = (stats.avg_processing_time_ms * (n - 1.0) + ms) / n;
        }

        ProcessResult {
            url: url.clone(),
            success: true,
            content: Some(content),
            metadata: Some(metadata),
            simhash: Some(simhash),
            discovered_urls,
            error: None,
            duration: start.elapsed(),
        }
    }

    /// Filter discovered URLs based on configuration
    fn filter_discovered_urls(&self, response: &FetchResult, source_url: &Url) -> Vec<Url> {
        let all_urls = extract_urls(response);
        let _source_domain = source_url.host_str().unwrap_or_default();

        all_urls
            .into_iter()
            .filter(|url| {
                // Check stay_on_domain
                if self.config.stay_on_domain {
                    let domain = url.host_str().unwrap_or_default();
                    if !self.seed_domains.contains(domain) {
                        return false;
                    }
                }

                // Check exclude patterns
                let url_str = url.as_str();
                for pattern in &self.config.exclude_patterns {
                    if url_str.contains(pattern) {
                        return false;
                    }
                }

                // Check include patterns (if any)
                if !self.config.include_patterns.is_empty() {
                    let mut matches = false;
                    for pattern in &self.config.include_patterns {
                        if url_str.contains(pattern) {
                            matches = true;
                            break;
                        }
                    }
                    if !matches {
                        return false;
                    }
                }

                true
            })
            .collect()
    }

    /// Add discovered URLs to the frontier
    pub async fn add_discovered_urls(&self, urls: Vec<Url>, depth: u8) {
        if depth >= self.config.max_depth {
            return;
        }

        let mut frontier = self.frontier.write().await;
        for url in urls {
            frontier.add_discovered_url(url, depth + 1);
        }
    }

    /// Get the next URL to process
    pub async fn get_next_url(&self) -> Option<ScoredUrl> {
        let politeness = self.politeness.read().await;
        let ready_domains: HashSet<String> = politeness.ready_domains().into_iter().collect();
        drop(politeness);

        let mut frontier = self.frontier.write().await;
        frontier.pop_next(&ready_domains)
    }

    /// Run the scraping loop
    pub async fn run(&self) {
        {
            let mut running = self.running.write().await;
            *running = true;
        }

        tracing::info!("Starting scraping coordinator");

        while *self.running.read().await {
            // Get next URL
            let next_url = self.get_next_url().await;

            match next_url {
                Some(scored_url) => {
                    let result = self.process_url(&scored_url.url).await;

                    if result.success {
                        tracing::debug!(
                            "Processed {} - {} words, {} URLs discovered",
                            result.url,
                            result.content.as_ref().map(|c| c.word_count).unwrap_or(0),
                            result.discovered_urls.len()
                        );

                        // Add discovered URLs
                        self.add_discovered_urls(result.discovered_urls, scored_url.depth)
                            .await;
                    } else {
                        tracing::debug!(
                            "Failed to process {}: {}",
                            result.url,
                            result.error.unwrap_or_default()
                        );
                    }
                }
                None => {
                    // No URLs ready, wait a bit
                    tokio::time::sleep(self.config.scrape_interval).await;
                }
            }
        }

        tracing::info!("Scraping coordinator stopped");
    }

    /// Stop the scraping loop
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
    }

    /// Check if running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Get current statistics
    pub async fn stats(&self) -> ScrapingStats {
        let mut stats = self.stats.read().await.clone();
        let frontier = self.frontier.read().await;

        stats.queue_size = frontier.pending_count();
        stats.active_domains = frontier.domain_count();

        stats
    }

    /// Convert extracted content to a Document for indexing
    pub fn to_document(
        url: &Url,
        content: &ExtractedContent,
        metadata: &ExtractedMetadata,
    ) -> Document {
        let mut doc = Document::new(&content.text_content)
            .with_title(&content.title)
            .with_url(url.as_str());

        if let Some(author) = &metadata.author {
            doc.metadata.insert("author".to_string(), author.clone());
        }

        if let Some(date) = &metadata.published_date {
            doc.metadata.insert("published_date".to_string(), date.to_rfc3339());
        }

        if let Some(lang) = &metadata.language {
            doc.metadata.insert("language".to_string(), lang.clone());
        }

        doc.metadata.insert("word_count".to_string(), content.word_count.to_string());
        doc.metadata.insert("reading_time".to_string(), content.reading_time_minutes.to_string());
        doc.metadata.insert("domain".to_string(), metadata.domain.clone());

        doc
    }

    /// Handle a node joining the network
    pub async fn on_node_join(&self, peer_id: PeerId) {
        let mut assignment = self.domain_assignment.write().await;
        assignment.on_node_join(peer_id);

        // Update frontier with new assignment
        let mut frontier = self.frontier.write().await;
        frontier.update_domain_assignment(assignment.clone());
    }

    /// Handle a node leaving the network
    pub async fn on_node_leave(&self, peer_id: &PeerId) {
        let mut assignment = self.domain_assignment.write().await;
        assignment.on_node_leave(peer_id);

        // Update frontier with new assignment
        let mut frontier = self.frontier.write().await;
        frontier.update_domain_assignment(assignment.clone());
    }

    /// Get outbound URL batches for other nodes
    pub async fn get_outbound_batches(&self) -> Vec<super::frontier::UrlExchangeBatch> {
        let mut frontier = self.frontier.write().await;
        frontier.take_ready_batches()
    }

    /// Receive URL batch from another node
    pub async fn receive_url_batch(&self, batch: super::frontier::UrlExchangeBatch) {
        let mut stats = self.stats.write().await;
        stats.urls_exchanged += batch.len() as u64;
        drop(stats);

        let mut frontier = self.frontier.write().await;
        frontier.receive_batch(batch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = ScrapingConfig::default();
        let coordinator = ScrapingCoordinator::new(config, "test_peer".to_string());

        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_add_seeds() {
        let config = ScrapingConfig::default();
        let mut coordinator = ScrapingCoordinator::new(config, "test_peer".to_string()).unwrap();

        let seeds = vec![
            Url::parse("https://example.com").unwrap(),
            Url::parse("https://test.org").unwrap(),
        ];

        coordinator.add_seeds(seeds).await;

        let stats = coordinator.stats().await;
        assert!(stats.queue_size > 0);
    }

    #[tokio::test]
    async fn test_document_conversion() {
        let url = Url::parse("https://example.com/article").unwrap();

        let content = ExtractedContent {
            title: "Test Article".to_string(),
            text_content: "This is the article content.".to_string(),
            clean_html: None,
            author: Some("John Doe".to_string()),
            published_date: None,
            excerpt: None,
            language: Some("en".to_string()),
            word_count: 5,
            reading_time_minutes: 1,
        };

        let metadata = ExtractedMetadata {
            url: url.to_string(),
            canonical_url: None,
            title: "Test Article".to_string(),
            description: None,
            author: Some("John Doe".to_string()),
            published_date: None,
            modified_date: None,
            language: Some("en".to_string()),
            content_type: super::super::extractor::ContentType::Article,
            word_count: 5,
            reading_time_minutes: 1,
            domain: "example.com".to_string(),
            fetched_at: chrono::Utc::now(),
            extra: std::collections::HashMap::new(),
        };

        let doc = ScrapingCoordinator::to_document(&url, &content, &metadata);

        assert_eq!(doc.title, Some("Test Article".to_string()));
        assert_eq!(doc.url, Some("https://example.com/article".to_string()));
        assert_eq!(doc.metadata.get("author"), Some(&"John Doe".to_string()));
    }
}
