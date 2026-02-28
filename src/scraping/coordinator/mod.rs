//! Scraping coordinator orchestrating the entire scraping pipeline
//!
//! Coordinates the flow from URL discovery through fetching, extraction,
//! deduplication, and indexing. Manages the scraping lifecycle and integrates
//! with the P2P network for distributed crawling.

mod pipeline;
mod types;
mod url_filter;

pub use types::*;

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, Semaphore};
use url::Url;

use super::{
    dedup::{ContentDeduplicator, SimHash},
    domain_assignment::{DomainAssignment, PeerId},
    extractor::{ContentExtractor, ExtractedContent, ExtractedMetadata},
    fetcher::{extract_urls, FetchEngine, FetchError},
    frontier::{ScoredUrl, UrlFrontier},
    politeness::{FetchDecision, PolitenessController},
};

use crate::index::{DocumentRegistry, DuplicateCheckResult};
use crate::types::{Document, DocumentIdentity};

/// Scraping coordinator managing the entire scraping pipeline
pub struct ScrapingCoordinator {
    /// Configuration
    config: ScrapingConfig,
    /// Local peer ID
    _local_peer_id: PeerId,
    /// Domain assignment
    domain_assignment: Arc<RwLock<DomainAssignment>>,
    /// URL frontier
    frontier: Arc<RwLock<UrlFrontier>>,
    /// Politeness controller
    politeness: Arc<RwLock<PolitenessController>>,
    /// Fetch engine (lock-free — uses internal atomics for stats)
    fetcher: Arc<FetchEngine>,
    /// Content extractor (Arc for spawn_blocking sharing)
    extractor: Arc<ContentExtractor>,
    /// Content deduplicator (legacy, used if no registry is provided)
    content_dedup: Arc<RwLock<ContentDeduplicator>>,
    /// Document registry for unified deduplication (optional)
    document_registry: Option<Arc<DocumentRegistry>>,
    /// Statistics
    stats: Arc<RwLock<ScrapingStats>>,
    /// Seed domains (for stay_on_domain mode)
    seed_domains: HashSet<String>,
    /// Running flag (lock-free)
    running: Arc<AtomicBool>,
    /// Pre-compiled include patterns
    compiled_include: Vec<regex::Regex>,
    /// Pre-compiled exclude patterns
    compiled_exclude: Vec<regex::Regex>,
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
        let extractor = Arc::new(ContentExtractor::new(config.extractor.clone()));

        let content_dedup = ContentDeduplicator::new(
            CONTENT_DEDUP_CAPACITY,
            CONTENT_DEDUP_HAMMING_THRESHOLD,
            local_peer_id.clone(),
        );

        // Pre-compile include/exclude regex patterns
        let compiled_include: Vec<regex::Regex> = config
            .include_patterns
            .iter()
            .filter_map(|p| match regex::Regex::new(p) {
                Ok(r) => Some(r),
                Err(e) => {
                    tracing::warn!("Invalid include pattern '{}': {}", p, e);
                    None
                }
            })
            .collect();

        let compiled_exclude: Vec<regex::Regex> = config
            .exclude_patterns
            .iter()
            .filter_map(|p| match regex::Regex::new(p) {
                Ok(r) => Some(r),
                Err(e) => {
                    tracing::warn!("Invalid exclude pattern '{}': {}", p, e);
                    None
                }
            })
            .collect();

        Ok(Self {
            config,
            _local_peer_id: local_peer_id,
            domain_assignment,
            frontier: Arc::new(RwLock::new(frontier)),
            politeness: Arc::new(RwLock::new(politeness)),
            fetcher: Arc::new(fetcher),
            extractor,
            content_dedup: Arc::new(RwLock::new(content_dedup)),
            document_registry: None,
            stats: Arc::new(RwLock::new(ScrapingStats::default())),
            seed_domains: HashSet::new(),
            running: Arc::new(AtomicBool::new(false)),
            compiled_include,
            compiled_exclude,
        })
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

        // Check politeness
        let fetch_decision = {
            let mut politeness = self.politeness.write().await;
            politeness.can_fetch(url).await
        };

        match fetch_decision {
            FetchDecision::Disallowed => {
                return ProcessResult::failure(url, "Disallowed by robots.txt", start);
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

        // Fetch the URL (no lock needed — FetchEngine uses internal atomics)
        let fetch_result = self.fetcher.fetch(url).await;

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
                return ProcessResult::failure(url, "Rate limited (429)", start);
            }
            Err(e) => {
                let mut politeness = self.politeness.write().await;
                politeness.record_error(hostname);
                return ProcessResult::failure(url, format!("Fetch error: {}", e), start);
            }
        };

        // Extract content and metadata on a blocking thread to avoid
        // starving the tokio runtime with CPU-bound HTML parsing.
        // Uses extract_all() to parse the DOM once for both content and metadata.
        let extractor = Arc::clone(&self.extractor);
        let body_clone = response.body.clone();
        let url_clone = url.clone();
        let extraction = tokio::task::spawn_blocking(move || {
            extractor.extract_all(&body_clone, &url_clone)
        })
        .await;

        let (content, metadata) = match extraction {
            Ok(Ok(pair)) => pair,
            Ok(Err(e)) => {
                return ProcessResult {
                    url: url.clone(),
                    outcome: ProcessOutcome::Failure { error: format!("Extraction error: {}", e) },
                    discovered_urls: extract_urls(&response),
                    duration: start.elapsed(),
                };
            }
            Err(e) => {
                return ProcessResult {
                    url: url.clone(),
                    outcome: ProcessOutcome::Failure { error: format!("Extraction task failed: {}", e) },
                    discovered_urls: extract_urls(&response),
                    duration: start.elapsed(),
                };
            }
        };

        // If a canonical URL differs from the fetched URL, mark the canonical as
        // seen in the frontier so that future encounters are deduplicated.
        if let Some(ref canonical) = metadata.canonical_url {
            if let Ok(canonical_parsed) = Url::parse(canonical) {
                let canonical_normalized = super::normalize_url(&canonical_parsed);
                let fetched_normalized = super::normalize_url(url);
                if canonical_normalized != fetched_normalized {
                    let mut frontier = self.frontier.write().await;
                    frontier.mark_url_seen(&canonical_parsed);
                }
            }
        }

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

                    return ProcessResult::failure(url, format!("Duplicate of {}", entry.content_id), start);
                }
                DuplicateCheckResult::NearDuplicate { entry, hamming_distance } => {
                    // Near-duplicate - update URL mapping and skip
                    registry.update_metadata(&entry.content_id, Some(url.to_string()), None);

                    let mut stats = self.stats.write().await;
                    stats.duplicates_skipped += 1;

                    return ProcessResult::failure(url, format!("Near-duplicate of {} (distance: {})", entry.content_id, hamming_distance), start);
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

                return ProcessResult::failure(url, format!("Duplicate of {}", existing_doc), start);
            }

            // Register new content
            let doc_id = format!("{}:{}", hostname, uuid::Uuid::new_v4());
            content_dedup.register(&content.text_content, doc_id)
        };

        // Extract URLs for further crawling
        let discovered_urls = url_filter::filter_discovered_urls(
            &response,
            url,
            &self.config,
            &self.seed_domains,
            &self.compiled_include,
            &self.compiled_exclude,
        );

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
            outcome: ProcessOutcome::Success(ProcessedContent {
                content,
                metadata,
                simhash,
            }),
            discovered_urls,
            duration: start.elapsed(),
        }
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
        let tracked_domains = politeness.tracked_domains();
        drop(politeness);

        let mut frontier = self.frontier.write().await;
        frontier.pop_next(&ready_domains, &tracked_domains)
    }

    /// Run the scraping loop with concurrent fetching.
    ///
    /// Uses a semaphore to limit concurrency to `max_concurrent_fetches` while
    /// maintaining per-domain politeness through the frontier and politeness controller.
    pub async fn run(self: &Arc<Self>) {
        self.running.store(true, Ordering::Relaxed);

        let max_concurrent = self.config.max_concurrent_fetches.max(1);
        let semaphore = Arc::new(Semaphore::new(max_concurrent));

        tracing::info!(
            "Starting scraping coordinator (max_concurrent_fetches={})",
            max_concurrent
        );

        while self.running.load(Ordering::Relaxed) {
            // Try to acquire a permit (non-blocking check first)
            let permit = match semaphore.clone().try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => {
                    // All slots full, wait for one to free up
                    match semaphore.clone().acquire_owned().await {
                        Ok(permit) => permit,
                        Err(_) => break, // Semaphore closed
                    }
                }
            };

            // Get next URL
            let next_url = self.get_next_url().await;

            match next_url {
                Some(scored_url) => {
                    let coordinator = Arc::clone(self);
                    tokio::spawn(async move {
                        let _permit = permit; // Held until task completes

                        let result = coordinator.process_url(&scored_url.url).await;

                        match &result.outcome {
                            ProcessOutcome::Success(processed) => {
                                tracing::debug!(
                                    "Processed {} - {} words, {} URLs discovered",
                                    result.url,
                                    processed.content.word_count,
                                    result.discovered_urls.len()
                                );

                                coordinator
                                    .add_discovered_urls(result.discovered_urls, scored_url.depth)
                                    .await;
                            }
                            ProcessOutcome::Failure { error } => {
                                tracing::debug!(
                                    "Failed to process {}: {}",
                                    result.url,
                                    error
                                );
                            }
                        }
                    });
                }
                None => {
                    // No URLs ready, release permit and wait
                    drop(permit);
                    tokio::time::sleep(self.config.scrape_interval).await;
                }
            }
        }

        tracing::info!("Scraping coordinator stopped");
    }

    /// Stop the scraping loop
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
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
        pipeline::to_document(url, content, metadata)
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
    use crate::scraping::politeness::PolitenessConfig;

    fn default_coordinator() -> ScrapingCoordinator {
        let config = ScrapingConfig::default();
        ScrapingCoordinator::new(config, "test_peer".to_string()).unwrap()
    }

    #[tokio::test]
    async fn test_coordinator_creation() {
        let config = ScrapingConfig::default();
        let coordinator = ScrapingCoordinator::new(config, "test_peer".to_string());

        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_coordinator_creation_with_custom_config() {
        let config = ScrapingConfig {
            max_concurrent_fetches: 5,
            max_depth: 10,
            stay_on_domain: true,
            ..ScrapingConfig::default()
        };
        let coordinator = ScrapingCoordinator::new(config, "custom_peer".to_string());
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
    async fn test_add_seeds_tracks_domains() {
        let mut coordinator = default_coordinator();

        let seeds = vec![
            Url::parse("https://foo.com/page1").unwrap(),
            Url::parse("https://bar.org/page2").unwrap(),
            Url::parse("https://foo.com/page3").unwrap(), // duplicate domain
        ];

        coordinator.add_seeds(seeds).await;

        // The coordinator should have tracked the seed domains
        assert!(coordinator.seed_domains.contains("foo.com"));
        assert!(coordinator.seed_domains.contains("bar.org"));
        assert_eq!(coordinator.seed_domains.len(), 2);
    }

    #[tokio::test]
    async fn test_add_seeds_empty_list() {
        let mut coordinator = default_coordinator();
        coordinator.add_seeds(vec![]).await;

        let stats = coordinator.stats().await;
        assert_eq!(stats.queue_size, 0);
    }

    #[tokio::test]
    async fn test_stats_default_values() {
        let coordinator = default_coordinator();

        let stats = coordinator.stats().await;
        assert_eq!(stats.urls_processed, 0);
        assert_eq!(stats.successful_fetches, 0);
        assert_eq!(stats.failed_fetches, 0);
        assert_eq!(stats.duplicates_skipped, 0);
        assert_eq!(stats.urls_discovered, 0);
        assert_eq!(stats.urls_exchanged, 0);
        assert_eq!(stats.queue_size, 0);
        assert_eq!(stats.active_domains, 0);
        assert!((stats.avg_processing_time_ms - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_stats_reflects_queue_size() {
        let mut coordinator = default_coordinator();

        coordinator
            .add_seeds(vec![
                Url::parse("https://example.com/a").unwrap(),
                Url::parse("https://example.com/b").unwrap(),
                Url::parse("https://other.com/c").unwrap(),
            ])
            .await;

        let stats = coordinator.stats().await;
        assert!(stats.queue_size >= 3, "Queue should have at least 3 URLs");
        assert!(stats.active_domains >= 1, "Should have at least 1 active domain");
    }

    #[tokio::test]
    async fn test_is_running_initially_false() {
        let coordinator = default_coordinator();
        assert!(!coordinator.is_running());
    }

    #[tokio::test]
    async fn test_stop_sets_not_running() {
        let coordinator = default_coordinator();

        // Manually set running to true
        coordinator.running.store(true, Ordering::Relaxed);
        assert!(coordinator.is_running());

        // Stop should set it back to false
        coordinator.stop();
        assert!(!coordinator.is_running());
    }

    #[tokio::test]
    async fn test_get_next_url_empty_frontier() {
        let coordinator = default_coordinator();

        // With no seeds added, get_next_url should return None
        let next = coordinator.get_next_url().await;
        assert!(next.is_none());
    }

    #[tokio::test]
    async fn test_get_next_url_returns_seeded_url() {
        let mut coordinator = default_coordinator();

        coordinator
            .add_seeds(vec![Url::parse("https://example.com/page").unwrap()])
            .await;

        // The politeness controller's ready_domains() only returns domains
        // that have entries in domain_state. We need to register the domain
        // by recording a success so the domain appears as "ready".
        {
            let mut politeness = coordinator.politeness.write().await;
            politeness.record_success("example.com");
        }
        // Wait for the default delay to expire so the domain is ready
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;

        let next = coordinator.get_next_url().await;
        assert!(next.is_some(), "Should return a URL from the frontier");
        let scored_url = next.unwrap();
        assert_eq!(
            scored_url.url.host_str(),
            Some("example.com")
        );
    }

    #[tokio::test]
    async fn test_get_next_url_depletes_queue() {
        let config = ScrapingConfig {
            politeness: PolitenessConfig {
                default_delay: std::time::Duration::from_millis(10),
                min_delay: std::time::Duration::from_millis(5),
                ..PolitenessConfig::default()
            },
            ..ScrapingConfig::default()
        };
        let mut coordinator =
            ScrapingCoordinator::new(config, "test_peer".to_string()).unwrap();

        coordinator
            .add_seeds(vec![Url::parse("https://example.com/only").unwrap()])
            .await;

        // Register the domain in politeness state so ready_domains works
        {
            let mut politeness = coordinator.politeness.write().await;
            // Set last_fetch to far in the past so it's immediately ready
            politeness.record_success("example.com");
        }
        // Wait for the short delay to expire
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        // Pop the only URL
        let first = coordinator.get_next_url().await;
        assert!(first.is_some());

        // Queue should now be empty
        let second = coordinator.get_next_url().await;
        assert!(second.is_none(), "Queue should be empty after popping the only URL");
    }

    #[tokio::test]
    async fn test_add_discovered_urls_respects_max_depth() {
        let config = ScrapingConfig {
            max_depth: 2,
            ..ScrapingConfig::default()
        };
        let coordinator = ScrapingCoordinator::new(config, "test_peer".to_string()).unwrap();

        // Adding URLs at depth >= max_depth should be a no-op
        coordinator
            .add_discovered_urls(
                vec![Url::parse("https://example.com/deep").unwrap()],
                2, // depth=2, max_depth=2, so depth+1=3 >= max_depth, should skip
            )
            .await;

        let stats = coordinator.stats().await;
        assert_eq!(stats.queue_size, 0, "URLs at max_depth should not be added");
    }

    #[tokio::test]
    async fn test_add_discovered_urls_within_depth() {
        let config = ScrapingConfig {
            max_depth: 3,
            ..ScrapingConfig::default()
        };
        let mut coordinator =
            ScrapingCoordinator::new(config, "test_peer".to_string()).unwrap();

        // Need to seed first so the domain is known to the domain assignment
        coordinator
            .add_seeds(vec![Url::parse("https://example.com").unwrap()])
            .await;
        // Pop the seed so frontier is clear for the discovered URL test
        let _ = coordinator.get_next_url().await;

        coordinator
            .add_discovered_urls(
                vec![Url::parse("https://example.com/page2").unwrap()],
                1, // depth 1, max_depth 3 -> depth+1=2 < 3, should be added
            )
            .await;

        let stats = coordinator.stats().await;
        assert!(stats.queue_size > 0, "URL within depth limit should be added");
    }

    #[tokio::test]
    async fn test_frontier_deduplicates_urls() {
        let mut coordinator = default_coordinator();

        let url = Url::parse("https://example.com/dup").unwrap();

        // Add as seed — first time should succeed
        coordinator.add_seeds(vec![url.clone()]).await;
        let stats = coordinator.stats().await;
        assert!(stats.queue_size > 0, "First add should succeed");

        // Adding the same URL as a discovered URL should be deduped by the frontier
        coordinator.add_discovered_urls(vec![url], 0).await;
        // Queue size should not increase (URL already seen by frontier)
        let stats2 = coordinator.stats().await;
        assert_eq!(
            stats.queue_size, stats2.queue_size,
            "Duplicate URL should be rejected by frontier"
        );
    }

    #[tokio::test]
    async fn test_on_node_join_and_leave() {
        let coordinator = default_coordinator();

        // Should not panic
        coordinator.on_node_join("peer2".to_string()).await;
        coordinator.on_node_leave(&"peer2".to_string()).await;
    }

    #[tokio::test]
    async fn test_receive_url_batch_updates_stats() {
        let coordinator = default_coordinator();

        let mut batch = super::super::frontier::UrlExchangeBatch::new("remote_peer".to_string());
        batch.add(
            "example.com".to_string(),
            ScoredUrl::new(Url::parse("https://example.com/remote").unwrap()),
        );

        coordinator.receive_url_batch(batch).await;

        let stats = coordinator.stats().await;
        assert_eq!(stats.urls_exchanged, 1);
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
        assert_eq!(doc.metadata.get("language"), Some(&"en".to_string()));
        assert_eq!(doc.metadata.get("word_count"), Some(&"5".to_string()));
        assert_eq!(doc.metadata.get("domain"), Some(&"example.com".to_string()));
    }

    #[tokio::test]
    async fn test_document_conversion_without_optional_fields() {
        let url = Url::parse("https://example.com/bare").unwrap();

        let content = ExtractedContent {
            title: "Bare Article".to_string(),
            text_content: "Minimal content.".to_string(),
            clean_html: None,
            author: None,
            published_date: None,
            excerpt: None,
            language: None,
            word_count: 2,
            reading_time_minutes: 1,
        };

        let metadata = ExtractedMetadata {
            url: url.to_string(),
            canonical_url: None,
            title: "Bare Article".to_string(),
            description: None,
            author: None,
            published_date: None,
            modified_date: None,
            language: None,
            content_type: super::super::extractor::ContentType::Article,
            word_count: 2,
            reading_time_minutes: 1,
            domain: "example.com".to_string(),
            fetched_at: chrono::Utc::now(),
            extra: std::collections::HashMap::new(),
        };

        let doc = ScrapingCoordinator::to_document(&url, &content, &metadata);

        assert_eq!(doc.title, Some("Bare Article".to_string()));
        // Optional fields should not be in metadata
        assert!(doc.metadata.get("author").is_none());
        assert!(doc.metadata.get("language").is_none());
    }
}
