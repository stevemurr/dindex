//! URL Frontier with per-domain priority queues
//!
//! Manages URLs to be crawled with priority scoring based on:
//! - Depth from seed URLs
//! - Inlink count
//! - Domain authority
//! - Freshness hints
//! - Content type

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;
use url::Url;

use super::domain_assignment::{DomainAssignment, PeerId};

/// URL path patterns that indicate aggregator/listing pages rather than original content.
/// These are deprioritized since they typically duplicate content found at canonical URLs.
const AGGREGATOR_PATH_SEGMENTS: &[&str] = &[
    "/tag/",
    "/tags/",
    "/category/",
    "/categories/",
    "/archive/",
    "/archives/",
    "/author/",
    "/page/",
    "/label/",
    "/topic/",
];

/// Query parameter keys that indicate sorting/filtering views of the same content.
const AGGREGATOR_QUERY_KEYS: &[&str] = &[
    "sort", "order", "orderby", "filter", "view", "page", "p",
];

/// A URL with its priority score and metadata
#[derive(Debug, Clone)]
pub struct ScoredUrl {
    /// The URL to crawl
    pub url: Url,
    /// Priority score (higher = crawl sooner)
    pub priority: f32,
    /// Hops from seed URL
    pub depth: u8,
    /// When this URL was discovered
    pub discovered_at: Instant,
    /// Number of pages linking to this URL (observed so far)
    pub inlink_count: u32,
    /// Whether this appears to be time-sensitive content
    pub is_fresh_content: bool,
    /// Whether this URL looks like an aggregator/listing page
    pub is_aggregator: bool,
}

impl ScoredUrl {
    /// Create a new scored URL with default values
    pub fn new(url: Url) -> Self {
        let is_aggregator = Self::detect_aggregator(&url);
        Self {
            url,
            priority: 0.0,
            depth: 0,
            discovered_at: Instant::now(),
            inlink_count: 0,
            is_fresh_content: false,
            is_aggregator,
        }
    }

    /// Create a new scored URL with specified depth
    pub fn with_depth(url: Url, depth: u8) -> Self {
        let mut scored = Self::new(url);
        scored.depth = depth;
        scored.recalculate_priority();
        scored
    }

    /// Recalculate priority based on current values
    pub fn recalculate_priority(&mut self) {
        // Priority weights from architecture spec:
        // - Depth from seed: -0.1 per hop
        // - Inlink count: +0.2 per inlink
        // - Freshness hint: +0.5 if news/blog
        // - Content type: +0.2 for HTML (assumed here)
        // - Aggregator penalty: -0.5 for tag/category/archive pages

        self.priority = 1.0; // Base priority
        self.priority -= 0.1 * self.depth as f32; // Breadth-first bias
        self.priority += 0.2 * (self.inlink_count.min(10) as f32); // Cap inlink boost
        if self.is_fresh_content {
            self.priority += 0.5;
        }
        self.priority += 0.2; // Assume HTML content
        if self.is_aggregator {
            self.priority -= 0.5; // Deprioritize aggregator pages
        }
    }

    /// Detect whether a URL looks like an aggregator/listing page
    fn detect_aggregator(url: &Url) -> bool {
        let path = url.path().to_lowercase();

        // Check path segments
        if AGGREGATOR_PATH_SEGMENTS.iter().any(|seg| path.contains(seg)) {
            return true;
        }

        // Check query parameters for sort/filter/pagination
        if let Some(query) = url.query() {
            for param in query.split('&') {
                let key = param.split('=').next().unwrap_or("");
                let key_lower = key.to_lowercase();
                if AGGREGATOR_QUERY_KEYS.contains(&key_lower.as_str()) {
                    return true;
                }
            }
        }

        false
    }

    /// Increment inlink count and recalculate priority
    pub fn add_inlink(&mut self) {
        self.inlink_count = self.inlink_count.saturating_add(1);
        self.recalculate_priority();
    }
}

impl PartialEq for ScoredUrl {
    fn eq(&self, other: &Self) -> bool {
        self.url == other.url
    }
}

impl Eq for ScoredUrl {}

impl PartialOrd for ScoredUrl {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredUrl {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then older discovery time
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.discovered_at.cmp(&self.discovered_at))
    }
}

/// Batch of URLs to exchange with other nodes
#[derive(Debug, Clone)]
pub struct UrlExchangeBatch {
    /// The node sending these URLs
    pub from_node: PeerId,
    /// URLs grouped by target hostname
    pub urls: Vec<(String, ScoredUrl)>, // (hostname, url)
    /// When this batch was created
    pub timestamp: Instant,
}

impl UrlExchangeBatch {
    pub fn new(from_node: PeerId) -> Self {
        Self {
            from_node,
            urls: Vec::new(),
            timestamp: Instant::now(),
        }
    }

    pub fn add(&mut self, hostname: String, url: ScoredUrl) {
        self.urls.push((hostname, url));
    }

    pub fn len(&self) -> usize {
        self.urls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.urls.is_empty()
    }
}

/// Entry in the cross-domain priority heap representing the best URL from a domain
#[derive(Debug, Clone)]
struct DomainTopEntry {
    domain: String,
    best_url: ScoredUrl,
}

impl PartialEq for DomainTopEntry {
    fn eq(&self, other: &Self) -> bool {
        self.domain == other.domain
    }
}

impl Eq for DomainTopEntry {}

impl PartialOrd for DomainTopEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DomainTopEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.best_url.cmp(&other.best_url)
    }
}

/// URL Frontier managing per-domain priority queues
pub struct UrlFrontier {
    /// Per-domain priority heaps, keyed by hostname
    domain_queues: HashMap<String, BinaryHeap<ScoredUrl>>,

    /// Cross-domain priority heap for O(log D) pop_next
    cross_domain_heap: BinaryHeap<DomainTopEntry>,

    /// Domains whose cross-domain entry needs refreshing
    stale_domains: HashSet<String>,

    /// Global seen-URL filter using hashed URLs to save memory
    seen_urls: HashSet<u64>,

    /// Domain assignment for routing
    domain_assignment: DomainAssignment,

    /// Local peer ID
    local_peer_id: PeerId,

    /// Outbound URL batches for other nodes
    outbound_batches: HashMap<PeerId, UrlExchangeBatch>,

    /// Maximum URLs per domain queue
    max_queue_size: usize,

    /// Batch size threshold for sending
    batch_send_threshold: usize,
}

impl UrlFrontier {
    /// Create a new URL frontier
    pub fn new(local_peer_id: PeerId, domain_assignment: DomainAssignment) -> Self {
        Self {
            domain_queues: HashMap::new(),
            cross_domain_heap: BinaryHeap::new(),
            stale_domains: HashSet::new(),
            seen_urls: HashSet::new(),
            domain_assignment,
            local_peer_id,
            outbound_batches: HashMap::new(),
            max_queue_size: 10000,
            batch_send_threshold: 1000,
        }
    }

    /// Hash a normalized URL string to u64
    fn hash_url(normalized: &str) -> u64 {
        xxhash_rust::xxh3::xxh3_64(normalized.as_bytes())
    }

    /// Add a discovered URL to the appropriate queue
    pub fn add_discovered_url(&mut self, url: Url, depth: u8) -> UrlDestination {
        // Normalize and check if seen
        let normalized = Self::normalize_url(&url);
        let hash = Self::hash_url(&normalized);
        if self.seen_urls.contains(&hash) {
            return UrlDestination::AlreadySeen;
        }

        // Mark as seen
        self.seen_urls.insert(hash);

        let hostname = url.host_str().unwrap_or_default().to_string();
        let scored_url = ScoredUrl::with_depth(url, depth);

        // Check domain assignment
        let assigned_node = self.domain_assignment.assign_domain(&hostname);

        match assigned_node {
            Some(ref peer) if peer == &self.local_peer_id => {
                // Local domain - add to our queue
                self.add_local_url(&hostname, scored_url);
                UrlDestination::Local
            }
            Some(peer) => {
                // Remote domain - batch for exchange
                let peer_clone = peer.clone();
                self.add_remote_url(peer, hostname, scored_url);
                UrlDestination::Remote(peer_clone)
            }
            None => {
                // No nodes in ring - add locally as fallback
                self.add_local_url(&hostname, scored_url);
                UrlDestination::Local
            }
        }
    }

    /// Add a URL to the local queue for a domain
    fn add_local_url(&mut self, hostname: &str, url: ScoredUrl) {
        let domain = hostname.to_string();
        let queue = self
            .domain_queues
            .entry(domain.clone())
            .or_default();

        // Enforce max queue size
        if queue.len() < self.max_queue_size {
            queue.push(url);
            // Mark domain as needing cross-domain heap refresh
            self.stale_domains.insert(domain);
        }
    }

    /// Rebuild stale entries in the cross-domain heap
    fn refresh_cross_domain_heap(&mut self) {
        if self.stale_domains.is_empty() {
            return;
        }

        for domain in self.stale_domains.drain() {
            if let Some(queue) = self.domain_queues.get(&domain) {
                if let Some(top) = queue.peek() {
                    self.cross_domain_heap.push(DomainTopEntry {
                        domain,
                        best_url: top.clone(),
                    });
                }
            }
        }
    }

    /// Add a URL to the outbound batch for a remote node
    fn add_remote_url(&mut self, peer_id: PeerId, hostname: String, url: ScoredUrl) {
        let batch = self
            .outbound_batches
            .entry(peer_id.clone())
            .or_insert_with(|| UrlExchangeBatch::new(self.local_peer_id.clone()));

        batch.add(hostname, url);
    }

    /// Get the next URL to crawl, respecting domain politeness.
    ///
    /// `ready_domains` contains domains tracked by the politeness controller that
    /// are currently eligible for fetching. `tracked_domains` contains all domains
    /// the politeness controller knows about (fetched at least once). Domains in
    /// the frontier that are NOT in `tracked_domains` have never been fetched and
    /// are implicitly ready.
    ///
    /// Uses a cross-domain priority heap for O(log D) amortized performance
    /// instead of scanning all domain queues.
    pub fn pop_next(
        &mut self,
        ready_domains: &HashSet<String>,
        tracked_domains: &HashSet<String>,
    ) -> Option<ScoredUrl> {
        // Refresh any stale entries first
        self.refresh_cross_domain_heap();

        // Entries we skip because their domain isn't ready — re-insert after
        let mut deferred = Vec::new();

        let result = loop {
            let entry = match self.cross_domain_heap.pop() {
                Some(e) => e,
                None => break None,
            };

            let domain = &entry.domain;

            // Check if domain is eligible
            let is_tracked = tracked_domains.contains(domain);
            let is_ready = ready_domains.contains(domain);

            if is_tracked && !is_ready {
                // Not ready — defer and try next
                deferred.push(entry);
                continue;
            }

            // Check if the domain queue still has this URL (it may have been
            // consumed by a concurrent stale entry)
            let queue = match self.domain_queues.get_mut(domain) {
                Some(q) => q,
                None => continue,
            };

            // The cross-domain heap may have stale entries. Verify the top of
            // the domain queue still matches (or is at least present).
            if queue.is_empty() {
                continue;
            }

            let url = queue.pop().unwrap();

            // Re-insert next-best from this domain into cross-domain heap
            if let Some(next_top) = queue.peek() {
                self.cross_domain_heap.push(DomainTopEntry {
                    domain: entry.domain,
                    best_url: next_top.clone(),
                });
            }

            break Some(url);
        };

        // Put back deferred entries
        for entry in deferred {
            self.cross_domain_heap.push(entry);
        }

        result
    }

    /// Get outbound batches that are ready to send
    pub fn take_ready_batches(&mut self) -> Vec<UrlExchangeBatch> {
        let ready_peers: Vec<_> = self
            .outbound_batches
            .iter()
            .filter(|(_, batch)| batch.len() >= self.batch_send_threshold)
            .map(|(peer, _)| peer.clone())
            .collect();

        ready_peers
            .into_iter()
            .filter_map(|peer| self.outbound_batches.remove(&peer))
            .collect()
    }

    /// Receive URLs from another node
    pub fn receive_batch(&mut self, batch: UrlExchangeBatch) {
        for (hostname, scored_url) in batch.urls {
            // Add to local queue (we should own these domains now)
            self.add_local_url(&hostname, scored_url);
        }
    }

    /// Get total number of pending URLs
    pub fn pending_count(&self) -> usize {
        self.domain_queues.values().map(|q| q.len()).sum()
    }

    /// Get number of domains being crawled
    pub fn domain_count(&self) -> usize {
        self.domain_queues.len()
    }

    /// Normalize a URL for deduplication
    fn normalize_url(url: &Url) -> String {
        super::normalize_url(url)
    }

    /// Add seed URLs
    pub fn add_seeds(&mut self, seeds: Vec<Url>) {
        for url in seeds {
            self.add_discovered_url(url, 0);
        }
    }

    /// Update domain assignment (when network topology changes)
    pub fn update_domain_assignment(&mut self, assignment: DomainAssignment) {
        self.domain_assignment = assignment;
    }
}

/// Destination for a discovered URL
#[derive(Debug, Clone, PartialEq)]
pub enum UrlDestination {
    /// URL is assigned to this node
    Local,
    /// URL is assigned to a remote node
    Remote(PeerId),
    /// URL was already seen
    AlreadySeen,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frontier() -> UrlFrontier {
        let mut assignment = DomainAssignment::new(10);
        assignment.on_node_join("local".to_string());
        assignment.set_local_peer("local".to_string());
        UrlFrontier::new("local".to_string(), assignment)
    }

    #[test]
    fn test_url_priority_ordering() {
        let mut url1 = ScoredUrl::new(Url::parse("https://example.com/a").unwrap());
        url1.priority = 1.0;

        let mut url2 = ScoredUrl::new(Url::parse("https://example.com/b").unwrap());
        url2.priority = 2.0;

        // Higher priority should come first in max-heap
        assert!(url2 > url1);
    }

    #[test]
    fn test_add_and_pop() {
        let mut frontier = create_test_frontier();

        frontier.add_discovered_url(Url::parse("https://example.com/page1").unwrap(), 0);
        frontier.add_discovered_url(Url::parse("https://example.com/page2").unwrap(), 1);

        assert!(frontier.pending_count() > 0);

        let ready: HashSet<_> = vec!["example.com".to_string()].into_iter().collect();
        let tracked = ready.clone();
        let next = frontier.pop_next(&ready, &tracked);

        assert!(next.is_some());
    }

    #[test]
    fn test_dedup() {
        let mut frontier = create_test_frontier();

        let url = Url::parse("https://example.com/page").unwrap();

        let first = frontier.add_discovered_url(url.clone(), 0);
        let second = frontier.add_discovered_url(url, 0);

        assert_eq!(first, UrlDestination::Local);
        assert_eq!(second, UrlDestination::AlreadySeen);
    }

    #[test]
    fn test_url_normalization() {
        let url1 = Url::parse("https://example.com/page#section").unwrap();
        let url2 = Url::parse("https://example.com/page").unwrap();

        let norm1 = UrlFrontier::normalize_url(&url1);
        let norm2 = UrlFrontier::normalize_url(&url2);

        // Fragment should be stripped
        assert_eq!(norm1, norm2);
    }
}
