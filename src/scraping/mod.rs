//! Web scraping subsystem for the decentralized semantic search index
//!
//! This module implements a distributed, polite, LLM-optimized web scraping system
//! that feeds content into the decentralized semantic search index. The architecture
//! uses consistent hashing to partition domains across nodes, with gossip protocols
//! for URL exchange and SimHash for content deduplication.
//!
//! Key components:
//! - `DomainAssignment`: Consistent hashing for domain-to-node mapping
//! - `UrlFrontier`: Per-domain priority queues with URL scoring
//! - `PolitenessController`: robots.txt handling and rate limiting
//! - `FetchEngine`: HTTP + headless browser fetching
//! - `ContentExtractor`: HTML to clean text extraction
//! - `Deduplicator`: URL Bloom filter + SimHash content deduplication
//! - `ScrapingCoordinator`: Orchestrates the entire scraping pipeline

pub mod coordinator;
pub mod dedup;
pub mod domain_assignment;
pub mod extractor;
pub mod fetcher;
pub mod frontier;
pub mod politeness;
pub mod trap_detection;

pub use coordinator::ScrapingCoordinator;
pub use dedup::{ContentDeduplicator, UrlDeduplicator};
pub use domain_assignment::DomainAssignment;
pub use extractor::ContentExtractor;
pub use fetcher::FetchEngine;
pub use frontier::{ScoredUrl, UrlFrontier};
pub use politeness::{FetchDecision, PolitenessController};

/// Tracking/session query parameters to strip during normalization
const TRACKING_PARAMS: &[&str] = &[
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "sid",
    "sessionid",
    "ref",
    "source",
];

/// Normalize a URL for deduplication
///
/// - Strips fragments
/// - Removes `www.` prefix from hostnames
/// - Removes trailing slashes from non-root paths
/// - Strips tracking/session query parameters
/// - Sorts remaining query parameters
/// - Lowercases the result
pub(crate) fn normalize_url(url: &url::Url) -> String {
    let mut normalized = url.clone();

    // Remove fragment
    normalized.set_fragment(None);

    // Strip www. prefix from hostname
    if let Some(host) = normalized.host_str().map(|h| h.to_string()) {
        if let Some(stripped) = host.strip_prefix("www.") {
            if let Err(e) = normalized.set_host(Some(stripped)) {
                tracing::warn!("Failed to strip www. from {}: {}", host, e);
            }
        }
    }

    // Remove trailing slash from non-root paths
    let path = normalized.path().to_string();
    if path.len() > 1 && path.ends_with('/') {
        normalized.set_path(&path[..path.len() - 1]);
    }

    // Filter out tracking parameters and sort remaining ones
    if let Some(query) = normalized.query() {
        let params: Vec<_> = query
            .split('&')
            .filter(|p| {
                let key = p.split('=').next().unwrap_or("");
                let key_lower = key.to_lowercase();
                !TRACKING_PARAMS.contains(&key_lower.as_str())
            })
            .collect();

        if params.is_empty() {
            normalized.set_query(None);
        } else {
            let mut sorted_params = params;
            sorted_params.sort();
            normalized.set_query(Some(&sorted_params.join("&")));
        }
    }

    normalized.as_str().to_lowercase()
}
