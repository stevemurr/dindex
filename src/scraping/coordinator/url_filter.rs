//! URL filtering based on domain, patterns, and crawl trap detection

use std::collections::HashSet;
use url::Url;

use crate::scraping::{
    fetcher::{extract_urls, FetchResult},
    trap_detection,
};

use super::types::ScrapingConfig;

/// Filter discovered URLs based on coordinator configuration
pub(super) fn filter_discovered_urls(
    response: &FetchResult,
    config: &ScrapingConfig,
    seed_domains: &HashSet<String>,
    compiled_include: &[regex::Regex],
    compiled_exclude: &[regex::Regex],
) -> Vec<Url> {
    let all_urls = extract_urls(response);

    all_urls
        .into_iter()
        .filter(|url| {
            // Check crawl trap detection
            if trap_detection::is_crawl_trap(url, &config.trap_detector) {
                return false;
            }

            // Check stay_on_domain
            if config.stay_on_domain {
                let domain = url.host_str().unwrap_or_default();
                if !seed_domains.contains(domain) {
                    return false;
                }
            }

            // Check exclude patterns (compiled regex)
            let url_str = url.as_str();
            for pattern in compiled_exclude {
                if pattern.is_match(url_str) {
                    return false;
                }
            }

            // Check include patterns (if any, compiled regex)
            if !compiled_include.is_empty() {
                if !compiled_include.iter().any(|p| p.is_match(url_str)) {
                    return false;
                }
            }

            true
        })
        .collect()
}
