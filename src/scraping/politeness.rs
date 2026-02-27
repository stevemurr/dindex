//! Politeness controller for web scraping
//!
//! Handles robots.txt parsing, per-domain rate limiting, and 429 backoff.
//! Ensures the scraper is a good citizen of the web.

use lru::LruCache;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};
use url::Url;

/// Decision about whether a fetch is allowed
#[derive(Debug, Clone, PartialEq)]
pub enum FetchDecision {
    /// Fetch is allowed
    Allowed,
    /// URL is disallowed by robots.txt
    Disallowed,
    /// Must wait for the specified duration
    WaitFor(Duration),
    /// Rate limited until the specified instant
    RateLimited(Instant),
}

/// Cached robots.txt data for a domain
#[derive(Debug, Clone)]
pub struct CachedRobots {
    /// Parsed disallow patterns for our user agent
    disallow_patterns: Vec<String>,
    /// Parsed allow patterns for our user agent
    allow_patterns: Vec<String>,
    /// Crawl delay specified for our user agent
    crawl_delay: Option<Duration>,
    /// When this was fetched
    fetched_at: Instant,
    /// TTL for this cache entry
    ttl: Duration,
}

impl CachedRobots {
    /// Create a new cached robots.txt
    pub fn new(content: String, user_agent: &str) -> Self {
        let (disallow_patterns, allow_patterns, crawl_delay) =
            Self::parse_robots(&content, user_agent);

        Self {
            disallow_patterns,
            allow_patterns,
            crawl_delay,
            fetched_at: Instant::now(),
            ttl: Duration::from_secs(24 * 60 * 60), // 24 hours default
        }
    }

    /// Create an empty (allow-all) robots.txt for when fetch fails
    pub fn allow_all() -> Self {
        Self {
            disallow_patterns: Vec::new(),
            allow_patterns: Vec::new(),
            crawl_delay: None,
            fetched_at: Instant::now(),
            ttl: Duration::from_secs(60 * 60), // 1 hour for failed fetches
        }
    }

    /// Check if a path is allowed
    pub fn is_allowed(&self, path: &str) -> bool {
        // Check allow patterns first (they take precedence for matching length)
        let mut longest_allow_match = 0;
        for pattern in &self.allow_patterns {
            if Self::path_matches(path, pattern) {
                longest_allow_match = longest_allow_match.max(pattern.len());
            }
        }

        let mut longest_disallow_match = 0;
        for pattern in &self.disallow_patterns {
            if Self::path_matches(path, pattern) {
                longest_disallow_match = longest_disallow_match.max(pattern.len());
            }
        }

        // Longer match wins; if equal, allow wins
        longest_allow_match >= longest_disallow_match
    }

    /// Check if cache is still valid
    pub fn is_valid(&self) -> bool {
        self.fetched_at.elapsed() < self.ttl
    }

    /// Get crawl delay
    pub fn crawl_delay(&self) -> Option<Duration> {
        self.crawl_delay
    }

    /// Parse robots.txt content
    fn parse_robots(content: &str, user_agent: &str) -> (Vec<String>, Vec<String>, Option<Duration>) {
        let mut disallow = Vec::new();
        let mut allow = Vec::new();
        let mut crawl_delay = None;

        let ua_lower = user_agent.to_lowercase();
        let mut current_agent_applies = false;
        let mut found_specific_agent = false;

        for line in content.lines() {
            let line = line.trim();

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse directive
            if let Some((directive, value)) = line.split_once(':') {
                let directive = directive.trim().to_lowercase();
                let value = value.trim();

                match directive.as_str() {
                    "user-agent" => {
                        let agent = value.to_lowercase();
                        if agent == "*" {
                            // Wildcard matches if we haven't found a specific match
                            current_agent_applies = !found_specific_agent;
                        } else if ua_lower.contains(&agent) || agent.contains(&ua_lower) {
                            current_agent_applies = true;
                            found_specific_agent = true;
                            // Clear previous wildcard rules
                            disallow.clear();
                            allow.clear();
                        } else {
                            current_agent_applies = false;
                        }
                    }
                    "disallow" if current_agent_applies => {
                        if !value.is_empty() {
                            disallow.push(value.to_string());
                        }
                    }
                    "allow" if current_agent_applies => {
                        if !value.is_empty() {
                            allow.push(value.to_string());
                        }
                    }
                    "crawl-delay" if current_agent_applies => {
                        if let Ok(delay) = value.parse::<f64>() {
                            crawl_delay = Some(Duration::from_secs_f64(delay));
                        }
                    }
                    _ => {}
                }
            }
        }

        (disallow, allow, crawl_delay)
    }

    /// Check if a path matches a robots.txt pattern
    fn path_matches(path: &str, pattern: &str) -> bool {
        if pattern.is_empty() {
            return false;
        }

        // Check for end anchor
        let (pattern, must_end_match) = if pattern.ends_with('$') {
            (&pattern[..pattern.len() - 1], true)
        } else {
            (pattern, false)
        };

        // Handle wildcards
        if pattern.contains('*') {
            let parts: Vec<&str> = pattern.split('*').collect();
            let mut pos = 0;

            for (i, part) in parts.iter().enumerate() {
                if part.is_empty() {
                    continue;
                }

                if let Some(found_pos) = path[pos..].find(part) {
                    if i == 0 && found_pos != 0 {
                        // First part must match at start
                        return false;
                    }
                    pos += found_pos + part.len();
                } else {
                    return false;
                }
            }

            // Check end anchor constraint
            if must_end_match {
                return pos == path.len();
            }

            return true;
        }

        // Handle end anchor without wildcards
        if must_end_match {
            return path == pattern;
        }

        // Simple prefix match
        path.starts_with(pattern)
    }
}

/// Per-domain state for rate limiting
#[derive(Debug, Clone)]
pub struct DomainState {
    /// Last successful fetch time
    pub last_fetch: Instant,
    /// Number of consecutive 429 responses
    pub consecutive_429s: u32,
    /// Backoff until this time (if rate limited)
    pub backoff_until: Option<Instant>,
    /// Number of successful fetches
    pub fetch_count: u64,
}

impl Default for DomainState {
    fn default() -> Self {
        Self {
            last_fetch: Instant::now() - Duration::from_secs(3600),
            consecutive_429s: 0,
            backoff_until: None,
            fetch_count: 0,
        }
    }
}

/// Configuration for the politeness controller
#[derive(Debug, Clone)]
pub struct PolitenessConfig {
    /// User agent string
    pub user_agent: String,
    /// Default delay between requests to same domain
    pub default_delay: Duration,
    /// Minimum delay (even if robots.txt says less)
    pub min_delay: Duration,
    /// Maximum delay (cap robots.txt crawl-delay)
    pub max_delay: Duration,
    /// robots.txt cache size
    pub cache_size: usize,
    /// Request timeout
    pub request_timeout: Duration,
}

impl Default for PolitenessConfig {
    fn default() -> Self {
        Self {
            user_agent: "DecentralizedSearchBot/1.0".to_string(),
            default_delay: Duration::from_millis(1000),
            min_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            cache_size: 10000,
            request_timeout: Duration::from_secs(30),
        }
    }
}

/// Politeness controller managing robots.txt and rate limiting
pub struct PolitenessController {
    /// Cache of robots.txt files
    robots_cache: LruCache<String, CachedRobots>,
    /// Per-domain state
    domain_state: HashMap<String, DomainState>,
    /// Configuration
    config: PolitenessConfig,
    /// HTTP client for fetching robots.txt
    http_client: reqwest::Client,
}

impl PolitenessController {
    /// Create a new politeness controller
    pub fn new(config: PolitenessConfig) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .user_agent(&config.user_agent)
            .build()
            .unwrap_or_default();

        // Use max(1) to ensure cache_size is at least 1, with fallback default of 1000
        let cache_size = config.cache_size.max(1);
        let cache_capacity = NonZeroUsize::new(cache_size)
            .expect("cache_size.max(1) guarantees non-zero");

        Self {
            robots_cache: LruCache::new(cache_capacity),
            domain_state: HashMap::new(),
            config,
            http_client,
        }
    }

    /// Check if a URL can be fetched (synchronous check, may need async robots.txt fetch)
    pub fn can_fetch_sync(&mut self, url: &Url) -> FetchDecision {
        let hostname = url.host_str().unwrap_or_default().to_string();
        let path = url.path();

        // Check robots.txt cache
        if let Some(robots) = self.robots_cache.get(&hostname) {
            if robots.is_valid() && !robots.is_allowed(path) {
                return FetchDecision::Disallowed;
            }
        }

        // Check rate limiting
        self.check_rate_limit(&hostname)
    }

    /// Check if a URL can be fetched (async, will fetch robots.txt if needed)
    pub async fn can_fetch(&mut self, url: &Url) -> FetchDecision {
        let hostname = url.host_str().unwrap_or_default().to_string();
        let path = url.path();

        // Ensure we have robots.txt
        let robots = self.get_or_fetch_robots(&hostname, url).await;

        // Check if path is allowed
        if !robots.is_allowed(path) {
            return FetchDecision::Disallowed;
        }

        // Check rate limiting
        self.check_rate_limit(&hostname)
    }

    /// Check rate limiting for a domain
    fn check_rate_limit(&mut self, hostname: &str) -> FetchDecision {
        // Get delay first before borrowing domain_state mutably
        let min_delay = self.get_delay_for_domain(hostname);

        let state = self
            .domain_state
            .entry(hostname.to_string())
            .or_insert_with(DomainState::default);

        // Check if we're in a backoff period
        if let Some(backoff_until) = state.backoff_until {
            if Instant::now() < backoff_until {
                return FetchDecision::RateLimited(backoff_until);
            }
            // Backoff expired
            state.backoff_until = None;
            state.consecutive_429s = 0;
        }

        // Check minimum delay between requests
        let elapsed = state.last_fetch.elapsed();

        if elapsed < min_delay {
            return FetchDecision::WaitFor(min_delay - elapsed);
        }

        FetchDecision::Allowed
    }

    /// Get the required delay for a domain
    fn get_delay_for_domain(&self, hostname: &str) -> Duration {
        if let Some(robots) = self.robots_cache.peek(hostname) {
            if let Some(crawl_delay) = robots.crawl_delay() {
                // Clamp to configured bounds
                return crawl_delay.clamp(self.config.min_delay, self.config.max_delay);
            }
        }
        self.config.default_delay
    }

    /// Fetch robots.txt for a domain
    async fn get_or_fetch_robots(&mut self, hostname: &str, url: &Url) -> CachedRobots {
        // Check cache
        if let Some(robots) = self.robots_cache.get(hostname) {
            if robots.is_valid() {
                return robots.clone();
            }
        }

        // Construct robots.txt URL
        let robots_url = format!("{}://{}/robots.txt", url.scheme(), hostname);

        let robots = match self.fetch_robots(&robots_url).await {
            Ok(content) => CachedRobots::new(content, &self.config.user_agent),
            Err(_) => CachedRobots::allow_all(),
        };

        // Cache it
        self.robots_cache.put(hostname.to_string(), robots.clone());

        robots
    }

    /// Fetch robots.txt content
    async fn fetch_robots(&self, robots_url: &str) -> Result<String, reqwest::Error> {
        let response = self.http_client.get(robots_url).send().await?;

        if response.status().is_success() {
            response.text().await
        } else {
            Ok(String::new()) // Treat non-200 as allow-all
        }
    }

    /// Record a successful fetch
    pub fn record_success(&mut self, hostname: &str) {
        let state = self
            .domain_state
            .entry(hostname.to_string())
            .or_insert_with(DomainState::default);

        state.last_fetch = Instant::now();
        state.consecutive_429s = 0;
        state.fetch_count += 1;
    }

    /// Record a 429 (Too Many Requests) response
    pub fn record_429(&mut self, hostname: &str, retry_after: Option<Duration>) {
        let state = self
            .domain_state
            .entry(hostname.to_string())
            .or_insert_with(DomainState::default);

        state.consecutive_429s += 1;
        state.last_fetch = Instant::now();

        // Exponential backoff: 30s, 60s, 120s, 240s, max 10min
        let backoff = retry_after.unwrap_or_else(|| {
            Duration::from_secs(30 * 2u64.pow(state.consecutive_429s.min(4) - 1))
        });

        let capped_backoff = backoff.min(Duration::from_secs(600)); // Max 10 minutes
        state.backoff_until = Some(Instant::now() + capped_backoff);
    }

    /// Record an error (connection failure, timeout, etc.)
    pub fn record_error(&mut self, hostname: &str) {
        let state = self
            .domain_state
            .entry(hostname.to_string())
            .or_insert_with(DomainState::default);

        state.last_fetch = Instant::now();
        // Small backoff on errors
        state.backoff_until = Some(Instant::now() + Duration::from_secs(5));
    }

    /// Get all domains tracked by the politeness controller (fetched at least once).
    pub fn tracked_domains(&self) -> HashSet<String> {
        self.domain_state.keys().cloned().collect()
    }

    /// Get all domains that are ready to crawl
    pub fn ready_domains(&self) -> Vec<String> {
        let now = Instant::now();

        self.domain_state
            .iter()
            .filter(|(hostname, state)| {
                // Check if backoff expired
                if let Some(backoff_until) = state.backoff_until {
                    if now < backoff_until {
                        return false;
                    }
                }

                // Check delay
                let delay = self.get_delay_for_domain(hostname);
                state.last_fetch.elapsed() >= delay
            })
            .map(|(hostname, _)| hostname.clone())
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> PolitenessStats {
        let rate_limited = self
            .domain_state
            .values()
            .filter(|s| s.backoff_until.map(|b| Instant::now() < b).unwrap_or(false))
            .count();

        PolitenessStats {
            domains_tracked: self.domain_state.len(),
            robots_cached: self.robots_cache.len(),
            rate_limited_domains: rate_limited,
            total_fetches: self.domain_state.values().map(|s| s.fetch_count).sum(),
        }
    }

    /// Get the user agent string
    pub fn user_agent(&self) -> &str {
        &self.config.user_agent
    }
}

/// Statistics from the politeness controller
#[derive(Debug, Clone)]
pub struct PolitenessStats {
    pub domains_tracked: usize,
    pub robots_cached: usize,
    pub rate_limited_domains: usize,
    pub total_fetches: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robots_parsing() {
        let content = r#"
User-agent: *
Disallow: /private/
Allow: /private/public/
Crawl-delay: 2

User-agent: DecentralizedSearchBot
Disallow: /admin/
Crawl-delay: 1
"#;

        let robots = CachedRobots::new(content.to_string(), "DecentralizedSearchBot");

        // Should use specific rules for our bot
        assert!(robots.is_allowed("/public/page.html"));
        assert!(!robots.is_allowed("/admin/settings"));
        assert!(robots.is_allowed("/private/test")); // Specific bot doesn't have this rule
        assert_eq!(robots.crawl_delay(), Some(Duration::from_secs(1)));
    }

    #[test]
    fn test_robots_wildcard() {
        let content = r#"
User-agent: *
Disallow: /private/
Disallow: /*.pdf$
Allow: /private/readme.txt
"#;

        let robots = CachedRobots::new(content.to_string(), "TestBot");

        assert!(robots.is_allowed("/public/page.html"));
        assert!(!robots.is_allowed("/private/secret"));
        assert!(robots.is_allowed("/private/readme.txt")); // Allow takes precedence
        assert!(!robots.is_allowed("/docs/manual.pdf"));
        assert!(robots.is_allowed("/docs/manual.html"));
    }

    #[test]
    fn test_path_matching() {
        // Simple prefix
        assert!(CachedRobots::path_matches("/admin/test", "/admin/"));
        assert!(!CachedRobots::path_matches("/public/test", "/admin/"));

        // Wildcard
        assert!(CachedRobots::path_matches("/images/cat.jpg", "/images/*.jpg"));

        // End anchor
        assert!(CachedRobots::path_matches("/page.html", "/page.html$"));
        assert!(!CachedRobots::path_matches("/page.html?query", "/page.html$"));
    }

    #[test]
    fn test_rate_limiting() {
        let config = PolitenessConfig {
            default_delay: Duration::from_millis(100),
            min_delay: Duration::from_millis(50),
            ..Default::default()
        };

        let mut controller = PolitenessController::new(config);
        let url = Url::parse("https://example.com/page").unwrap();

        // First fetch should be allowed
        controller.record_success("example.com");

        // Immediate second fetch should require waiting
        let decision = controller.can_fetch_sync(&url);
        match decision {
            FetchDecision::WaitFor(d) => assert!(d > Duration::ZERO),
            _ => panic!("Expected WaitFor"),
        }
    }

    #[test]
    fn test_429_backoff() {
        let mut controller = PolitenessController::new(PolitenessConfig::default());

        controller.record_429("example.com", None);

        let state = controller.domain_state.get("example.com").unwrap();
        assert!(state.backoff_until.is_some());
        assert_eq!(state.consecutive_429s, 1);
    }
}
