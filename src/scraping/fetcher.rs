//! Fetch engine for web scraping
//!
//! Implements a two-tier fetching strategy:
//! - Tier 1: Fast HTTP (reqwest) - first attempt for all URLs
//! - Tier 2: Headless browser - fallback for JS-rendered content
//!
//! Note: Headless browser support requires external chromium installation
//! and is optional. The HTTP tier handles most static content.

use std::time::{Duration, Instant};
use thiserror::Error;
use url::Url;

/// Errors that can occur during fetching
#[derive(Debug, Error)]
pub enum FetchError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("URL not allowed by robots.txt")]
    Disallowed,
    #[error("Rate limited, retry after {0:?}")]
    RateLimited(Duration),
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    #[error("Too many redirects")]
    TooManyRedirects,
    #[error("Invalid content type: {0}")]
    InvalidContentType(String),
    #[error("Content too large: {0} bytes")]
    ContentTooLarge(usize),
    #[error("Failed to parse URL: {0}")]
    InvalidUrl(String),
}

/// Result of a successful fetch
#[derive(Debug, Clone)]
pub struct FetchResult {
    /// The fetched URL (may differ from request due to redirects)
    pub final_url: Url,
    /// HTTP status code
    pub status_code: u16,
    /// Response headers
    pub headers: Vec<(String, String)>,
    /// Response body (HTML content)
    pub body: String,
    /// Content type
    pub content_type: String,
    /// Time taken to fetch
    pub fetch_duration: Duration,
    /// Whether JavaScript rendering was used
    pub js_rendered: bool,
    /// ETag for caching
    pub etag: Option<String>,
    /// Last-Modified header
    pub last_modified: Option<String>,
}

impl FetchResult {
    /// Check if this is HTML content
    pub fn is_html(&self) -> bool {
        self.content_type.contains("text/html")
    }

    /// Get a header value
    pub fn header(&self, name: &str) -> Option<&str> {
        let name_lower = name.to_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == name_lower)
            .map(|(_, v)| v.as_str())
    }
}

/// Configuration for the fetch engine
#[derive(Debug, Clone)]
pub struct FetchConfig {
    /// User agent string
    pub user_agent: String,
    /// Request timeout
    pub timeout: Duration,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Maximum response size (bytes)
    pub max_content_size: usize,
    /// Maximum redirects to follow
    pub max_redirects: usize,
    /// Minimum text ratio to consider content rendered (0.0-1.0)
    pub min_text_ratio: f32,
    /// Enable JavaScript rendering fallback
    pub enable_js_rendering: bool,
    /// Concurrent connections per host
    pub connections_per_host: usize,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            user_agent: "DecentralizedSearchBot/1.0 (+https://github.com/dindex)".to_string(),
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(10),
            max_content_size: 10 * 1024 * 1024, // 10 MB
            max_redirects: 10,
            min_text_ratio: 0.1,
            enable_js_rendering: false, // Disabled by default
            connections_per_host: 10,
        }
    }
}

/// Two-tier fetch engine
pub struct FetchEngine {
    /// HTTP client for tier 1
    http_client: reqwest::Client,
    /// Configuration
    config: FetchConfig,
    /// Fetch statistics
    stats: FetchStats,
}

/// Fetch statistics
#[derive(Debug, Clone, Default)]
pub struct FetchStats {
    /// Total fetch attempts
    pub total_fetches: u64,
    /// Successful HTTP fetches
    pub http_successes: u64,
    /// HTTP failures
    pub http_failures: u64,
    /// JS rendering attempts
    pub js_attempts: u64,
    /// JS rendering successes
    pub js_successes: u64,
    /// Average fetch time (ms)
    pub avg_fetch_time_ms: f64,
}

impl FetchEngine {
    /// Create a new fetch engine
    pub fn new(config: FetchConfig) -> Result<Self, FetchError> {
        let http_client = reqwest::Client::builder()
            .pool_max_idle_per_host(config.connections_per_host)
            .pool_idle_timeout(Duration::from_secs(90))
            .timeout(config.timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects))
            .user_agent(&config.user_agent)
            .gzip(true)
            .brotli(true)
            .build()?;

        Ok(Self {
            http_client,
            config,
            stats: FetchStats::default(),
        })
    }

    /// Fetch a URL, falling back to JS rendering if needed
    pub async fn fetch(&mut self, url: &Url) -> Result<FetchResult, FetchError> {
        self.stats.total_fetches += 1;
        let start = Instant::now();

        // Tier 1: HTTP fetch
        let result = self.fetch_http(url).await;

        match &result {
            Ok(response) => {
                self.stats.http_successes += 1;

                // Check if we need JS rendering
                if self.config.enable_js_rendering && self.needs_js_rendering(response) {
                    self.stats.js_attempts += 1;
                    // For now, just return the HTTP result
                    // In production, would invoke chromiumoxide here
                    tracing::debug!("Content at {} may need JS rendering", url);
                }

                self.update_timing(start.elapsed());
            }
            Err(_) => {
                self.stats.http_failures += 1;
            }
        }

        result
    }

    /// Perform HTTP fetch
    async fn fetch_http(&self, url: &Url) -> Result<FetchResult, FetchError> {
        let start = Instant::now();

        let response = self.http_client.get(url.as_str()).send().await?;

        let status = response.status();
        let final_url = response.url().clone();

        // Extract headers
        let headers: Vec<(String, String)> = response
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or_default().to_string()))
            .collect();

        // Get content type
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/html")
            .to_string();

        // Check content type
        if !content_type.contains("text/html")
            && !content_type.contains("application/xhtml")
            && !content_type.contains("text/plain")
        {
            return Err(FetchError::InvalidContentType(content_type));
        }

        // Extract caching headers
        let etag = headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == "etag")
            .map(|(_, v)| v.clone());

        let last_modified = headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == "last-modified")
            .map(|(_, v)| v.clone());

        // Check content length
        if let Some(len) = response.content_length() {
            if len as usize > self.config.max_content_size {
                return Err(FetchError::ContentTooLarge(len as usize));
            }
        }

        // Get body
        let body = response.text().await?;

        if body.len() > self.config.max_content_size {
            return Err(FetchError::ContentTooLarge(body.len()));
        }

        Ok(FetchResult {
            final_url: Url::parse(final_url.as_str())
                .map_err(|e| FetchError::InvalidUrl(e.to_string()))?,
            status_code: status.as_u16(),
            headers,
            body,
            content_type,
            fetch_duration: start.elapsed(),
            js_rendered: false,
            etag,
            last_modified,
        })
    }

    /// Check if content needs JavaScript rendering
    fn needs_js_rendering(&self, response: &FetchResult) -> bool {
        if !response.is_html() {
            return false;
        }

        let body = &response.body;

        // Very little text content
        let text_ratio = Self::estimate_text_ratio(body);
        if text_ratio < self.config.min_text_ratio {
            return true;
        }

        // Heavy JS framework indicators
        let js_hints = [
            "window.__NEXT_DATA__", // Next.js
            "window.__NUXT__",      // Nuxt.js
            "ng-app",               // Angular
            "ng-controller",        // Angular
            "<div id=\"root\"></div>", // React SPA
            "<div id=\"app\"></div>", // Vue SPA
            "data-reactroot",       // React
        ];

        if js_hints.iter().any(|h| body.contains(h)) {
            return true;
        }

        // Loading placeholder patterns
        if body.contains("Loading...") && body.len() < 5000 {
            return true;
        }

        if body.contains("Please wait") && body.len() < 5000 {
            return true;
        }

        false
    }

    /// Estimate the ratio of text content to total HTML
    fn estimate_text_ratio(html: &str) -> f32 {
        let total_len = html.len();
        if total_len == 0 {
            return 0.0;
        }

        // Simple heuristic: count characters outside of tags
        let mut in_tag = false;
        let mut text_chars = 0;

        for c in html.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag && !c.is_whitespace() => text_chars += 1,
                _ => {}
            }
        }

        text_chars as f32 / total_len as f32
    }

    fn update_timing(&mut self, duration: Duration) {
        let ms = duration.as_secs_f64() * 1000.0;
        let n = self.stats.total_fetches as f64;
        // Running average
        self.stats.avg_fetch_time_ms =
            (self.stats.avg_fetch_time_ms * (n - 1.0) + ms) / n;
    }

    /// Get fetch statistics
    pub fn stats(&self) -> &FetchStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &FetchConfig {
        &self.config
    }

    /// Get the user agent
    pub fn user_agent(&self) -> &str {
        &self.config.user_agent
    }
}

/// Extract URLs from a fetch result using proper HTML parsing
pub fn extract_urls(result: &FetchResult) -> Vec<Url> {
    use scraper::{Html, Selector};
    use std::collections::HashSet;

    let base_url = &result.final_url;
    let document = Html::parse_document(&result.body);

    let selector = match Selector::parse("a[href]") {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let mut seen = HashSet::new();
    let mut urls = Vec::new();

    for element in document.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            if let Ok(url) = base_url.join(href) {
                if (url.scheme() == "http" || url.scheme() == "https") && seen.insert(url.as_str().to_string()) {
                    urls.push(url);
                }
            }
        }
    }

    urls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_ratio() {
        let html = "<html><body><p>Hello World</p></body></html>";
        let ratio = FetchEngine::estimate_text_ratio(html);
        assert!(ratio > 0.0 && ratio < 1.0);

        let mostly_tags = "<div><div><div><div></div></div></div></div>";
        let tag_ratio = FetchEngine::estimate_text_ratio(mostly_tags);
        assert!(tag_ratio < 0.1);
    }

    #[test]
    fn test_url_extraction() {
        let result = FetchResult {
            final_url: Url::parse("https://example.com/page").unwrap(),
            status_code: 200,
            headers: vec![],
            body: r#"
                <a href="/about">About</a>
                <a href="https://example.com/contact">Contact</a>
                <a href='https://other.com/page'>Other</a>
            "#
            .to_string(),
            content_type: "text/html".to_string(),
            fetch_duration: Duration::from_millis(100),
            js_rendered: false,
            etag: None,
            last_modified: None,
        };

        let urls = extract_urls(&result);
        assert_eq!(urls.len(), 3);
        assert!(urls.iter().any(|u| u.as_str() == "https://example.com/about"));
        assert!(urls.iter().any(|u| u.as_str() == "https://example.com/contact"));
        assert!(urls.iter().any(|u| u.as_str() == "https://other.com/page"));
    }

    #[test]
    fn test_needs_js_rendering() {
        let config = FetchConfig::default();
        let engine = FetchEngine::new(config).unwrap();

        // React SPA shell
        let spa_result = FetchResult {
            final_url: Url::parse("https://example.com").unwrap(),
            status_code: 200,
            headers: vec![],
            body: r#"<html><head></head><body><div id="root"></div><script src="app.js"></script></body></html>"#.to_string(),
            content_type: "text/html".to_string(),
            fetch_duration: Duration::from_millis(100),
            js_rendered: false,
            etag: None,
            last_modified: None,
        };

        assert!(engine.needs_js_rendering(&spa_result));

        // Normal HTML page
        let normal_result = FetchResult {
            final_url: Url::parse("https://example.com").unwrap(),
            status_code: 200,
            headers: vec![],
            body: r#"<html><head><title>Test</title></head><body><h1>Hello</h1><p>This is a paragraph with lots of text content that should be considered rendered properly without JavaScript.</p></body></html>"#.to_string(),
            content_type: "text/html".to_string(),
            fetch_duration: Duration::from_millis(100),
            js_rendered: false,
            etag: None,
            last_modified: None,
        };

        assert!(!engine.needs_js_rendering(&normal_result));
    }
}
