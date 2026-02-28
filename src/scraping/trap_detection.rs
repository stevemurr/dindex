//! Crawl trap detection
//!
//! Detects URL patterns that indicate crawl traps:
//! - Excessive path depth (e.g., /a/b/c/d/e/f/g/h)
//! - Repetitive path patterns (e.g., /a/b/a/b/a/b)
//! - Extremely long URLs
//! - Calendar traps (infinite date pagination)

use url::Url;

/// Configuration for crawl trap detection
#[derive(Debug, Clone)]
pub struct TrapDetectorConfig {
    /// Maximum URL path depth (number of segments)
    pub max_path_depth: usize,
    /// Maximum URL length in characters
    pub max_url_length: usize,
    /// Maximum number of repeated path segments
    pub max_repeated_segments: usize,
}

impl Default for TrapDetectorConfig {
    fn default() -> Self {
        Self {
            max_path_depth: 15,
            max_url_length: 2048,
            max_repeated_segments: 3,
        }
    }
}

/// Detect if a URL is likely a crawl trap
pub fn is_crawl_trap(url: &Url, config: &TrapDetectorConfig) -> bool {
    // Check URL length
    if url.as_str().len() > config.max_url_length {
        return true;
    }

    let path = url.path();
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    // Check path depth
    if segments.len() > config.max_path_depth {
        return true;
    }

    // Check for repetitive patterns
    if has_repetitive_pattern(&segments, config.max_repeated_segments) {
        return true;
    }

    // Check for calendar trap patterns
    if is_calendar_trap(path) {
        return true;
    }

    false
}

fn has_repetitive_pattern(segments: &[&str], max_repeats: usize) -> bool {
    if segments.len() < 4 {
        return false;
    }

    // Check for repeated individual segments
    for window_size in 1..=segments.len() / 2 {
        let mut repeat_count = 0;
        for i in 0..segments.len().saturating_sub(window_size) {
            if segments[i] == segments[i + window_size] {
                repeat_count += 1;
                if repeat_count >= max_repeats {
                    return true;
                }
            }
        }
    }

    false
}

fn is_calendar_trap(path: &str) -> bool {
    // Match patterns like /2024/01/02, /calendar/2024/01
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let mut consecutive_numbers = 0;
    for part in &parts {
        if part.parse::<u32>().is_ok() {
            consecutive_numbers += 1;
            if consecutive_numbers >= 3 {
                return true;
            }
        } else {
            consecutive_numbers = 0;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_url_is_not_trap() {
        let url = Url::parse("https://example.com/blog/my-post").unwrap();
        let config = TrapDetectorConfig::default();
        assert!(!is_crawl_trap(&url, &config));
    }

    #[test]
    fn test_deep_path_is_trap() {
        let url = Url::parse(
            "https://example.com/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p",
        )
        .unwrap();
        let config = TrapDetectorConfig::default();
        assert!(is_crawl_trap(&url, &config));
    }

    #[test]
    fn test_long_url_is_trap() {
        let long_path = "a/".repeat(1025);
        let url = Url::parse(&format!("https://example.com/{}", long_path)).unwrap();
        let config = TrapDetectorConfig::default();
        assert!(is_crawl_trap(&url, &config));
    }

    #[test]
    fn test_repetitive_pattern_is_trap() {
        let url =
            Url::parse("https://example.com/forum/thread/forum/thread/forum/thread/page")
                .unwrap();
        let config = TrapDetectorConfig::default();
        assert!(is_crawl_trap(&url, &config));
    }

    #[test]
    fn test_calendar_trap() {
        let url = Url::parse("https://example.com/calendar/2024/01/15").unwrap();
        let config = TrapDetectorConfig::default();
        assert!(is_crawl_trap(&url, &config));
    }

    #[test]
    fn test_two_numbers_not_calendar_trap() {
        let url = Url::parse("https://example.com/blog/2024/01").unwrap();
        let config = TrapDetectorConfig::default();
        assert!(!is_crawl_trap(&url, &config));
    }

    #[test]
    fn test_custom_config() {
        let url = Url::parse("https://example.com/a/b/c/d/e").unwrap();
        let config = TrapDetectorConfig {
            max_path_depth: 3,
            ..Default::default()
        };
        assert!(is_crawl_trap(&url, &config));
    }

    #[test]
    fn test_short_path_no_repetition() {
        let url = Url::parse("https://example.com/a/b/c").unwrap();
        let config = TrapDetectorConfig::default();
        assert!(!is_crawl_trap(&url, &config));
    }
}
