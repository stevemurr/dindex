//! Content extraction from HTML
//!
//! Uses Mozilla's readability algorithm for content extraction:
//! - Removes boilerplate (navigation, ads, footers)
//! - Extracts main article content
//! - Parses metadata (title, author, date, etc.)

use chrono::{DateTime, Utc};
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::io::Cursor;
use thiserror::Error;
use url::Url;

/// Errors during content extraction
#[derive(Debug, Error)]
pub enum ExtractError {
    #[error("No content found")]
    NoContent,
    #[error("Failed to parse HTML")]
    ParseError,
    #[error("Content too short: {0} chars")]
    TooShort(usize),
}

/// Extracted content from a web page
#[derive(Debug, Clone)]
pub struct ExtractedContent {
    /// Page title
    pub title: String,
    /// Main text content (clean, no HTML)
    pub text_content: String,
    /// Clean HTML (if available)
    pub clean_html: Option<String>,
    /// Author name
    pub author: Option<String>,
    /// Published date
    pub published_date: Option<DateTime<Utc>>,
    /// Page description/excerpt
    pub excerpt: Option<String>,
    /// Language
    pub language: Option<String>,
    /// Word count
    pub word_count: usize,
    /// Estimated reading time in minutes
    pub reading_time_minutes: u8,
}

/// Full extracted metadata from a page
#[derive(Debug, Clone)]
pub struct ExtractedMetadata {
    /// Original URL
    pub url: String,
    /// Canonical URL if specified
    pub canonical_url: Option<String>,
    /// Page title
    pub title: String,
    /// Description
    pub description: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Published date
    pub published_date: Option<DateTime<Utc>>,
    /// Modified date
    pub modified_date: Option<DateTime<Utc>>,
    /// Language code
    pub language: Option<String>,
    /// Content type (article, product, recipe, etc.)
    pub content_type: ContentType,
    /// Word count
    pub word_count: usize,
    /// Reading time in minutes
    pub reading_time_minutes: u8,
    /// Source domain
    pub domain: String,
    /// When scraped
    pub fetched_at: DateTime<Utc>,
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

/// Type of content on the page
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ContentType {
    #[default]
    Article,
    Product,
    Recipe,
    Video,
    Profile,
    Organization,
    Event,
    Other(String),
}

impl ContentType {
    fn from_schema_type(schema_type: &str) -> Self {
        match schema_type.to_lowercase().as_str() {
            "article" | "newsarticle" | "blogposting" | "technicalarticle" => ContentType::Article,
            "product" => ContentType::Product,
            "recipe" => ContentType::Recipe,
            "video" | "videoobject" => ContentType::Video,
            "person" | "profilepage" => ContentType::Profile,
            "organization" | "localbusiness" => ContentType::Organization,
            "event" => ContentType::Event,
            other => ContentType::Other(other.to_string()),
        }
    }
}

/// Configuration for content extraction
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Minimum content length in characters
    pub min_content_length: usize,
    /// Minimum word count
    pub min_word_count: usize,
    /// Average words per minute for reading time calculation
    pub words_per_minute: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            min_content_length: 100,
            min_word_count: 20,
            words_per_minute: 200,
        }
    }
}

/// Content extractor
pub struct ContentExtractor {
    config: ExtractorConfig,
    /// Pre-compiled selectors for finding main content
    content_selectors: Vec<Selector>,
}

impl ContentExtractor {
    /// Create a new content extractor
    pub fn new(config: ExtractorConfig) -> Self {
        // Selectors for main content (in priority order)
        let content_selectors: Vec<Selector> = [
            "article",
            "main",
            "[role='main']",
            ".post-content",
            ".article-content",
            ".entry-content",
            ".content",
            "#content",
            ".post",
            ".article",
        ]
        .iter()
        .filter_map(|s| Selector::parse(s).ok())
        .collect();

        // Note: We don't need remove_selectors because the `readability` library
        // handles boilerplate removal (scripts, styles, nav, etc.) internally

        Self {
            config,
            content_selectors,
        }
    }

    /// Extract content from HTML using readability
    pub fn extract(&self, html: &str, url: &Url) -> Result<ExtractedContent, ExtractError> {
        // Use readability for main content extraction
        let mut cursor = Cursor::new(html.as_bytes());
        let product = readability::extractor::extract(&mut cursor, url)
            .map_err(|_| ExtractError::NoContent)?;

        let text_content = product.text;
        let title = product.title;
        let clean_html = Some(product.content);

        if text_content.len() < self.config.min_content_length {
            return Err(ExtractError::TooShort(text_content.len()));
        }

        let word_count = text_content.split_whitespace().count();
        if word_count < self.config.min_word_count {
            return Err(ExtractError::TooShort(word_count));
        }

        let reading_time = ((word_count as f32 / self.config.words_per_minute as f32).ceil() as u8)
            .max(1);

        // Extract additional metadata from DOM (readability doesn't provide these)
        let document = Html::parse_document(html);
        let author = self.extract_author(&document);
        let published_date = self.extract_date(&document);
        let language = self.extract_language(&document);
        let excerpt = self.extract_excerpt(&document, &text_content);

        Ok(ExtractedContent {
            title,
            text_content,
            clean_html,
            author,
            published_date,
            excerpt,
            language,
            word_count,
            reading_time_minutes: reading_time,
        })
    }

    /// Extract full metadata from HTML
    pub fn extract_metadata(&self, html: &str, url: &Url) -> ExtractedMetadata {
        let document = Html::parse_document(html);

        // Extract JSON-LD
        let json_ld = self.extract_json_ld(&document);

        // Extract OpenGraph
        let og = self.extract_opengraph(&document);

        // Extract Twitter cards
        let twitter = self.extract_twitter_cards(&document);

        // Extract meta tags
        let meta = self.extract_meta_tags(&document);

        // Extract title with fallback chain
        let title = json_ld
            .get("headline")
            .or_else(|| json_ld.get("name"))
            .or_else(|| og.get("title"))
            .or_else(|| twitter.get("title"))
            .or_else(|| meta.get("title"))
            .cloned()
            .unwrap_or_else(|| self.extract_title(&document));

        // Extract description
        let description = json_ld
            .get("description")
            .or_else(|| og.get("description"))
            .or_else(|| twitter.get("description"))
            .or_else(|| meta.get("description"))
            .cloned();

        // Extract author
        let author = json_ld
            .get("author")
            .or_else(|| meta.get("author"))
            .cloned()
            .or_else(|| self.extract_author(&document));

        // Extract dates
        let published_date = json_ld
            .get("datePublished")
            .or_else(|| meta.get("date"))
            .and_then(|d| Self::parse_date(d))
            .or_else(|| {
                // Try article:published_time meta property
                self.get_meta_content(&document, "article:published_time")
                    .and_then(|d| Self::parse_date(&d))
            });

        let modified_date = json_ld
            .get("dateModified")
            .and_then(|d| Self::parse_date(d));

        // Extract language
        let language = meta
            .get("language")
            .cloned()
            .or_else(|| self.extract_language(&document));

        // Determine content type
        let content_type = json_ld
            .get("@type")
            .map(|t| ContentType::from_schema_type(t))
            .unwrap_or_default();

        // Extract canonical URL
        let canonical_url = self.extract_canonical(&document);

        // Calculate content metrics
        let text_content = self.extract_text(&self.find_main_content(&document));
        let word_count = text_content.split_whitespace().count();
        let reading_time =
            ((word_count as f32 / self.config.words_per_minute as f32).ceil() as u8).max(1);

        ExtractedMetadata {
            url: url.to_string(),
            canonical_url,
            title,
            description,
            author,
            published_date,
            modified_date,
            language,
            content_type,
            word_count,
            reading_time_minutes: reading_time,
            domain: url.host_str().unwrap_or_default().to_string(),
            fetched_at: Utc::now(),
            extra: HashMap::new(),
        }
    }

    /// Extract page title
    fn extract_title(&self, document: &Html) -> String {
        // Try og:title first
        if let Some(og_title) = self.get_meta_content(document, "og:title") {
            return og_title;
        }

        // Then title tag
        if let Ok(selector) = Selector::parse("title") {
            if let Some(title_elem) = document.select(&selector).next() {
                let title = title_elem.text().collect::<String>().trim().to_string();
                if !title.is_empty() {
                    return title;
                }
            }
        }

        // Try h1
        if let Ok(selector) = Selector::parse("h1") {
            if let Some(h1_elem) = document.select(&selector).next() {
                let title = h1_elem.text().collect::<String>().trim().to_string();
                if !title.is_empty() {
                    return title;
                }
            }
        }

        "Untitled".to_string()
    }

    /// Find the main content area
    fn find_main_content(&self, document: &Html) -> String {
        // Try each content selector in priority order
        for selector in &self.content_selectors {
            if let Some(element) = document.select(selector).next() {
                let html = element.html();
                if html.len() > 200 {
                    return html;
                }
            }
        }

        // Fall back to body
        if let Ok(body_sel) = Selector::parse("body") {
            if let Some(body) = document.select(&body_sel).next() {
                return body.html();
            }
        }

        String::new()
    }

    /// Extract clean text from HTML
    fn extract_text(&self, html: &str) -> String {
        let fragment = Html::parse_fragment(html);

        // Get all text nodes, excluding scripts and styles
        let mut text = String::new();
        let mut last_was_block = false;

        for node in fragment.root_element().descendants() {
            if let Some(text_node) = node.value().as_text() {
                let t = text_node.trim();
                if !t.is_empty() {
                    if last_was_block {
                        text.push('\n');
                    } else if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(t);
                    last_was_block = false;
                }
            } else if let Some(elem) = node.value().as_element() {
                // Check if this is a block element
                let name = elem.name();
                let is_block = matches!(
                    name,
                    "p" | "div"
                        | "br"
                        | "h1"
                        | "h2"
                        | "h3"
                        | "h4"
                        | "h5"
                        | "h6"
                        | "li"
                        | "tr"
                        | "blockquote"
                );
                if is_block {
                    last_was_block = true;
                }
            }
        }

        // Clean up whitespace
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Extract author
    fn extract_author(&self, document: &Html) -> Option<String> {
        // Try meta author
        if let Some(author) = self.get_meta_content(document, "author") {
            return Some(author);
        }

        // Try schema.org author
        if let Ok(selector) = Selector::parse("[itemprop='author']") {
            if let Some(elem) = document.select(&selector).next() {
                let text = elem.text().collect::<String>().trim().to_string();
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }

        // Try common author class names
        for class in &[".author", ".byline", ".writer", "[rel='author']"] {
            if let Ok(selector) = Selector::parse(class) {
                if let Some(elem) = document.select(&selector).next() {
                    let text = elem.text().collect::<String>().trim().to_string();
                    if !text.is_empty() && text.len() < 100 {
                        return Some(text);
                    }
                }
            }
        }

        None
    }

    /// Extract publication date
    fn extract_date(&self, document: &Html) -> Option<DateTime<Utc>> {
        // Try time element with datetime attribute
        if let Ok(selector) = Selector::parse("time[datetime]") {
            if let Some(elem) = document.select(&selector).next() {
                if let Some(datetime) = elem.value().attr("datetime") {
                    if let Some(dt) = Self::parse_date(datetime) {
                        return Some(dt);
                    }
                }
            }
        }

        // Try meta date
        for name in &[
            "article:published_time",
            "date",
            "DC.date",
            "datePublished",
        ] {
            if let Some(date_str) = self.get_meta_content(document, name) {
                if let Some(dt) = Self::parse_date(&date_str) {
                    return Some(dt);
                }
            }
        }

        None
    }

    /// Extract language
    fn extract_language(&self, document: &Html) -> Option<String> {
        // Try html lang attribute
        if let Ok(selector) = Selector::parse("html") {
            if let Some(html_elem) = document.select(&selector).next() {
                if let Some(lang) = html_elem.value().attr("lang") {
                    return Some(lang.to_string());
                }
            }
        }

        // Try meta language
        self.get_meta_content(document, "language")
            .or_else(|| self.get_meta_content(document, "og:locale"))
    }

    /// Extract excerpt/description
    fn extract_excerpt(&self, document: &Html, text_content: &str) -> Option<String> {
        // Try meta description first
        if let Some(desc) = self.get_meta_content(document, "description") {
            if !desc.is_empty() {
                return Some(desc);
            }
        }

        if let Some(desc) = self.get_meta_content(document, "og:description") {
            if !desc.is_empty() {
                return Some(desc);
            }
        }

        // Generate from first paragraph
        let words: Vec<&str> = text_content.split_whitespace().take(50).collect();
        if words.len() >= 10 {
            let excerpt = words.join(" ");
            return Some(if words.len() == 50 {
                format!("{}...", excerpt)
            } else {
                excerpt
            });
        }

        None
    }

    /// Extract canonical URL
    fn extract_canonical(&self, document: &Html) -> Option<String> {
        if let Ok(selector) = Selector::parse("link[rel='canonical']") {
            if let Some(elem) = document.select(&selector).next() {
                return elem.value().attr("href").map(|s| s.to_string());
            }
        }
        self.get_meta_content(document, "og:url")
    }

    /// Get meta content by name or property
    fn get_meta_content(&self, document: &Html, name: &str) -> Option<String> {
        // Try name attribute
        let name_selector = format!("meta[name='{}']", name);
        if let Ok(selector) = Selector::parse(&name_selector) {
            if let Some(elem) = document.select(&selector).next() {
                if let Some(content) = elem.value().attr("content") {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
        }

        // Try property attribute (for OpenGraph)
        let prop_selector = format!("meta[property='{}']", name);
        if let Ok(selector) = Selector::parse(&prop_selector) {
            if let Some(elem) = document.select(&selector).next() {
                if let Some(content) = elem.value().attr("content") {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
        }

        None
    }

    /// Extract JSON-LD structured data
    fn extract_json_ld(&self, document: &Html) -> HashMap<String, String> {
        let mut data = HashMap::new();

        if let Ok(selector) = Selector::parse("script[type='application/ld+json']") {
            for script in document.select(&selector) {
                let json_text = script.text().collect::<String>();
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_text) {
                    Self::flatten_json_ld(&value, &mut data);
                }
            }
        }

        data
    }

    /// Flatten JSON-LD to simple key-value pairs
    fn flatten_json_ld(value: &serde_json::Value, data: &mut HashMap<String, String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, val) in map {
                    match val {
                        serde_json::Value::String(s) => {
                            data.insert(key.clone(), s.clone());
                        }
                        serde_json::Value::Object(nested) => {
                            // Handle nested author, etc.
                            if let Some(name) = nested.get("name") {
                                if let Some(s) = name.as_str() {
                                    data.insert(key.clone(), s.to_string());
                                }
                            }
                        }
                        serde_json::Value::Array(arr) => {
                            // Handle @graph arrays
                            for item in arr {
                                Self::flatten_json_ld(item, data);
                            }
                        }
                        _ => {}
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::flatten_json_ld(item, data);
                }
            }
            _ => {}
        }
    }

    /// Extract OpenGraph metadata
    fn extract_opengraph(&self, document: &Html) -> HashMap<String, String> {
        let mut data = HashMap::new();

        for prop in &["title", "description", "type", "url", "image", "locale"] {
            let key = format!("og:{}", prop);
            if let Some(value) = self.get_meta_content(document, &key) {
                data.insert(prop.to_string(), value);
            }
        }

        data
    }

    /// Extract Twitter Cards metadata
    fn extract_twitter_cards(&self, document: &Html) -> HashMap<String, String> {
        let mut data = HashMap::new();

        for prop in &["title", "description", "card", "site", "creator"] {
            let key = format!("twitter:{}", prop);
            if let Some(value) = self.get_meta_content(document, &key) {
                data.insert(prop.to_string(), value);
            }
        }

        data
    }

    /// Extract basic meta tags
    fn extract_meta_tags(&self, document: &Html) -> HashMap<String, String> {
        let mut data = HashMap::new();

        for name in &["title", "description", "author", "keywords", "date", "language"] {
            if let Some(value) = self.get_meta_content(document, name) {
                data.insert(name.to_string(), value);
            }
        }

        data
    }

    /// Parse a date string into DateTime
    fn parse_date(date_str: &str) -> Option<DateTime<Utc>> {
        // Try ISO 8601 formats
        if let Ok(dt) = DateTime::parse_from_rfc3339(date_str) {
            return Some(dt.with_timezone(&Utc));
        }

        if let Ok(dt) = DateTime::parse_from_rfc2822(date_str) {
            return Some(dt.with_timezone(&Utc));
        }

        // Try common formats
        let formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
        ];

        for format in &formats {
            if let Ok(naive) = chrono::NaiveDate::parse_from_str(date_str, format) {
                if let Some(naive_dt) = naive.and_hms_opt(0, 0, 0) {
                    return Some(DateTime::from_naive_utc_and_offset(naive_dt, Utc));
                }
            }
            if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(date_str, format) {
                return Some(DateTime::from_naive_utc_and_offset(naive, Utc));
            }
        }

        None
    }
}

impl Default for ContentExtractor {
    fn default() -> Self {
        Self::new(ExtractorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_title() {
        let html = r#"
            <html>
            <head><title>Test Page Title</title></head>
            <body><h1>Main Heading</h1></body>
            </html>
        "#;

        let extractor = ContentExtractor::default();
        let document = Html::parse_document(html);
        let title = extractor.extract_title(&document);

        assert_eq!(title, "Test Page Title");
    }

    #[test]
    fn test_extract_content() {
        let html = r#"
            <html>
            <head><title>Test Article</title></head>
            <body>
                <nav>Navigation menu here</nav>
                <article>
                    <h1>Article Title</h1>
                    <p>This is the main article content. It has several paragraphs with enough text to meet the minimum requirements for extraction.</p>
                    <p>Second paragraph with more content to ensure we have enough words for the extraction to succeed properly.</p>
                </article>
                <footer>Footer content</footer>
            </body>
            </html>
        "#;

        let extractor = ContentExtractor::new(ExtractorConfig {
            min_content_length: 50,
            min_word_count: 10,
            ..Default::default()
        });

        let url = Url::parse("https://example.com/article").unwrap();
        let result = extractor.extract(html, &url);

        assert!(result.is_ok());
        let content = result.unwrap();
        assert!(!content.text_content.is_empty());
        assert!(content.word_count >= 10);
    }

    #[test]
    fn test_extract_metadata() {
        let html = r#"
            <html lang="en">
            <head>
                <title>Test Article</title>
                <meta property="og:title" content="OG Title">
                <meta property="og:description" content="OG Description">
                <meta name="author" content="John Doe">
                <meta property="article:published_time" content="2024-01-15T10:00:00Z">
            </head>
            <body>
                <article>
                    <p>Article content goes here with enough words to make this a proper article.</p>
                </article>
            </body>
            </html>
        "#;

        let extractor = ContentExtractor::default();
        let url = Url::parse("https://example.com/article").unwrap();
        let metadata = extractor.extract_metadata(html, &url);

        assert_eq!(metadata.title, "OG Title");
        assert_eq!(metadata.description, Some("OG Description".to_string()));
        assert_eq!(metadata.author, Some("John Doe".to_string()));
        assert_eq!(metadata.language, Some("en".to_string()));
        assert!(metadata.published_date.is_some());
    }

    #[test]
    fn test_parse_date() {
        assert!(ContentExtractor::parse_date("2024-01-15").is_some());
        assert!(ContentExtractor::parse_date("2024-01-15T10:00:00Z").is_some());
        assert!(ContentExtractor::parse_date("January 15, 2024").is_some());
    }
}
