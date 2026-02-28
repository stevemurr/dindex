//! Content extraction from HTML
//!
//! Uses Mozilla's readability algorithm for content extraction:
//! - Removes boilerplate (navigation, ads, footers)
//! - Extracts main article content
//! - Parses metadata (title, author, date, etc.)

mod metadata;
mod scoring;
mod text;
mod types;

pub use types::*;

use scraper::{Html, Selector};
use std::collections::HashMap;
use std::io::Cursor;
use url::Url;

/// Content extractor
pub struct ContentExtractor {
    pub(crate) config: ExtractorConfig,
    /// Pre-compiled selectors for finding main content
    pub(crate) content_selectors: Vec<Selector>,
    /// Pre-compiled meta selectors: maps meta name → (name selector, property selector)
    pub(crate) meta_selectors: HashMap<String, (Option<Selector>, Option<Selector>)>,
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

        // Pre-compile meta selectors for all known meta names used across extraction
        let meta_names = [
            "og:title", "og:description", "og:type", "og:url", "og:image", "og:locale",
            "twitter:title", "twitter:description", "twitter:card", "twitter:site", "twitter:creator",
            "title", "description", "author", "keywords", "date", "language",
            "article:published_time",
        ];

        let mut meta_selectors = HashMap::with_capacity(meta_names.len());
        for name in &meta_names {
            let name_sel = Selector::parse(&format!("meta[name='{}']", name)).ok();
            let prop_sel = Selector::parse(&format!("meta[property='{}']", name)).ok();
            meta_selectors.insert(name.to_string(), (name_sel, prop_sel));
        }

        Self {
            config,
            content_selectors,
            meta_selectors,
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

    /// Extract both content and metadata from HTML, minimizing redundant parsing.
    ///
    /// Readability does its own internal HTML parsing (unavoidable), but all
    /// remaining DOM queries (author, date, language, OpenGraph, JSON-LD, etc.)
    /// share a single `Html::parse_document` call instead of parsing two or
    /// three times.
    pub fn extract_all(
        &self,
        html: &str,
        url: &Url,
    ) -> Result<(ExtractedContent, ExtractedMetadata), ExtractError> {
        // 1. Readability for main content (does its own internal parse)
        let mut cursor = Cursor::new(html.as_bytes());
        let product = readability::extractor::extract(&mut cursor, url)
            .map_err(|_| ExtractError::NoContent)?;

        let text_content = product.text;
        let title_from_readability = product.title;
        let clean_html = Some(product.content);

        if text_content.len() < self.config.min_content_length {
            return Err(ExtractError::TooShort(text_content.len()));
        }

        let word_count = text_content.split_whitespace().count();
        if word_count < self.config.min_word_count {
            return Err(ExtractError::TooShort(word_count));
        }

        let reading_time = ((word_count as f32 / self.config.words_per_minute as f32).ceil()
            as u8)
            .max(1);

        // 2. Parse the DOM once for all metadata queries
        let document = Html::parse_document(html);

        let author = self.extract_author(&document);
        let published_date = self.extract_date(&document);
        let language = self.extract_language(&document);
        let excerpt = self.extract_excerpt(&document, &text_content);

        let content = ExtractedContent {
            title: title_from_readability,
            text_content: text_content.clone(),
            clean_html,
            author: author.clone(),
            published_date,
            excerpt,
            language: language.clone(),
            word_count,
            reading_time_minutes: reading_time,
        };

        // 3. Build metadata from the same parsed document
        let metadata = self.build_metadata_from_document(
            &document,
            url,
            &text_content,
            word_count,
            reading_time,
            author,
            published_date,
            language,
        );

        Ok((content, metadata))
    }

    /// Extract full metadata from HTML
    pub fn extract_metadata(&self, html: &str, url: &Url) -> ExtractedMetadata {
        let document = Html::parse_document(html);

        // Calculate content metrics from DOM
        let text_content = self.extract_text(&self.find_main_content(&document));
        let word_count = text_content.split_whitespace().count();
        let reading_time =
            ((word_count as f32 / self.config.words_per_minute as f32).ceil() as u8).max(1);

        self.build_metadata_from_document(
            &document,
            url,
            &text_content,
            word_count,
            reading_time,
            None,
            None,
            None,
        )
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

    #[test]
    fn test_extract_text_pre_does_not_leak() {
        let extractor = ContentExtractor::default();
        let html = r#"<pre>  code  block  </pre><p>Normal paragraph text here.</p>"#;
        let text = extractor.extract_text(html);
        // After the <pre> block, normal whitespace handling should resume:
        // the paragraph text should be trimmed/collapsed, not preserved raw.
        assert!(
            text.contains("Normal paragraph text here."),
            "Text after <pre> should have normal whitespace handling, got: {:?}",
            text
        );
    }
}
