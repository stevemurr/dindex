//! HTML content extraction using readability
//!
//! Extracts clean readable content from HTML pages using Mozilla's readability algorithm.

use super::ExtractedDocument;
use anyhow::{Context, Result};
use std::io::Cursor;
use url::Url;

/// HTML extractor using readability
pub struct HtmlExtractor;

impl HtmlExtractor {
    /// Extract readable content from HTML bytes
    pub fn extract(html: &[u8], url: Option<&str>) -> Result<ExtractedDocument> {
        let html_str = String::from_utf8_lossy(html);
        Self::extract_from_str(&html_str, url)
    }

    /// Extract readable content from an HTML string
    pub fn extract_from_str(html: &str, url: Option<&str>) -> Result<ExtractedDocument> {
        // Parse the URL or use a placeholder
        let parsed_url = url
            .and_then(|u| Url::parse(u).ok())
            .unwrap_or_else(|| Url::parse("http://localhost/").unwrap());

        let mut cursor = Cursor::new(html.as_bytes());

        let product = readability::extractor::extract(&mut cursor, &parsed_url)
            .context("Failed to extract readable content from HTML")?;

        let mut doc = ExtractedDocument::new(product.text);

        if !product.title.is_empty() {
            doc = doc.with_title(product.title);
        }

        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_simple_html() {
        let html = r#"
            <!DOCTYPE html>
            <html>
            <head><title>Test Article</title></head>
            <body>
                <nav>Navigation here</nav>
                <article>
                    <h1>Main Article Title</h1>
                    <p>This is the main content of the article. It contains important information
                    that should be extracted by the readability algorithm.</p>
                    <p>Here is another paragraph with more content to ensure proper extraction.</p>
                </article>
                <footer>Footer content here</footer>
            </body>
            </html>
        "#;

        let result = HtmlExtractor::extract_from_str(html, Some("https://example.com/article"));
        assert!(result.is_ok());

        let doc = result.unwrap();
        assert!(!doc.content.is_empty());
    }
}
