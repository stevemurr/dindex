//! Content extraction module
//!
//! Provides unified content extraction from various file formats (PDF, text, HTML, etc.)
//! for both local files and remote URLs.

mod html;
mod pdf;
mod text;

pub use html::HtmlExtractor;
pub use pdf::PdfExtractor;
pub use text::TextExtractor;

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Extracted document content with metadata
#[derive(Debug, Clone)]
pub struct ExtractedDocument {
    /// The extracted text content
    pub content: String,
    /// Document title (if detected)
    pub title: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ExtractedDocument {
    /// Create a new extracted document
    pub fn new(content: String) -> Self {
        Self {
            content,
            title: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Supported content types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Plain text (.txt, .md, etc.)
    Text,
    /// PDF document
    Pdf,
    /// HTML content
    Html,
    /// Unknown/unsupported
    Unknown,
}

impl ContentType {
    /// Detect content type from file extension
    pub fn from_extension(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("pdf") => ContentType::Pdf,
            Some("txt" | "md" | "markdown" | "rst" | "text") => ContentType::Text,
            Some("html" | "htm" | "xhtml") => ContentType::Html,
            _ => ContentType::Unknown,
        }
    }

    /// Detect content type from MIME type
    pub fn from_mime(mime: &str) -> Self {
        let mime = mime.to_lowercase();
        if mime.contains("application/pdf") {
            ContentType::Pdf
        } else if mime.contains("text/plain") {
            ContentType::Text
        } else if mime.contains("text/html") || mime.contains("application/xhtml") {
            ContentType::Html
        } else {
            ContentType::Unknown
        }
    }

    /// Detect content type from URL path
    pub fn from_url(url: &str) -> Self {
        if url.ends_with(".pdf") {
            ContentType::Pdf
        } else if url.ends_with(".txt") || url.ends_with(".md") {
            ContentType::Text
        } else if url.ends_with(".html") || url.ends_with(".htm") {
            ContentType::Html
        } else {
            // Default to HTML for web pages
            ContentType::Html
        }
    }
}

/// Extract content from a local file
pub fn extract_from_path(path: &Path) -> Result<ExtractedDocument> {
    let content_type = ContentType::from_extension(path);

    match content_type {
        ContentType::Pdf => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("Failed to read file: {}", path.display()))?;
            PdfExtractor::extract(&bytes)
        }
        ContentType::Text | ContentType::Unknown => {
            // Try to read as text
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read file as text: {}", path.display()))?;
            Ok(TextExtractor::extract(content))
        }
        ContentType::Html => {
            // For local HTML files, just read as text for now
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read HTML file: {}", path.display()))?;
            Ok(TextExtractor::extract(content))
        }
    }
}

/// Extract content from bytes with a known content type
///
/// For HTML content, an optional URL can be provided to help with relative link resolution.
pub fn extract_from_bytes(bytes: &[u8], content_type: ContentType) -> Result<ExtractedDocument> {
    extract_from_bytes_with_url(bytes, content_type, None)
}

/// Extract content from bytes with a known content type and optional source URL
///
/// For HTML content, the URL helps with relative link resolution and metadata extraction.
pub fn extract_from_bytes_with_url(
    bytes: &[u8],
    content_type: ContentType,
    url: Option<&str>,
) -> Result<ExtractedDocument> {
    match content_type {
        ContentType::Pdf => PdfExtractor::extract(bytes),
        ContentType::Html => HtmlExtractor::extract(bytes, url),
        ContentType::Text | ContentType::Unknown => {
            let content = String::from_utf8_lossy(bytes).to_string();
            Ok(TextExtractor::extract(content))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_content_type_from_extension() {
        assert_eq!(
            ContentType::from_extension(&PathBuf::from("test.pdf")),
            ContentType::Pdf
        );
        assert_eq!(
            ContentType::from_extension(&PathBuf::from("test.txt")),
            ContentType::Text
        );
        assert_eq!(
            ContentType::from_extension(&PathBuf::from("test.md")),
            ContentType::Text
        );
        assert_eq!(
            ContentType::from_extension(&PathBuf::from("test.html")),
            ContentType::Html
        );
        assert_eq!(
            ContentType::from_extension(&PathBuf::from("test.unknown")),
            ContentType::Unknown
        );
    }

    #[test]
    fn test_content_type_from_mime() {
        assert_eq!(
            ContentType::from_mime("application/pdf"),
            ContentType::Pdf
        );
        assert_eq!(
            ContentType::from_mime("text/plain; charset=utf-8"),
            ContentType::Text
        );
        assert_eq!(
            ContentType::from_mime("text/html"),
            ContentType::Html
        );
    }

    #[test]
    fn test_content_type_from_url() {
        assert_eq!(
            ContentType::from_url("https://arxiv.org/pdf/2407.21075.pdf"),
            ContentType::Pdf
        );
        assert_eq!(
            ContentType::from_url("https://example.com/doc.txt"),
            ContentType::Text
        );
        assert_eq!(
            ContentType::from_url("https://example.com/page"),
            ContentType::Html
        );
    }
}
