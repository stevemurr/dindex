//! Plain text extraction
//!
//! Simple wrapper for plain text content.

use super::ExtractedDocument;

/// Plain text extractor
pub struct TextExtractor;

impl TextExtractor {
    /// Create an ExtractedDocument from plain text content
    pub fn extract(content: String) -> ExtractedDocument {
        let title = Self::extract_title(&content);
        let mut doc = ExtractedDocument::new(content);
        if let Some(t) = title {
            doc = doc.with_title(t);
        }
        doc
    }

    /// Try to extract title from markdown headers or first line
    fn extract_title(content: &str) -> Option<String> {
        for line in content.lines().take(10) {
            let trimmed = line.trim();

            // Check for markdown header
            if let Some(title) = trimmed.strip_prefix("# ") {
                if !title.is_empty() {
                    return Some(title.trim().to_string());
                }
            }

            // Check for underlined header (next line is === or ---)
            // Skip for now, would need to look ahead

            // First substantial non-header line could be title
            if trimmed.len() >= 10
                && trimmed.len() <= 200
                && !trimmed.starts_with('#')
                && !trimmed.starts_with("http")
                && !trimmed.starts_with("//")
                && !trimmed.starts_with("/*")
            {
                return Some(trimmed.to_string());
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_markdown_title() {
        let content = "# My Document Title\n\nSome content here.";
        let doc = TextExtractor::extract(content.to_string());
        assert_eq!(doc.title, Some("My Document Title".to_string()));
    }

    #[test]
    fn test_extract_first_line_title() {
        let content = "This Is The Document Title\n\nAnd this is the content.";
        let doc = TextExtractor::extract(content.to_string());
        assert_eq!(doc.title, Some("This Is The Document Title".to_string()));
    }

    #[test]
    fn test_preserves_content() {
        let content = "Some content here";
        let doc = TextExtractor::extract(content.to_string());
        assert_eq!(doc.content, "Some content here");
    }
}
