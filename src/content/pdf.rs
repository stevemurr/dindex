//! PDF text extraction
//!
//! Extracts text content from PDF documents using pdf-extract.

use super::ExtractedDocument;
use anyhow::{Context, Result};

/// PDF content extractor
pub struct PdfExtractor;

impl PdfExtractor {
    /// Extract text content from PDF bytes
    pub fn extract(bytes: &[u8]) -> Result<ExtractedDocument> {
        // Suppress noisy stderr output from pdf-extract library
        // (Unicode mismatch warnings, unknown glyph names, etc.)
        let text = Self::extract_with_suppressed_stderr(bytes)
            .context("Failed to extract text from PDF")?;

        // Clean up the extracted text
        let cleaned = Self::clean_text(&text);

        if cleaned.is_empty() {
            anyhow::bail!("PDF contains no extractable text (may be image-only)");
        }

        let title = Self::extract_title(&cleaned);

        let mut doc = ExtractedDocument::new(cleaned);
        if let Some(t) = title {
            doc = doc.with_title(t);
        }

        Ok(doc)
    }

    /// Extract text while suppressing stdout/stderr noise from pdf-extract
    ///
    /// NOTE: The FD redirection below is not thread-safe — it temporarily redirects
    /// process-wide stdout/stderr, so other threads may lose output during extraction.
    /// This is acceptable because pdf-extract's println! noise is purely cosmetic
    /// (Unicode mismatch warnings, unknown glyph names) and the window is brief.
    fn extract_with_suppressed_stderr(bytes: &[u8]) -> Result<String, pdf_extract::OutputError> {
        // pdf-extract uses println! for warnings about Unicode mismatches,
        // unknown glyphs, missing chars, etc. These are not actionable by users
        // and clutter the output. We redirect both stdout and stderr to /dev/null.

        #[cfg(unix)]
        {
            use std::fs::File;
            use std::os::unix::io::IntoRawFd;

            const STDOUT_FD: i32 = 1;
            const STDERR_FD: i32 = 2;

            // Save original stdout and stderr
            let saved_stdout = unsafe { libc::dup(STDOUT_FD) };
            let saved_stderr = unsafe { libc::dup(STDERR_FD) };

            // Open /dev/null and redirect both stdout and stderr to it
            if let Ok(dev_null) = File::open("/dev/null") {
                let null_fd = dev_null.into_raw_fd();
                unsafe { libc::dup2(null_fd, STDOUT_FD) };
                unsafe { libc::dup2(null_fd, STDERR_FD) };
                unsafe { libc::close(null_fd) };
            }

            // Perform the extraction
            let result = pdf_extract::extract_text_from_mem(bytes);

            // Restore original stdout and stderr
            if saved_stdout >= 0 {
                unsafe { libc::dup2(saved_stdout, STDOUT_FD) };
                unsafe { libc::close(saved_stdout) };
            }
            if saved_stderr >= 0 {
                unsafe { libc::dup2(saved_stderr, STDERR_FD) };
                unsafe { libc::close(saved_stderr) };
            }

            result
        }

        #[cfg(not(unix))]
        {
            // On non-Unix platforms, just run without suppression
            pdf_extract::extract_text_from_mem(bytes)
        }
    }

    /// Clean up common PDF extraction artifacts
    fn clean_text(text: &str) -> String {
        text.lines()
            // Trim whitespace from each line
            .map(|l| l.trim())
            // Remove empty lines but preserve paragraph breaks
            .fold(Vec::new(), |mut acc, line| {
                if line.is_empty() {
                    // Only add empty line if previous wasn't empty
                    if acc.last().map(|l: &String| !l.is_empty()).unwrap_or(false) {
                        acc.push(String::new());
                    }
                } else {
                    acc.push(line.to_string());
                }
                acc
            })
            .join("\n")
            .trim()
            .to_string()
    }

    /// Try to extract title from the first lines of the document
    fn extract_title(text: &str) -> Option<String> {
        // Look for a substantial first line that could be a title
        for line in text.lines().take(5) {
            let trimmed = line.trim();
            // Title should be:
            // - Not too short (likely page number or header)
            // - Not too long (likely paragraph text)
            // - Not starting with common non-title patterns
            if trimmed.len() >= 10
                && trimmed.len() <= 200
                && !trimmed.starts_with("http")
                && !trimmed.starts_with("www.")
                && !trimmed.chars().all(|c| c.is_numeric() || c.is_whitespace())
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
    fn test_clean_text() {
        let dirty = "  Line 1  \n\n\n  Line 2  \n  \n  Line 3  ";
        let cleaned = PdfExtractor::clean_text(dirty);
        assert_eq!(cleaned, "Line 1\n\nLine 2\n\nLine 3");
    }

    #[test]
    fn test_extract_title() {
        let text = "A Very Important Research Paper Title\n\nAuthors: John Doe\n\nAbstract...";
        let title = PdfExtractor::extract_title(text);
        assert_eq!(title, Some("A Very Important Research Paper Title".to_string()));
    }

    #[test]
    fn test_extract_title_skips_short() {
        let text = "123\n\nActual Title Here\n\nContent...";
        let title = PdfExtractor::extract_title(text);
        assert_eq!(title, Some("Actual Title Here".to_string()));
    }
}
