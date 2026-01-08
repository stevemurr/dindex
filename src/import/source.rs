//! Core types and traits for bulk import functionality

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// A document extracted from a dump
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DumpDocument {
    /// Unique identifier within the dump (e.g., Wikipedia page ID)
    pub id: String,
    /// Document title
    pub title: String,
    /// Clean plaintext content
    pub content: String,
    /// Source URL (reconstructed for Wikipedia, actual for WARC)
    pub url: Option<String>,
    /// Last modification timestamp
    pub modified: Option<DateTime<Utc>>,
    /// Source-specific metadata
    pub metadata: HashMap<String, String>,
}

impl DumpDocument {
    /// Create a new dump document
    pub fn new(id: impl Into<String>, title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            content: content.into(),
            url: None,
            modified: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the URL
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set the modification timestamp
    pub fn with_modified(mut self, modified: DateTime<Utc>) -> Self {
        self.modified = Some(modified);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Convert to the standard Document type for indexing
    pub fn to_document(&self) -> crate::types::Document {
        let mut doc = crate::types::Document::new(&self.content)
            .with_id(&self.id)
            .with_title(&self.title);

        if let Some(ref url) = self.url {
            doc = doc.with_url(url);
        }

        // Copy metadata
        doc.metadata = self.metadata.clone();

        doc
    }
}

/// Trait for dump sources that can be iterated over
pub trait DumpSource: Send {
    /// Iterate over documents in the dump
    fn iter_documents(&mut self) -> Box<dyn Iterator<Item = Result<DumpDocument, ImportError>> + '_>;

    /// Get total document count if known (for progress reporting)
    fn document_count_hint(&self) -> Option<u64>;

    /// Get current byte position (for resume support)
    fn byte_position(&self) -> u64;

    /// Seek to byte position (for resume support)
    fn seek_to(&mut self, position: u64) -> Result<(), ImportError>;

    /// Get the source name for display
    fn source_name(&self) -> &str;
}

/// Import configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConfig {
    /// Batch size for indexing
    pub batch_size: usize,
    /// Enable content deduplication
    pub deduplicate: bool,
    /// Checkpoint interval (documents)
    pub checkpoint_interval: usize,
    /// Checkpoint file path
    pub checkpoint_path: Option<PathBuf>,
    /// Filter: minimum content length
    pub min_content_length: usize,
    /// Filter: namespace allowlist (Wikipedia-specific)
    pub allowed_namespaces: Option<Vec<i32>>,
    /// Maximum documents to import (None = unlimited)
    pub max_documents: Option<usize>,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            deduplicate: true,
            checkpoint_interval: 1000,
            checkpoint_path: None,
            min_content_length: 100,
            allowed_namespaces: Some(vec![0]), // Main namespace only by default
            max_documents: None,
        }
    }
}

/// Import statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportStats {
    /// Total documents processed
    pub documents_processed: usize,
    /// Documents successfully imported
    pub documents_imported: usize,
    /// Documents skipped (filtered or duplicate)
    pub documents_skipped: usize,
    /// Documents with errors
    pub documents_errored: usize,
    /// Total chunks created
    pub chunks_created: usize,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Processing time in seconds
    pub elapsed_seconds: f64,
    /// Current documents per second rate
    pub docs_per_second: f64,
}

impl ImportStats {
    /// Calculate documents per second
    pub fn update_rate(&mut self) {
        if self.elapsed_seconds > 0.0 {
            self.docs_per_second = self.documents_processed as f64 / self.elapsed_seconds;
        }
    }
}

/// Checkpoint for resume support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportCheckpoint {
    /// Source file path
    pub source_path: PathBuf,
    /// Byte position in source
    pub byte_position: u64,
    /// Documents processed so far
    pub documents_processed: usize,
    /// Documents imported so far
    pub documents_imported: usize,
    /// Timestamp of checkpoint
    pub timestamp: DateTime<Utc>,
}

impl ImportCheckpoint {
    /// Create a new checkpoint
    pub fn new(source_path: PathBuf, byte_position: u64, stats: &ImportStats) -> Self {
        Self {
            source_path,
            byte_position,
            documents_processed: stats.documents_processed,
            documents_imported: stats.documents_imported,
            timestamp: Utc::now(),
        }
    }

    /// Save checkpoint to file
    pub fn save(&self, path: &PathBuf) -> Result<(), ImportError> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &PathBuf) -> Result<Self, ImportError> {
        let json = std::fs::read_to_string(path)?;
        let checkpoint = serde_json::from_str(&json)?;
        Ok(checkpoint)
    }
}

/// Errors that can occur during import
#[derive(Debug, Error)]
pub enum ImportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XML parse error: {0}")]
    XmlParse(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Invalid dump format: {0}")]
    InvalidFormat(String),

    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("UTF-8 decode error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("Parse error: {0}")]
    Parse(String),
}

impl From<quick_xml::Error> for ImportError {
    fn from(e: quick_xml::Error) -> Self {
        ImportError::XmlParse(e.to_string())
    }
}

/// Dump format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DumpFormat {
    /// Wikimedia XML dump (compressed with bzip2)
    WikimediaXml,
    /// ZIM file (Kiwix format)
    Zim,
    /// WARC web archive
    Warc,
    /// Plain text files
    PlainText,
}

impl DumpFormat {
    /// Detect format from file path
    pub fn detect(path: &std::path::Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?;
        let name_lower = name.to_lowercase();

        if name_lower.contains("wiki") && name_lower.ends_with(".xml.bz2") {
            Some(DumpFormat::WikimediaXml)
        } else if name_lower.ends_with(".xml.bz2") {
            Some(DumpFormat::WikimediaXml)
        } else if name_lower.ends_with(".zim") {
            Some(DumpFormat::Zim)
        } else if name_lower.ends_with(".warc") || name_lower.ends_with(".warc.gz") {
            Some(DumpFormat::Warc)
        } else if name_lower.ends_with(".txt") || name_lower.ends_with(".md") {
            Some(DumpFormat::PlainText)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_dump_document_creation() {
        let doc = DumpDocument::new("123", "Test Article", "This is test content.")
            .with_url("https://example.com/test")
            .with_metadata("namespace", "0");

        assert_eq!(doc.id, "123");
        assert_eq!(doc.title, "Test Article");
        assert_eq!(doc.content, "This is test content.");
        assert_eq!(doc.url, Some("https://example.com/test".to_string()));
        assert_eq!(doc.metadata.get("namespace"), Some(&"0".to_string()));
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            DumpFormat::detect(Path::new("enwiki-latest-pages-articles.xml.bz2")),
            Some(DumpFormat::WikimediaXml)
        );
        assert_eq!(
            DumpFormat::detect(Path::new("wikipedia_en_all.zim")),
            Some(DumpFormat::Zim)
        );
        assert_eq!(
            DumpFormat::detect(Path::new("archive.warc.gz")),
            Some(DumpFormat::Warc)
        );
        assert_eq!(
            DumpFormat::detect(Path::new("document.txt")),
            Some(DumpFormat::PlainText)
        );
    }

    #[test]
    fn test_import_config_defaults() {
        let config = ImportConfig::default();
        assert_eq!(config.batch_size, 100);
        assert!(config.deduplicate);
        assert_eq!(config.min_content_length, 100);
    }
}
