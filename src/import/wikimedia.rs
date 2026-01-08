//! Wikimedia XML dump parser
//!
//! Parses Wikimedia XML dump files (typically compressed with bzip2) and yields documents.

use super::source::{DumpDocument, DumpSource, ImportError};
use super::wikitext::WikiTextParser;
use bzip2::read::BzDecoder;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// Wikimedia XML dump source
pub struct WikimediaSource {
    /// Path to the dump file
    path: PathBuf,
    /// XML reader (wrapped in various decompression layers)
    reader: WikimediaReader,
    /// WikiText parser
    wikitext_parser: WikiTextParser,
    /// Current page being parsed
    current_page: Option<PartialPage>,
    /// Bytes read so far (approximate)
    bytes_read: u64,
    /// Allowed namespaces (None = all, Some([]) = none, Some([0]) = main only)
    allowed_namespaces: Option<HashSet<i32>>,
    /// Base URL for Wikipedia (to construct URLs)
    base_url: String,
}

/// Reader abstraction for different compression formats
enum WikimediaReader {
    /// Bzip2 compressed
    Bzip2(Reader<BufReader<BzDecoder<File>>>),
    /// Uncompressed XML
    Plain(Reader<BufReader<File>>),
}

impl WikimediaReader {
    fn read_event<'a>(&mut self, buf: &'a mut Vec<u8>) -> Result<Event<'a>, quick_xml::Error> {
        buf.clear();
        match self {
            WikimediaReader::Bzip2(reader) => reader.read_event_into(buf),
            WikimediaReader::Plain(reader) => reader.read_event_into(buf),
        }
    }
}

/// Partial page being built from XML events
#[derive(Debug, Default)]
struct PartialPage {
    title: Option<String>,
    id: Option<String>,
    namespace: Option<i32>,
    text: Option<String>,
    timestamp: Option<String>,
    redirect: bool,
}

/// Result of parsing a page from the XML stream
enum ParseResult {
    /// Successfully parsed a document
    Document(DumpDocument),
    /// Page was skipped (redirect, filtered namespace, too short, etc.)
    Skipped,
    /// End of file reached
    Eof,
}

impl WikimediaSource {
    /// Open a Wikimedia XML dump file
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ImportError> {
        let path = path.as_ref().to_path_buf();

        let file = File::open(&path)?;
        let is_bz2 = path
            .extension()
            .map(|e| e == "bz2")
            .unwrap_or(false)
            || path.to_string_lossy().ends_with(".xml.bz2");

        let reader = if is_bz2 {
            let decoder = BzDecoder::new(file);
            let buf_reader = BufReader::with_capacity(1024 * 1024, decoder); // 1MB buffer
            let xml_reader = Reader::from_reader(buf_reader);
            WikimediaReader::Bzip2(xml_reader)
        } else {
            let buf_reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer
            let xml_reader = Reader::from_reader(buf_reader);
            WikimediaReader::Plain(xml_reader)
        };

        // Detect Wikipedia language from filename
        let filename = path.file_name().unwrap_or_default().to_string_lossy();
        let base_url = if filename.starts_with("enwiki") {
            "https://en.wikipedia.org/wiki/".to_string()
        } else if filename.starts_with("dewiki") {
            "https://de.wikipedia.org/wiki/".to_string()
        } else if filename.starts_with("frwiki") {
            "https://fr.wikipedia.org/wiki/".to_string()
        } else if filename.starts_with("eswiki") {
            "https://es.wikipedia.org/wiki/".to_string()
        } else if filename.starts_with("simplewiki") {
            "https://simple.wikipedia.org/wiki/".to_string()
        } else {
            "https://en.wikipedia.org/wiki/".to_string() // Default to English
        };

        Ok(Self {
            path,
            reader,
            wikitext_parser: WikiTextParser::new(),
            current_page: None,
            bytes_read: 0,
            allowed_namespaces: Some(HashSet::from([0])), // Main namespace by default
            base_url,
        })
    }

    /// Set allowed namespaces (None = all namespaces)
    pub fn with_namespaces(mut self, namespaces: Option<Vec<i32>>) -> Self {
        self.allowed_namespaces = namespaces.map(|ns| ns.into_iter().collect());
        self
    }

    /// Set the base URL for constructing article URLs
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Parse the next page from the XML stream
    fn parse_next_page(&mut self) -> Result<ParseResult, ImportError> {
        let mut buf = Vec::with_capacity(8192);
        let mut text_buf = String::new();
        let mut current_element: Option<String> = None;

        loop {
            let event = self.reader.read_event(&mut buf)?;

            match event {
                Event::Start(ref e) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    match name.as_str() {
                        "page" => {
                            self.current_page = Some(PartialPage::default());
                        }
                        "redirect" => {
                            if let Some(ref mut page) = self.current_page {
                                page.redirect = true;
                            }
                        }
                        "title" | "id" | "ns" | "text" | "timestamp" => {
                            current_element = Some(name);
                            text_buf.clear();
                        }
                        _ => {}
                    }
                }
                Event::Text(ref e) => {
                    if current_element.is_some() {
                        if let Ok(text) = e.unescape() {
                            text_buf.push_str(&text);
                        }
                    }
                }
                Event::CData(ref e) => {
                    if current_element.is_some() {
                        if let Ok(text) = String::from_utf8(e.to_vec()) {
                            text_buf.push_str(&text);
                        }
                    }
                }
                Event::End(ref e) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    if let Some(ref mut page) = self.current_page {
                        match name.as_str() {
                            "title" => {
                                page.title = Some(text_buf.clone());
                                current_element = None;
                            }
                            "id" => {
                                // Only set ID if it's the page ID (first ID in page)
                                if page.id.is_none() {
                                    page.id = Some(text_buf.clone());
                                }
                                current_element = None;
                            }
                            "ns" => {
                                page.namespace = text_buf.parse().ok();
                                current_element = None;
                            }
                            "text" => {
                                page.text = Some(text_buf.clone());
                                current_element = None;
                            }
                            "timestamp" => {
                                page.timestamp = Some(text_buf.clone());
                                current_element = None;
                            }
                            "page" => {
                                // Page complete, convert to document
                                let page = self.current_page.take().unwrap();
                                return Ok(self.page_to_document(page));
                            }
                            _ => {}
                        }
                    }
                }
                Event::Eof => {
                    return Ok(ParseResult::Eof);
                }
                _ => {}
            }

            buf.clear();
        }
    }

    /// Convert a parsed page to a document (or Skipped if filtered)
    fn page_to_document(&self, page: PartialPage) -> ParseResult {
        // Filter by namespace
        if let Some(ref allowed) = self.allowed_namespaces {
            let ns = page.namespace.unwrap_or(0);
            if !allowed.contains(&ns) {
                return ParseResult::Skipped;
            }
        }

        // Skip redirects
        if page.redirect {
            return ParseResult::Skipped;
        }

        // Need title, id, and text
        let title = match page.title {
            Some(t) => t,
            None => return ParseResult::Skipped,
        };
        let id = match page.id {
            Some(i) => i,
            None => return ParseResult::Skipped,
        };
        let wikitext = match page.text {
            Some(t) => t,
            None => return ParseResult::Skipped,
        };

        // Skip empty or very short articles
        if wikitext.len() < 100 {
            return ParseResult::Skipped;
        }

        // Convert WikiText to plaintext
        let content = self.wikitext_parser.parse(&wikitext);

        // Skip if content is too short after parsing
        if content.len() < 50 {
            return ParseResult::Skipped;
        }

        // Build URL
        let url = format!(
            "{}{}",
            self.base_url,
            title.replace(' ', "_")
        );

        // Parse timestamp
        let modified = page.timestamp.and_then(|ts| {
            chrono::DateTime::parse_from_rfc3339(&ts)
                .ok()
                .map(|dt| dt.with_timezone(&chrono::Utc))
        });

        let mut doc = DumpDocument::new(id, title, content).with_url(url);

        if let Some(ts) = modified {
            doc = doc.with_modified(ts);
        }

        if let Some(ns) = page.namespace {
            doc = doc.with_metadata("namespace", ns.to_string());
        }

        ParseResult::Document(doc)
    }
}

impl DumpSource for WikimediaSource {
    fn iter_documents(&mut self) -> Box<dyn Iterator<Item = Result<DumpDocument, ImportError>> + '_> {
        Box::new(WikimediaIterator { source: self })
    }

    fn document_count_hint(&self) -> Option<u64> {
        // Could estimate from file size, but not reliable
        // English Wikipedia has ~6.7 million articles
        None
    }

    fn byte_position(&self) -> u64 {
        self.bytes_read
    }

    fn seek_to(&mut self, position: u64) -> Result<(), ImportError> {
        // Seeking in bzip2 streams is not efficient
        // For resume, we'd need to track page boundaries
        // For now, just track position
        self.bytes_read = position;
        Ok(())
    }

    fn source_name(&self) -> &str {
        self.path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("wikimedia dump")
    }
}

/// Iterator over documents in a Wikimedia dump
struct WikimediaIterator<'a> {
    source: &'a mut WikimediaSource,
}

impl<'a> Iterator for WikimediaIterator<'a> {
    type Item = Result<DumpDocument, ImportError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.source.parse_next_page() {
                Ok(ParseResult::Document(doc)) => return Some(Ok(doc)),
                Ok(ParseResult::Skipped) => continue, // Filtered page, keep going
                Ok(ParseResult::Eof) => return None,  // End of file
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Helper to create a source for testing with plain XML
pub fn from_xml_string(xml: &str) -> Result<WikimediaSource, ImportError> {
    use std::io::Write;

    // Write to a temp file
    let mut temp_file = tempfile::NamedTempFile::new()?;
    temp_file.write_all(xml.as_bytes())?;
    temp_file.flush()?;

    let path = temp_file.path().to_path_buf();

    // Keep the temp file around by leaking it
    // (In production code, you'd want to handle this differently)
    let _ = temp_file.into_temp_path();

    let file = File::open(&path)?;
    let buf_reader = BufReader::new(file);
    let xml_reader = Reader::from_reader(buf_reader);

    Ok(WikimediaSource {
        path,
        reader: WikimediaReader::Plain(xml_reader),
        wikitext_parser: WikiTextParser::new(),
        current_page: None,
        bytes_read: 0,
        allowed_namespaces: Some(HashSet::from([0])),
        base_url: "https://en.wikipedia.org/wiki/".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>Test Article</title>
    <ns>0</ns>
    <id>12345</id>
    <revision>
      <id>100</id>
      <timestamp>2024-01-15T10:30:00Z</timestamp>
      <text>This is a '''test article''' about [[testing]]. It contains various types of wiki markup that need to be properly parsed and converted to plain text.

The article discusses multiple topics including software development, quality assurance, and documentation standards. Testing is a fundamental part of the software development lifecycle.

== Section ==
More content here that helps demonstrate how the parser handles different types of wiki markup. This section contains enough text to ensure the article passes minimum length requirements for indexing.

== Another Section ==
Additional content with more details about the testing methodology used in this example article.</text>
    </revision>
  </page>
  <page>
    <title>Another Article</title>
    <ns>0</ns>
    <id>12346</id>
    <revision>
      <id>101</id>
      <text>Another article with enough content to be indexed properly. This article contains several paragraphs of text that discuss various topics related to knowledge management and information retrieval systems.

The content here is designed to pass the minimum length requirements while also demonstrating how the parser handles plain text without much wiki markup.</text>
    </revision>
  </page>
  <page>
    <title>Talk:Test Article</title>
    <ns>1</ns>
    <id>12347</id>
    <revision>
      <id>102</id>
      <text>Discussion about the test article. This talk page contains conversations between editors about how to improve the main article content. There are several points raised about accuracy, citations, and style guidelines that need to be addressed.</text>
    </revision>
  </page>
</mediawiki>
"#;

    #[test]
    fn test_parse_sample_xml() {
        // Write sample XML to temp file
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut temp_file, SAMPLE_XML.as_bytes()).unwrap();

        let mut source = WikimediaSource::open(temp_file.path()).unwrap();
        let docs: Vec<_> = source.iter_documents().collect();

        // Should have 2 main namespace articles (Talk page filtered)
        assert_eq!(docs.len(), 2);

        let doc1 = docs[0].as_ref().unwrap();
        assert_eq!(doc1.id, "12345");
        assert_eq!(doc1.title, "Test Article");
        assert!(doc1.content.contains("test article"));
        assert!(!doc1.content.contains("'''")); // Bold markup removed

        let doc2 = docs[1].as_ref().unwrap();
        assert_eq!(doc2.id, "12346");
        assert_eq!(doc2.title, "Another Article");
    }

    #[test]
    fn test_namespace_filtering() {
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut temp_file, SAMPLE_XML.as_bytes()).unwrap();

        // Allow all namespaces
        let mut source = WikimediaSource::open(temp_file.path())
            .unwrap()
            .with_namespaces(None);

        let docs: Vec<_> = source.iter_documents().collect();

        // Should have all 3 articles
        assert_eq!(docs.len(), 3);
    }

    #[test]
    fn test_url_generation() {
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut temp_file, SAMPLE_XML.as_bytes()).unwrap();

        let mut source = WikimediaSource::open(temp_file.path()).unwrap();
        let docs: Vec<_> = source.iter_documents().collect();

        let doc = docs[0].as_ref().unwrap();
        assert_eq!(doc.url, Some("https://en.wikipedia.org/wiki/Test_Article".to_string()));
    }
}
