//! Document splitting into semantic chunks

use crate::config::ChunkingConfig;
use crate::types::{Chunk, ChunkMetadata, Document};
use unicode_segmentation::UnicodeSegmentation;
use tracing::debug;

/// Text splitter for creating semantic chunks
pub struct TextSplitter {
    config: ChunkingConfig,
    /// Approximate tokens per character (varies by tokenizer)
    chars_per_token: f32,
}

impl TextSplitter {
    /// Create a new text splitter
    pub fn new(config: ChunkingConfig) -> Self {
        Self {
            config,
            chars_per_token: 4.0, // Rough approximation
        }
    }

    /// Split a document into chunks
    pub fn split_document(&self, document: &Document) -> Vec<Chunk> {
        let content = &document.content;
        if content.is_empty() {
            return Vec::new();
        }

        // Detect sections
        let sections = self.detect_sections(content);

        // Split into chunks
        let mut chunks = Vec::new();
        let mut chunk_idx = 0;
        let total_chars = content.chars().count();

        for section in &sections {
            let section_chunks = self.split_section(
                &section.content,
                &document.id,
                &section.hierarchy,
                chunk_idx,
                total_chars,
            );

            chunk_idx += section_chunks.len();
            chunks.extend(section_chunks);
        }

        // Link chunks (preceding/following)
        self.link_chunks(&mut chunks);

        // Set document metadata
        for chunk in &mut chunks {
            chunk.metadata.source_url = document.url.clone();
            chunk.metadata.source_title = document.title.clone();
        }

        debug!(
            "Split document {} into {} chunks",
            document.id,
            chunks.len()
        );

        chunks
    }

    /// Detect sections in content based on headings
    fn detect_sections(&self, content: &str) -> Vec<Section> {
        let lines: Vec<&str> = content.lines().collect();
        let mut sections = Vec::new();
        let mut current_hierarchy: Vec<String> = Vec::new();
        let mut current_content = String::new();
        for line in &lines {
            // Check for markdown headings
            if let Some(heading) = self.parse_heading(line) {
                // Save previous section if it has content
                if !current_content.trim().is_empty() {
                    sections.push(Section {
                        hierarchy: current_hierarchy.clone(),
                        content: current_content.trim().to_string(),
                    });
                }

                // Update hierarchy based on heading level
                let level = heading.level;
                while current_hierarchy.len() >= level {
                    current_hierarchy.pop();
                }
                current_hierarchy.push(heading.text.clone());

                current_content = String::new();
            } else {
                if !current_content.is_empty() {
                    current_content.push('\n');
                }
                current_content.push_str(line);
            }
        }

        // Add final section
        if !current_content.trim().is_empty() {
            sections.push(Section {
                hierarchy: current_hierarchy,
                content: current_content.trim().to_string(),
            });
        }

        // If no sections detected, treat entire content as one section
        if sections.is_empty() {
            sections.push(Section {
                hierarchy: Vec::new(),
                content: content.to_string(),
            });
        }

        sections
    }

    /// Parse a markdown heading from a line
    fn parse_heading(&self, line: &str) -> Option<Heading> {
        let trimmed = line.trim();

        // Markdown ATX headings (# Heading)
        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|c| *c == '#').count();
            if level <= 6 {
                let text = trimmed[level..].trim().trim_end_matches('#').trim();
                if !text.is_empty() {
                    return Some(Heading {
                        level,
                        text: text.to_string(),
                    });
                }
            }
        }

        None
    }

    /// Split a section into chunks
    fn split_section(
        &self,
        content: &str,
        document_id: &str,
        hierarchy: &[String],
        start_idx: usize,
        total_chars: usize,
    ) -> Vec<Chunk> {
        if content.is_empty() {
            return Vec::new();
        }

        let target_chars = (self.config.chunk_size as f32 * self.chars_per_token) as usize;
        let overlap_chars = (target_chars as f32 * self.config.overlap_fraction) as usize;
        let min_chars = (self.config.min_chunk_size as f32 * self.chars_per_token) as usize;

        let mut chunks = Vec::new();
        let mut start = 0;
        let content_chars: Vec<char> = content.chars().collect();
        let content_len = content_chars.len();

        while start < content_len {
            let end = (start + target_chars).min(content_len);

            // Find a good split point (sentence or paragraph boundary)
            let split_end = self.find_split_point(&content_chars, start, end, content_len);

            // Extract chunk text
            let chunk_text: String = content_chars[start..split_end].iter().collect();
            let chunk_text = chunk_text.trim().to_string();

            if chunk_text.len() >= min_chars || start + target_chars >= content_len {
                let position = if total_chars > 0 {
                    start as f32 / total_chars as f32
                } else {
                    0.0
                };
                let token_count = self.estimate_tokens(&chunk_text);

                let metadata = ChunkMetadata {
                    chunk_id: format!("{}_{}", document_id, start_idx + chunks.len()),
                    document_id: document_id.to_string(),
                    source_url: None,
                    source_title: None,
                    timestamp: chrono::Utc::now(),
                    position_in_doc: position,
                    section_hierarchy: hierarchy.to_vec(),
                    preceding_chunk_id: None,
                    following_chunk_id: None,
                    node_id: None,
                    extra: std::collections::HashMap::new(),
                };

                chunks.push(Chunk {
                    metadata,
                    content: chunk_text,
                    token_count,
                });
            }

            // Move start with overlap
            if split_end >= content_len {
                break;
            }
            start = (split_end - overlap_chars).max(start + 1);
        }

        chunks
    }

    /// Find a good split point near the target end
    fn find_split_point(
        &self,
        chars: &[char],
        _start: usize,
        target_end: usize,
        content_len: usize,
    ) -> usize {
        if target_end >= content_len {
            return content_len;
        }

        // Look for paragraph break first (double newline)
        let search_start = target_end.saturating_sub(100);
        let _search_end = (target_end + 100).min(content_len);

        for i in (search_start..target_end).rev() {
            if i + 1 < chars.len() && chars[i] == '\n' && chars[i + 1] == '\n' {
                return i + 2;
            }
        }

        // Look for sentence boundary (. ! ?)
        for i in (search_start..target_end).rev() {
            let c = chars[i];
            if (c == '.' || c == '!' || c == '?')
                && (i + 1 >= chars.len() || chars[i + 1].is_whitespace())
            {
                return i + 1;
            }
        }

        // Look for any newline
        for i in (search_start..target_end).rev() {
            if chars[i] == '\n' {
                return i + 1;
            }
        }

        // Fall back to word boundary
        for i in (search_start..target_end).rev() {
            if chars[i].is_whitespace() {
                return i + 1;
            }
        }

        target_end
    }

    /// Link chunks with preceding/following references
    fn link_chunks(&self, chunks: &mut [Chunk]) {
        let chunk_ids: Vec<String> = chunks.iter().map(|c| c.metadata.chunk_id.clone()).collect();

        for (i, chunk) in chunks.iter_mut().enumerate() {
            if i > 0 {
                chunk.metadata.preceding_chunk_id = Some(chunk_ids[i - 1].clone());
            }
            if i + 1 < chunk_ids.len() {
                chunk.metadata.following_chunk_id = Some(chunk_ids[i + 1].clone());
            }
        }
    }

    /// Estimate token count from text
    fn estimate_tokens(&self, text: &str) -> usize {
        // Simple word-based estimation
        text.unicode_words().count()
    }
}

/// Detected section in a document
struct Section {
    hierarchy: Vec<String>,
    content: String,
}

/// Detected heading
struct Heading {
    level: usize,
    text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_splitting() {
        let config = ChunkingConfig {
            chunk_size: 100,
            overlap_fraction: 0.15,
            min_chunk_size: 20,
            max_chunk_size: 200,
        };

        let splitter = TextSplitter::new(config);

        let doc = Document::new(
            "This is a test document. It has multiple sentences. \
             We want to test the chunking functionality. \
             This should create multiple chunks when the content is long enough. \
             Here is some more content to ensure we have enough text for testing. \
             And even more content here to really make sure we test the overlap behavior."
        );

        let chunks = splitter.split_document(&doc);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
            assert!(!chunk.metadata.chunk_id.is_empty());
        }
    }

    #[test]
    fn test_heading_detection() {
        let config = ChunkingConfig::default();
        let splitter = TextSplitter::new(config);

        let doc = Document::new(
            "# Main Title\n\n\
             Some introduction text here.\n\n\
             ## Section 1\n\n\
             Content of section 1.\n\n\
             ## Section 2\n\n\
             Content of section 2.\n\n\
             ### Subsection 2.1\n\n\
             More detailed content."
        );

        let chunks = splitter.split_document(&doc);

        // Should have detected section hierarchy
        let has_hierarchy = chunks.iter().any(|c| !c.metadata.section_hierarchy.is_empty());
        assert!(has_hierarchy, "Should detect section hierarchy");
    }

    #[test]
    fn test_chunk_linking() {
        let config = ChunkingConfig {
            chunk_size: 50,
            overlap_fraction: 0.1,
            min_chunk_size: 10,
            max_chunk_size: 100,
        };

        let splitter = TextSplitter::new(config);
        let doc = Document::new("a ".repeat(200));

        let chunks = splitter.split_document(&doc);

        if chunks.len() > 1 {
            // First chunk should have following but no preceding
            assert!(chunks[0].metadata.preceding_chunk_id.is_none());
            assert!(chunks[0].metadata.following_chunk_id.is_some());

            // Last chunk should have preceding but no following
            let last = chunks.last().expect("chunks.len() > 1 guarantees last exists");
            assert!(last.metadata.preceding_chunk_id.is_some());
            assert!(last.metadata.following_chunk_id.is_none());
        }
    }

    #[test]
    fn test_empty_document_produces_no_chunks() {
        let config = ChunkingConfig {
            chunk_size: 100,
            overlap_fraction: 0.15,
            min_chunk_size: 20,
            max_chunk_size: 200,
        };
        let splitter = TextSplitter::new(config);

        let doc = Document::new("");
        let chunks = splitter.split_document(&doc);
        assert!(
            chunks.is_empty(),
            "Empty document should produce no chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_very_short_document_single_chunk() {
        let config = ChunkingConfig {
            chunk_size: 100,
            overlap_fraction: 0.15,
            min_chunk_size: 5,
            max_chunk_size: 200,
        };
        let splitter = TextSplitter::new(config);

        let doc = Document::new("A short document.");
        let chunks = splitter.split_document(&doc);

        assert_eq!(
            chunks.len(),
            1,
            "Very short document should produce exactly one chunk"
        );
        assert_eq!(chunks[0].content, "A short document.");
    }

    #[test]
    fn test_headings_only_no_body_text() {
        let config = ChunkingConfig {
            chunk_size: 100,
            overlap_fraction: 0.15,
            min_chunk_size: 5,
            max_chunk_size: 200,
        };
        let splitter = TextSplitter::new(config);

        // Document with only headings and no body text at all
        let doc = Document::new("# Heading One\n## Heading Two\n### Heading Three");
        let chunks = splitter.split_document(&doc);

        // The splitter's detect_sections will parse headings and find no
        // content between them, resulting in no sections with content.
        // The fallback creates one section with the full text, but since
        // all lines are headings, sections might be empty.
        // Regardless, the function should not panic.
        // If any chunks are produced, they should have valid content.
        for chunk in &chunks {
            assert!(!chunk.metadata.chunk_id.is_empty());
        }
    }
}
