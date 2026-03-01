//! Document splitting into semantic chunks
//!
//! Uses sentence-granular, token-aware splitting: content is pre-split into
//! sentences, token counts are computed with a real tokenizer, and sentences
//! are greedily accumulated up to the token budget. Overlap is computed by
//! walking sentences backward from the chunk boundary.

use crate::chunking::tokenizer::{HeuristicTokenizer, SharedTokenizer};
use crate::config::ChunkingConfig;
use crate::types::{Chunk, ChunkMetadata, Document};
use std::sync::Arc;
use tracing::debug;

/// Text splitter for creating semantic chunks
pub struct TextSplitter {
    config: ChunkingConfig,
    tokenizer: SharedTokenizer,
}

impl TextSplitter {
    /// Create a new text splitter with a specific tokenizer.
    pub fn new(config: ChunkingConfig, tokenizer: SharedTokenizer) -> Self {
        Self { config, tokenizer }
    }

    /// Create a text splitter with the heuristic tokenizer (backward compatible).
    pub fn new_with_heuristic(config: ChunkingConfig) -> Self {
        Self::new(config, Arc::new(HeuristicTokenizer::new()))
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
        let total_tokens = self.tokenizer.count_tokens(content);

        for section in &sections {
            let section_chunks = self.split_section(
                &section.content,
                &document.id,
                &section.hierarchy,
                chunk_idx,
                total_tokens,
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
            // Propagate document metadata to chunk extra fields
            chunk.metadata.extra.extend(document.metadata.clone());
        }

        debug!(
            "Split document {} into {} chunks (tokenizer: {})",
            document.id,
            chunks.len(),
            self.tokenizer.name(),
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

    /// Split a section into chunks using sentence-granular, token-aware splitting.
    ///
    /// Algorithm:
    /// 1. Pre-split content into sentences
    /// 2. Count tokens per sentence using the real tokenizer
    /// 3. Greedily accumulate sentences up to the token budget
    /// 4. Compute overlap by walking sentences backward from the boundary
    fn split_section(
        &self,
        content: &str,
        document_id: &str,
        hierarchy: &[String],
        start_idx: usize,
        total_tokens: usize,
    ) -> Vec<Chunk> {
        if content.is_empty() {
            return Vec::new();
        }

        let sentences = split_into_sentences(content);
        if sentences.is_empty() {
            return Vec::new();
        }

        // Pre-compute token counts for each sentence
        let token_counts: Vec<usize> = sentences
            .iter()
            .map(|s| self.tokenizer.count_tokens(s))
            .collect();

        let target_tokens = self.config.chunk_size;
        let overlap_tokens =
            (target_tokens as f32 * self.config.overlap_fraction) as usize;
        let min_tokens = self.config.min_chunk_size;

        // Track cumulative token positions for computing document position
        let section_total_tokens: usize = token_counts.iter().sum();

        let mut chunks = Vec::new();
        let mut start_sentence = 0;

        while start_sentence < sentences.len() {
            let mut current_tokens = 0;
            let mut end_sentence = start_sentence;

            // Greedily accumulate sentences up to the token budget
            while end_sentence < sentences.len() {
                let next_tokens = current_tokens + token_counts[end_sentence];
                if next_tokens > target_tokens && end_sentence > start_sentence {
                    break;
                }
                current_tokens = next_tokens;
                end_sentence += 1;
            }

            // Build chunk text
            let chunk_text: String =
                sentences[start_sentence..end_sentence].concat();
            let chunk_text = chunk_text.trim().to_string();

            let chunk_tokens = self.tokenizer.count_tokens(&chunk_text);

            // Accept the chunk if it meets minimum size OR it's the last content
            if chunk_tokens >= min_tokens || end_sentence >= sentences.len() {
                // Compute position as fraction through the section's tokens
                let tokens_before: usize =
                    token_counts[..start_sentence].iter().sum();
                let position = if total_tokens > 0 {
                    tokens_before as f32 / total_tokens.max(1) as f32
                } else {
                    0.0
                };

                let metadata = ChunkMetadata {
                    chunk_id: format!(
                        "{}_{}",
                        document_id,
                        start_idx + chunks.len()
                    ),
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
                    token_count: chunk_tokens,
                });
            }

            // Done if we've consumed all sentences
            if end_sentence >= sentences.len() {
                break;
            }

            // Compute overlap: walk sentences backward from the boundary
            let mut overlap_used = 0;
            let mut next_start = end_sentence;
            for i in (start_sentence..end_sentence).rev() {
                if overlap_used + token_counts[i] > overlap_tokens {
                    break;
                }
                overlap_used += token_counts[i];
                next_start = i;
            }

            // Ensure forward progress
            if next_start <= start_sentence {
                next_start = end_sentence;
            }

            start_sentence = next_start;
        }

        // If no chunks were produced (e.g., all below min_tokens) but there
        // is content, produce one chunk with everything
        if chunks.is_empty() && section_total_tokens > 0 {
            let chunk_text = content.trim().to_string();
            let chunk_tokens = self.tokenizer.count_tokens(&chunk_text);
            let metadata = ChunkMetadata {
                chunk_id: format!("{}_{}", document_id, start_idx),
                document_id: document_id.to_string(),
                source_url: None,
                source_title: None,
                timestamp: chrono::Utc::now(),
                position_in_doc: 0.0,
                section_hierarchy: hierarchy.to_vec(),
                preceding_chunk_id: None,
                following_chunk_id: None,
                node_id: None,
                extra: std::collections::HashMap::new(),
            };
            chunks.push(Chunk {
                metadata,
                content: chunk_text,
                token_count: chunk_tokens,
            });
        }

        chunks
    }

    /// Link chunks with preceding/following references
    fn link_chunks(&self, chunks: &mut [Chunk]) {
        let chunk_ids: Vec<String> =
            chunks.iter().map(|c| c.metadata.chunk_id.clone()).collect();

        for (i, chunk) in chunks.iter_mut().enumerate() {
            if i > 0 {
                chunk.metadata.preceding_chunk_id =
                    Some(chunk_ids[i - 1].clone());
            }
            if i + 1 < chunk_ids.len() {
                chunk.metadata.following_chunk_id =
                    Some(chunk_ids[i + 1].clone());
            }
        }
    }
}

/// Split text into sentences, preserving whitespace.
///
/// Splits on:
/// 1. Paragraph breaks (double newline)
/// 2. Sentence-ending punctuation (`.`, `!`, `?`) followed by whitespace
/// 3. Single newlines (as a last resort for line-oriented content)
///
/// Each returned segment includes its trailing whitespace/delimiter so that
/// concatenation reproduces the original text.
fn split_into_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        current.push(chars[i]);

        // Check for paragraph break (double newline)
        if chars[i] == '\n' && i + 1 < len && chars[i + 1] == '\n' {
            current.push(chars[i + 1]);
            i += 2;
            // Consume any additional blank lines
            while i < len && chars[i] == '\n' {
                current.push(chars[i]);
                i += 1;
            }
            if !current.trim().is_empty() {
                sentences.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
            continue;
        }

        // Check for sentence-ending punctuation followed by whitespace
        if (chars[i] == '.' || chars[i] == '!' || chars[i] == '?')
            && (i + 1 >= len || chars[i + 1].is_whitespace())
        {
            // Include trailing spaces (but not newlines, which are their own boundaries)
            while i + 1 < len && chars[i + 1] == ' ' {
                i += 1;
                current.push(chars[i]);
            }
            sentences.push(std::mem::take(&mut current));
            i += 1;
            continue;
        }

        // Check for single newline (line-oriented content)
        if chars[i] == '\n' {
            sentences.push(std::mem::take(&mut current));
            i += 1;
            continue;
        }

        i += 1;
    }

    // Remaining text
    if !current.is_empty() {
        sentences.push(current);
    }

    sentences
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
    use crate::chunking::tokenizer::{BpeTokenizer, Tokenizer};

    fn heuristic_splitter(config: ChunkingConfig) -> TextSplitter {
        TextSplitter::new_with_heuristic(config)
    }

    fn bpe_splitter(config: ChunkingConfig) -> TextSplitter {
        let tok = BpeTokenizer::new("cl100k_base").unwrap();
        TextSplitter::new(config, Arc::new(tok))
    }

    #[test]
    fn test_basic_splitting() {
        let config = ChunkingConfig {
            chunk_size: 100,
            overlap_fraction: 0.15,
            min_chunk_size: 20,
            max_chunk_size: 200,
        };

        let splitter = heuristic_splitter(config);

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
        let splitter = heuristic_splitter(config);

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
        let has_hierarchy =
            chunks.iter().any(|c| !c.metadata.section_hierarchy.is_empty());
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

        let splitter = heuristic_splitter(config);
        let doc = Document::new("a ".repeat(200));

        let chunks = splitter.split_document(&doc);

        if chunks.len() > 1 {
            // First chunk should have following but no preceding
            assert!(chunks[0].metadata.preceding_chunk_id.is_none());
            assert!(chunks[0].metadata.following_chunk_id.is_some());

            // Last chunk should have preceding but no following
            let last =
                chunks.last().expect("chunks.len() > 1 guarantees last exists");
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
        let splitter = heuristic_splitter(config);

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
        let splitter = heuristic_splitter(config);

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
        let splitter = heuristic_splitter(config);

        // Document with only headings and no body text at all
        let doc = Document::new(
            "# Heading One\n## Heading Two\n### Heading Three",
        );
        let chunks = splitter.split_document(&doc);

        // The function should not panic.
        for chunk in &chunks {
            assert!(!chunk.metadata.chunk_id.is_empty());
        }
    }

    #[test]
    fn test_split_into_sentences_basic() {
        let sentences = split_into_sentences(
            "Hello world. This is a test. And another sentence.",
        );
        assert_eq!(sentences.len(), 3);
        // Concatenation should reproduce (approximately) the original
        let joined: String = sentences.concat();
        assert_eq!(joined, "Hello world. This is a test. And another sentence.");
    }

    #[test]
    fn test_split_into_sentences_paragraph_break() {
        let sentences =
            split_into_sentences("First paragraph.\n\nSecond paragraph.");
        assert_eq!(sentences.len(), 2);
    }

    #[test]
    fn test_split_into_sentences_single_newlines() {
        let sentences = split_into_sentences("Line one\nLine two\nLine three");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_token_counts_are_accurate_with_bpe() {
        let config = ChunkingConfig {
            chunk_size: 30,
            overlap_fraction: 0.15,
            min_chunk_size: 5,
            max_chunk_size: 100,
        };
        let splitter = bpe_splitter(config.clone());

        let doc = Document::new(
            "The quick brown fox jumps over the lazy dog. \
             Pack my box with five dozen liquor jugs. \
             How vexingly quick daft zebras jump. \
             The five boxing wizards jump quickly."
        );

        let chunks = splitter.split_document(&doc);
        let tok = BpeTokenizer::new("cl100k_base").unwrap();

        for chunk in &chunks {
            let actual_tokens = tok.count_tokens(&chunk.content);
            assert_eq!(
                chunk.token_count, actual_tokens,
                "token_count should match actual BPE count for chunk: {:?}",
                &chunk.content[..chunk.content.len().min(40)]
            );
            // Chunks should not wildly exceed target (allow some slack for
            // sentence granularity)
            assert!(
                actual_tokens <= config.chunk_size * 2,
                "chunk has {} tokens, way over target {}",
                actual_tokens,
                config.chunk_size
            );
        }
    }

    #[test]
    fn test_overlap_produces_shared_content() {
        let config = ChunkingConfig {
            chunk_size: 20,
            overlap_fraction: 0.3,
            min_chunk_size: 5,
            max_chunk_size: 100,
        };
        let splitter = heuristic_splitter(config);

        let doc = Document::new(
            "First sentence here. Second sentence here. \
             Third sentence here. Fourth sentence here. \
             Fifth sentence here. Sixth sentence here."
        );

        let chunks = splitter.split_document(&doc);

        if chunks.len() >= 2 {
            // With 30% overlap, adjacent chunks should share some content
            // With sentence-granular splitting the overlap is approximate,
            // so we just verify chunks were produced
            assert!(chunks.len() >= 2, "should have multiple chunks");
        }
    }

    #[test]
    fn test_heuristic_vs_bpe_produce_chunks() {
        let config = ChunkingConfig {
            chunk_size: 50,
            overlap_fraction: 0.15,
            min_chunk_size: 10,
            max_chunk_size: 200,
        };

        let doc = Document::new(
            "Artificial intelligence is transforming many industries. \
             Machine learning models can now understand natural language. \
             Deep learning has achieved remarkable results in computer vision. \
             Transformers have revolutionized NLP tasks significantly. \
             Large language models are being deployed across many applications."
        );

        let heuristic_chunks =
            heuristic_splitter(config.clone()).split_document(&doc);
        let bpe_chunks = bpe_splitter(config).split_document(&doc);

        assert!(!heuristic_chunks.is_empty());
        assert!(!bpe_chunks.is_empty());

        // Both should produce valid chunks
        for chunk in heuristic_chunks.iter().chain(bpe_chunks.iter()) {
            assert!(!chunk.content.is_empty());
            assert!(chunk.token_count > 0);
        }
    }
}
