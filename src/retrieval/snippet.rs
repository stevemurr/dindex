//! Snippet extraction for search result citations
//!
//! Extracts the most query-relevant sentence(s) from chunk content
//! to provide focused citation snippets instead of full 512-token chunks.

use std::cmp::Ordering;
use std::collections::HashSet;

/// Extract the best-matching snippet from content for a given query.
///
/// Splits content into sentences, scores each by query term overlap,
/// and returns the highest-scoring sentence(s) up to `max_chars`.
/// Falls back to the first sentence if no term overlap is found.
pub fn extract_snippet(query: &str, content: &str, max_chars: usize) -> Option<String> {
    if content.is_empty() || query.is_empty() {
        return None;
    }

    let sentences = split_sentences(content);
    if sentences.is_empty() {
        return None;
    }

    let query_lower = query.to_lowercase();
    let query_terms: HashSet<&str> = query_lower.split_whitespace().collect();

    if query_terms.is_empty() {
        return Some(truncate_to_chars(&sentences[0], max_chars));
    }

    // Score each sentence by query term overlap
    let scored: Vec<(usize, f32)> = sentences
        .iter()
        .enumerate()
        .map(|(i, sentence)| {
            let sentence_lower = sentence.to_lowercase();
            let overlap = query_terms
                .iter()
                .filter(|term| sentence_lower.contains(*term))
                .count();
            let score = overlap as f32 / query_terms.len() as f32;
            (i, score)
        })
        .collect();

    let has_keyword_match = scored.iter().any(|(_, s)| *s > 0.0);

    let result = if has_keyword_match {
        // Sort by score descending, then by position (prefer earlier for ties)
        let mut sorted = scored;
        sorted.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        // Take the best sentence(s) up to max_chars
        let mut buf = String::new();
        for (idx, _score) in &sorted {
            let sentence = &sentences[*idx];
            if buf.is_empty() {
                buf = sentence.to_string();
            } else if buf.len() + sentence.len() + 1 <= max_chars {
                buf.push(' ');
                buf.push_str(sentence);
            } else {
                break;
            }
        }
        buf
    } else {
        // No keyword overlap — pick the highest-quality prose sentence
        let best = sentences
            .iter()
            .enumerate()
            .map(|(i, s)| (i, sentence_quality_score(s, i, sentences.len())))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        sentences[best].to_string()
    };

    Some(truncate_to_chars(&result, max_chars))
}

/// Split text into sentences using punctuation-based heuristics.
///
/// First pre-splits on non-prose delimiters (pipes, tabs, multi-newlines)
/// to prevent navigation text from being treated as one long sentence,
/// then applies standard sentence-boundary splitting on `.!?`.
fn split_sentences(text: &str) -> Vec<String> {
    // Phase 1: Pre-split on non-prose delimiters
    let fragments = pre_split_on_delimiters(text);

    // Phase 2: Apply sentence splitting to each fragment
    let mut sentences = Vec::new();
    for fragment in &fragments {
        let sub = split_on_punctuation(fragment);
        sentences.extend(sub);
    }

    sentences
}

/// Pre-split text on pipes, tabs, and multi-newlines.
fn pre_split_on_delimiters(text: &str) -> Vec<String> {
    let mut fragments = Vec::new();

    // Split on pipes, tabs, and double+ newlines
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        let ch = chars[i];

        if ch == '|' || ch == '\t' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                fragments.push(trimmed);
            }
            current.clear();
            i += 1;
            continue;
        }

        // Check for multi-newline (two or more newlines in a row)
        if ch == '\n' && i + 1 < len && chars[i + 1] == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                fragments.push(trimmed);
            }
            current.clear();
            // Skip all consecutive newlines
            while i < len && chars[i] == '\n' {
                i += 1;
            }
            continue;
        }

        current.push(ch);
        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        fragments.push(trimmed);
    }

    fragments
}

/// Split a text fragment into sentences based on `.!?` punctuation.
fn split_on_punctuation(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    for i in 0..len {
        current.push(chars[i]);

        let is_terminal = chars[i] == '.' || chars[i] == '!' || chars[i] == '?';
        let followed_by_space = i + 1 < len && chars[i + 1].is_whitespace();
        let followed_by_upper = i + 2 < len && chars[i + 2].is_uppercase();
        let at_end = i + 1 == len;

        if is_terminal && (at_end || (followed_by_space && (followed_by_upper || i + 2 >= len))) {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Add remaining text as a sentence
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Score a sentence by how "prose-like" it is.
///
/// Used as a fallback when no keyword overlap is found, to avoid selecting
/// navigation junk (e.g. "Home | About | Contact") as the snippet.
///
/// Scoring factors:
/// - **Length**: penalize very short (<30 chars) and very long (>300 chars) fragments
/// - **Word structure**: prefer average word length 3–8 chars (typical prose)
/// - **Character composition**: prefer high ratio of alphabetic + space chars
/// - **Position**: mild preference for middle-of-document sentences (body content)
fn sentence_quality_score(sentence: &str, index: usize, total: usize) -> f32 {
    let char_count = sentence.chars().count();
    let len = char_count as f32;

    // Length score: prefer 30-300 chars
    let length_score = if len < 30.0 {
        len / 30.0
    } else if len > 300.0 {
        300.0 / len
    } else {
        1.0
    };

    // Word structure: prefer average word length 3-8
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let word_score = if words.is_empty() {
        0.0
    } else {
        let avg_word_len =
            words.iter().map(|w| w.chars().count() as f32).sum::<f32>() / words.len() as f32;
        if avg_word_len < 3.0 {
            avg_word_len / 3.0
        } else if avg_word_len > 8.0 {
            8.0 / avg_word_len
        } else {
            1.0
        }
    };

    // Character composition: ratio of alphabetic + space chars
    let alpha_space_count = sentence
        .chars()
        .filter(|c| c.is_alphabetic() || *c == ' ')
        .count() as f32;
    let alpha_score = if len > 0.0 {
        alpha_space_count / len
    } else {
        0.0
    };

    // Position score: mild preference for middle of document (body content)
    let position_score = if total <= 1 {
        1.0
    } else {
        let relative_pos = index as f32 / (total - 1) as f32;
        // Peak at ~0.3-0.5 (early-middle), tapering at edges
        1.0 - (relative_pos - 0.4).abs() * 0.5
    };

    // Weighted combination
    length_score * 0.25 + word_score * 0.25 + alpha_score * 0.35 + position_score * 0.15
}

/// Truncate a string to max_chars, adding "..." if truncated.
fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    // When max_chars is too small for "...", just hard-truncate
    if max_chars < 4 {
        return s[..s.floor_char_boundary(max_chars)].to_string();
    }
    // Find a word boundary near max_chars - 3 (for "...")
    let limit = max_chars - 3;
    let truncated = &s[..s.floor_char_boundary(limit)];
    // Try to break at last space
    if let Some(last_space) = truncated.rfind(' ') {
        format!("{}...", &truncated[..last_space])
    } else {
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_snippet_basic() {
        let content = "The cat sat on the mat. Machine learning is a branch of AI. Deep learning uses neural networks.";
        let snippet = extract_snippet("machine learning", content, 200).unwrap();
        assert!(snippet.contains("Machine learning"));
    }

    #[test]
    fn test_extract_snippet_fallback_to_best_quality_sentence() {
        let content = "The quick brown fox jumps over the lazy dog. Another sentence here.";
        let snippet = extract_snippet("quantum computing", content, 200).unwrap();
        // With quality scoring, the longer prose sentence should still be picked
        assert!(snippet.contains("quick brown fox"));
    }

    #[test]
    fn test_extract_snippet_empty_content() {
        assert!(extract_snippet("query", "", 200).is_none());
    }

    #[test]
    fn test_extract_snippet_empty_query() {
        assert!(extract_snippet("", "some content", 200).is_none());
    }

    #[test]
    fn test_extract_snippet_truncation() {
        let content = "This is a very long sentence that should be truncated when the max_chars limit is small.";
        let snippet = extract_snippet("long sentence", content, 50).unwrap();
        assert!(snippet.len() <= 50);
        assert!(snippet.ends_with("..."));
    }

    #[test]
    fn test_split_sentences() {
        let text = "First sentence. Second sentence! Third sentence?";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence.");
        assert_eq!(sentences[1], "Second sentence!");
        assert_eq!(sentences[2], "Third sentence?");
    }

    #[test]
    fn test_split_sentences_no_terminal() {
        let text = "No terminal punctuation here";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "No terminal punctuation here");
    }

    #[test]
    fn test_extract_snippet_prefers_best_match() {
        let content = "Cats are fluffy animals. Machine learning transforms industries. Dogs play in the park.";
        let snippet = extract_snippet("machine learning industries", content, 200).unwrap();
        assert!(snippet.contains("Machine learning"));
    }

    #[test]
    fn test_navigation_text_not_selected_as_snippet() {
        let content = "Hacker News | new | past | comments | ask | show | jobs | submit | login. \
            Machine learning has revolutionized the way computers process natural language and understand context.";
        let snippet = extract_snippet("quantum physics", content, 200).unwrap();
        // Should pick the prose sentence, not the navigation fragment
        assert!(
            snippet.contains("Machine learning"),
            "Expected prose content, got: {}",
            snippet
        );
    }

    #[test]
    fn test_pipe_delimited_text_split_into_fragments() {
        let text = "Home | About | Contact | Blog";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0], "Home");
        assert_eq!(sentences[1], "About");
        assert_eq!(sentences[2], "Contact");
        assert_eq!(sentences[3], "Blog");
    }

    #[test]
    fn test_quality_scoring_prefers_prose_over_short_fragment() {
        // After pipe-splitting, nav fragments become short strings like "Home"
        let nav_fragment = "Home";
        let prose = "Rust is a systems programming language that runs blazingly fast and prevents segfaults.";
        let nav_score = sentence_quality_score(nav_fragment, 0, 2);
        let prose_score = sentence_quality_score(prose, 1, 2);
        assert!(
            prose_score > nav_score,
            "Prose score ({}) should be higher than nav fragment score ({})",
            prose_score,
            nav_score
        );
    }

    #[test]
    fn test_keyword_matching_still_takes_priority() {
        // Even when the matching sentence is short/nav-like, keyword overlap should win
        let content = "Navigation menu items here. Quantum computing enables faster calculations. \
            This is a lovely piece of prose about everyday topics and regular things.";
        let snippet = extract_snippet("quantum computing", content, 200).unwrap();
        assert!(
            snippet.contains("Quantum computing"),
            "Keyword match should take priority, got: {}",
            snippet
        );
    }

    #[test]
    fn test_truncation_works_with_quality_fallback() {
        let content = "x | y | z | This is a moderately long sentence about interesting topics \
            that should be selected by quality scoring when there is no keyword overlap at all.";
        let snippet = extract_snippet("unrelated query terms", content, 50).unwrap();
        assert!(snippet.len() <= 50);
        assert!(snippet.ends_with("..."));
    }

    #[test]
    fn test_tab_delimited_text_split() {
        let text = "Column1\tColumn2\tColumn3";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Column1");
    }

    #[test]
    fn test_multi_newline_split() {
        let text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph.";
        let sentences = split_sentences(text);
        assert!(sentences.len() >= 3, "Expected at least 3 fragments, got: {:?}", sentences);
    }
}
