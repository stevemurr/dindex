//! Snippet extraction for search result citations
//!
//! Extracts the most query-relevant sentence(s) from chunk content
//! to provide focused citation snippets instead of full 512-token chunks.

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
    let mut scored: Vec<(usize, f32)> = sentences
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

    // Sort by score descending, then by position (prefer earlier sentences for ties)
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    // Take the best sentence(s) up to max_chars
    let mut result = String::new();
    for (idx, _score) in &scored {
        let sentence = &sentences[*idx];
        if result.is_empty() {
            result = sentence.to_string();
        } else if result.len() + sentence.len() + 1 <= max_chars {
            result.push(' ');
            result.push_str(sentence);
        } else {
            break;
        }
    }

    // Fallback: if no overlap found (all scores are 0), use first sentence
    if scored.first().map(|(_, s)| *s) == Some(0.0) {
        result = sentences[0].to_string();
    }

    Some(truncate_to_chars(&result, max_chars))
}

/// Split text into sentences using punctuation-based heuristics.
fn split_sentences(text: &str) -> Vec<String> {
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

/// Truncate a string to max_chars, adding "..." if truncated.
fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    // Find a word boundary near max_chars - 3 (for "...")
    let limit = max_chars.saturating_sub(3);
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
    fn test_extract_snippet_fallback_to_first_sentence() {
        let content = "The quick brown fox jumps over the lazy dog. Another sentence here.";
        let snippet = extract_snippet("quantum computing", content, 200).unwrap();
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
}
