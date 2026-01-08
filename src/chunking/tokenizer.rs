//! Simple tokenization utilities

use unicode_segmentation::UnicodeSegmentation;

/// Simple word tokenizer for token counting
pub struct SimpleTokenizer;

impl SimpleTokenizer {
    /// Count tokens in text (word-based approximation)
    pub fn count_tokens(text: &str) -> usize {
        text.unicode_words().count()
    }

    /// Split text into tokens
    pub fn tokenize(text: &str) -> Vec<&str> {
        text.unicode_words().collect()
    }

    /// Estimate character count from token count
    pub fn estimate_chars(token_count: usize) -> usize {
        token_count * 4 // Rough approximation
    }

    /// Estimate token count from character count
    pub fn estimate_tokens(char_count: usize) -> usize {
        char_count / 4 // Rough approximation
    }
}

/// Token-aware text truncation
pub fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    let words: Vec<&str> = text.unicode_words().collect();
    if words.len() <= max_tokens {
        return text.to_string();
    }

    words[..max_tokens].join(" ")
}

/// Calculate overlap region between two texts
pub fn calculate_overlap(text1: &str, text2: &str) -> Option<String> {
    let words1: Vec<&str> = text1.unicode_words().collect();
    let words2: Vec<&str> = text2.unicode_words().collect();

    if words1.is_empty() || words2.is_empty() {
        return None;
    }

    // Find longest common suffix of text1 that is prefix of text2
    let max_overlap = words1.len().min(words2.len());
    for len in (1..=max_overlap).rev() {
        let suffix = &words1[words1.len() - len..];
        let prefix = &words2[..len];
        if suffix == prefix {
            return Some(suffix.join(" "));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens() {
        let text = "Hello world, this is a test!";
        let count = SimpleTokenizer::count_tokens(text);
        assert_eq!(count, 6); // "Hello", "world", "this", "is", "a", "test"
    }

    #[test]
    fn test_truncate_to_tokens() {
        let text = "one two three four five";
        let truncated = truncate_to_tokens(text, 3);
        assert_eq!(truncated, "one two three");
    }
}
