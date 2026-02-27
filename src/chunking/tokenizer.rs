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
