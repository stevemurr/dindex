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

    #[test]
    fn test_tokenize_returns_correct_token_list() {
        let text = "Hello world, this is Rust!";
        let tokens = SimpleTokenizer::tokenize(text);
        // unicode_words splits on word boundaries, stripping punctuation
        assert_eq!(tokens, vec!["Hello", "world", "this", "is", "Rust"]);
    }

    #[test]
    fn test_tokenize_with_punctuation() {
        let text = "It's a test-driven approach.";
        let tokens = SimpleTokenizer::tokenize(text);
        // "It's" is treated as one word by unicode_words, "test-driven" may split
        assert!(!tokens.is_empty());
        assert!(tokens.contains(&"It's"));
    }

    #[test]
    fn test_estimate_chars_various_counts() {
        assert_eq!(SimpleTokenizer::estimate_chars(0), 0);
        assert_eq!(SimpleTokenizer::estimate_chars(1), 4);
        assert_eq!(SimpleTokenizer::estimate_chars(10), 40);
        assert_eq!(SimpleTokenizer::estimate_chars(100), 400);
    }

    #[test]
    fn test_estimate_tokens_various_counts() {
        assert_eq!(SimpleTokenizer::estimate_tokens(0), 0);
        assert_eq!(SimpleTokenizer::estimate_tokens(4), 1);
        assert_eq!(SimpleTokenizer::estimate_tokens(40), 10);
        assert_eq!(SimpleTokenizer::estimate_tokens(400), 100);
        // Below 4 chars, integer division truncates
        assert_eq!(SimpleTokenizer::estimate_tokens(3), 0);
    }

    #[test]
    fn test_empty_string_handling() {
        assert_eq!(SimpleTokenizer::count_tokens(""), 0);
        assert!(SimpleTokenizer::tokenize("").is_empty());

        // truncate_to_tokens with empty string
        let truncated = truncate_to_tokens("", 5);
        assert_eq!(truncated, "");
    }

    #[test]
    fn test_whitespace_only_string() {
        assert_eq!(SimpleTokenizer::count_tokens("   \t\n  "), 0);
        assert!(SimpleTokenizer::tokenize("   \t\n  ").is_empty());
    }
}
