//! Heuristic tokenizer: improved word-level estimation
//!
//! Better than the old `chars / 4` ratio because it uses character-class
//! analysis and handles CJK characters (which are typically 1-2 tokens each
//! rather than ~4 characters per token).

use super::Tokenizer;
use unicode_segmentation::UnicodeSegmentation;

/// Heuristic tokenizer that estimates token counts from word boundaries.
///
/// For Latin scripts this is roughly 1 word ≈ 1.3 BPE tokens.
/// For CJK scripts, each character is treated as ~1 token.
pub struct HeuristicTokenizer;

impl HeuristicTokenizer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for HeuristicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for HeuristicTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        let mut token_estimate: f32 = 0.0;

        for word in text.unicode_words() {
            let cjk_chars = word.chars().filter(|c| is_cjk(*c)).count();
            let non_cjk_len = word.chars().count() - cjk_chars;

            // CJK characters: ~1 token each
            token_estimate += cjk_chars as f32;

            // Non-CJK words: approximate 1.3 tokens per word (BPE typically
            // splits long words into subwords)
            if non_cjk_len > 0 {
                token_estimate += 1.0 + (non_cjk_len as f32 / 10.0).min(1.0);
            }
        }

        token_estimate.ceil() as usize
    }

    fn name(&self) -> &str {
        "heuristic"
    }
}

/// Check if a character is in a CJK Unicode block
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |    // CJK Unified Ideographs
        '\u{3400}'..='\u{4DBF}' |    // CJK Unified Ideographs Extension A
        '\u{F900}'..='\u{FAFF}' |    // CJK Compatibility Ideographs
        '\u{3040}'..='\u{309F}' |    // Hiragana
        '\u{30A0}'..='\u{30FF}' |    // Katakana
        '\u{AC00}'..='\u{D7AF}'      // Hangul Syllables
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        let tok = HeuristicTokenizer::new();
        assert_eq!(tok.count_tokens(""), 0);
    }

    #[test]
    fn test_english_words() {
        let tok = HeuristicTokenizer::new();
        let count = tok.count_tokens("Hello world this is a test");
        // 6 words, each ~1.1-1.4 tokens -> should be in range [6, 12]
        assert!(count >= 6 && count <= 12, "got {}", count);
    }

    #[test]
    fn test_cjk_characters() {
        let tok = HeuristicTokenizer::new();
        // 5 CJK characters — should be ~5 tokens
        let count = tok.count_tokens("你好世界测试");
        assert!(count >= 5 && count <= 7, "CJK count should be ~6, got {}", count);
    }

    #[test]
    fn test_mixed_content() {
        let tok = HeuristicTokenizer::new();
        let count = tok.count_tokens("Hello 世界");
        assert!(count >= 2, "mixed content should have at least 2 tokens, got {}", count);
    }

    #[test]
    fn test_name() {
        let tok = HeuristicTokenizer::new();
        assert_eq!(tok.name(), "heuristic");
    }

    #[test]
    fn test_max_token_length_none() {
        let tok = HeuristicTokenizer::new();
        assert!(tok.max_token_length().is_none());
    }

    #[test]
    fn test_punctuation_heavy_text() {
        let tok = HeuristicTokenizer::new();
        let count = tok.count_tokens("Hello!!! What??? No... Yes; maybe: okay.");
        // There are recognizable words here, so token count should be reasonable
        assert!(
            count >= 3,
            "punctuation-heavy text should still have at least 3 tokens, got {}",
            count
        );
    }

    #[test]
    fn test_single_word() {
        let tok = HeuristicTokenizer::new();
        let count = tok.count_tokens("hello");
        assert!(count >= 1, "single word should return at least 1 token, got {}", count);
    }

    #[test]
    fn test_korean_text() {
        let tok = HeuristicTokenizer::new();
        // Korean Hangul text: "안녕하세요 세계" (Hello world)
        let count = tok.count_tokens("안녕하세요 세계");
        // "안녕하세요" = 5 Hangul syllables, "세계" = 2 Hangul syllables
        // Each Hangul syllable is in the CJK block and counts as ~1 token
        assert!(
            count >= 5,
            "Korean text with 7 Hangul chars should have at least 5 tokens, got {}",
            count
        );
    }

    #[test]
    fn test_japanese_mixed_text() {
        let tok = HeuristicTokenizer::new();
        // Mixed Japanese: Hiragana + Katakana + Kanji
        // "こんにちは" (Hiragana, 5 chars) + "カタカナ" (Katakana, 4 chars) + "漢字" (Kanji, 2 chars)
        let count = tok.count_tokens("こんにちは カタカナ 漢字");
        // 5 Hiragana + 4 Katakana + 2 Kanji = 11 CJK chars total
        assert!(
            count >= 8,
            "Japanese mixed text with 11 CJK chars should have at least 8 tokens, got {}",
            count
        );
    }
}
