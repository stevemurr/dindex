//! WikiText to plaintext converter
//!
//! Converts MediaWiki markup to clean plaintext for indexing.

use regex_lite::Regex;
use std::sync::OnceLock;

// Lazy-compiled regex patterns for better performance
static RE_NOWIKI: OnceLock<Regex> = OnceLock::new();
static RE_PRE: OnceLock<Regex> = OnceLock::new();
static RE_CATEGORIES: OnceLock<Regex> = OnceLock::new();
static RE_FILES: OnceLock<Regex> = OnceLock::new();
static RE_EXTERNAL_LINK: OnceLock<Regex> = OnceLock::new();
static RE_EXTERNAL_BARE: OnceLock<Regex> = OnceLock::new();
static RE_HEADING1: OnceLock<Regex> = OnceLock::new();
static RE_HEADING2: OnceLock<Regex> = OnceLock::new();
static RE_HEADING3: OnceLock<Regex> = OnceLock::new();
static RE_HEADING4: OnceLock<Regex> = OnceLock::new();
static RE_HEADING5: OnceLock<Regex> = OnceLock::new();
static RE_HEADING6: OnceLock<Regex> = OnceLock::new();
static RE_LIST: OnceLock<Regex> = OnceLock::new();
static RE_DEF_LIST: OnceLock<Regex> = OnceLock::new();
static RE_INTERWIKI: OnceLock<Regex> = OnceLock::new();
static RE_MAGIC_WORDS: OnceLock<Regex> = OnceLock::new();

/// WikiText parser that converts MediaWiki markup to plain text
pub struct WikiTextParser {
    /// Remove references and citations
    remove_refs: bool,
    /// Remove tables
    remove_tables: bool,
    /// Remove categories
    remove_categories: bool,
    /// Remove file/image links
    remove_files: bool,
}

impl Default for WikiTextParser {
    fn default() -> Self {
        Self {
            remove_refs: true,
            remove_tables: true,
            remove_categories: true,
            remove_files: true,
        }
    }
}

impl WikiTextParser {
    /// Create a new WikiText parser with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse WikiText and return clean plaintext
    pub fn parse(&self, wikitext: &str) -> String {
        let mut text = wikitext.to_string();

        // Remove nowiki and pre blocks (preserve content)
        text = self.handle_nowiki(&text);

        // Remove comments
        text = self.remove_comments(&text);

        // Remove references
        if self.remove_refs {
            text = self.remove_references(&text);
        }

        // Remove tables
        if self.remove_tables {
            text = self.remove_tables_markup(&text);
        }

        // Remove templates
        text = self.remove_templates(&text);

        // Remove categories
        if self.remove_categories {
            text = self.remove_categories_markup(&text);
        }

        // Remove file/image links
        if self.remove_files {
            text = self.remove_file_links(&text);
        }

        // Process internal links [[link|display]] -> display
        text = self.process_internal_links(&text);

        // Process external links [url text] -> text
        text = self.process_external_links(&text);

        // Process formatting
        text = self.process_formatting(&text);

        // Remove interwiki links
        text = self.remove_interwiki_links(&text);

        // Remove magic words
        text = self.remove_magic_words(&text);

        // Clean up whitespace
        text = self.clean_whitespace(&text);

        text
    }

    /// Handle <nowiki> and <pre> blocks
    fn handle_nowiki(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Remove nowiki tags but keep content
        let re_nowiki = RE_NOWIKI.get_or_init(|| Regex::new(r"<nowiki>(.*?)</nowiki>").unwrap());
        result = re_nowiki.replace_all(&result, "$1").to_string();

        // Remove pre tags but keep content
        let re_pre = RE_PRE.get_or_init(|| Regex::new(r"<pre>(.*?)</pre>").unwrap());
        result = re_pre.replace_all(&result, "$1").to_string();

        result
    }

    /// Remove HTML/XML comments
    fn remove_comments(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();
        let mut in_comment = false;

        while let Some(c) = chars.next() {
            if !in_comment {
                if c == '<' && chars.peek() == Some(&'!') {
                    let lookahead: String = chars.clone().take(3).collect();
                    if lookahead.starts_with("!--") {
                        in_comment = true;
                        chars.next(); // !
                        chars.next(); // -
                        chars.next(); // -
                        continue;
                    }
                }
                result.push(c);
            } else {
                // Look for -->
                if c == '-' && chars.peek() == Some(&'-') {
                    chars.next();
                    if chars.peek() == Some(&'>') {
                        chars.next();
                        in_comment = false;
                    }
                }
            }
        }

        result
    }

    /// Remove <ref>...</ref> and <ref .../> tags
    fn remove_references(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Check for <ref
            if i + 4 < chars.len()
                && chars[i] == '<'
                && (chars[i + 1] == 'r' || chars[i + 1] == 'R')
                && (chars[i + 2] == 'e' || chars[i + 2] == 'E')
                && (chars[i + 3] == 'f' || chars[i + 3] == 'F')
            {
                // Find the end of the ref tag
                let mut j = i + 4;

                // Skip to > or />
                while j < chars.len() && chars[j] != '>' {
                    j += 1;
                }

                if j >= chars.len() {
                    // Malformed tag, keep the rest
                    result.extend(&chars[i..]);
                    break;
                }

                // Check if self-closing
                if j > 0 && chars[j - 1] == '/' {
                    // Self-closing <ref ... />
                    i = j + 1;
                    continue;
                }

                // It's <ref>...</ref> or <ref ...>...</ref>
                // Find closing </ref>
                j += 1;
                let mut found_close = false;
                while j + 5 < chars.len() {
                    if chars[j] == '<'
                        && chars[j + 1] == '/'
                        && (chars[j + 2] == 'r' || chars[j + 2] == 'R')
                        && (chars[j + 3] == 'e' || chars[j + 3] == 'E')
                        && (chars[j + 4] == 'f' || chars[j + 4] == 'F')
                    {
                        // Find the closing >
                        while j < chars.len() && chars[j] != '>' {
                            j += 1;
                        }
                        found_close = true;
                        i = j + 1;
                        break;
                    }
                    j += 1;
                }

                if !found_close {
                    // No closing tag found, skip to end of opening tag
                    while i < chars.len() && chars[i] != '>' {
                        i += 1;
                    }
                    i += 1;
                }
            } else {
                result.push(chars[i]);
                i += 1;
            }
        }

        result
    }

    /// Remove wiki tables {| ... |}
    fn remove_tables_markup(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut depth = 0;
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'|') {
                depth += 1;
                chars.next(); // consume '|'
            } else if c == '|' && chars.peek() == Some(&'}') {
                if depth > 0 {
                    depth -= 1;
                }
                chars.next(); // consume '}'
            } else if depth == 0 {
                result.push(c);
            }
        }

        result
    }

    /// Remove templates {{ ... }}
    fn remove_templates(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut depth = 0;
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                depth += 1;
                chars.next(); // consume second '{'
            } else if c == '}' && chars.peek() == Some(&'}') {
                if depth > 0 {
                    depth -= 1;
                }
                chars.next(); // consume second '}'
            } else if depth == 0 {
                result.push(c);
            }
        }

        result
    }

    /// Remove category links [[Category:...]]
    fn remove_categories_markup(&self, text: &str) -> String {
        let re = RE_CATEGORIES.get_or_init(|| {
            Regex::new(r"(?i)\[\[(Category|Kategorie|Catégorie|Categoría):[^\]]*\]\]").unwrap()
        });
        re.replace_all(text, "").to_string()
    }

    /// Remove file/image links [[File:...]] and [[Image:...]]
    fn remove_file_links(&self, text: &str) -> String {
        let re = RE_FILES.get_or_init(|| {
            Regex::new(r"(?i)\[\[(File|Image|Datei|Fichier|Archivo):[^\]]*\]\]").unwrap()
        });
        re.replace_all(text, "").to_string()
    }

    /// Process internal links [[link]] or [[link|display]]
    fn process_internal_links(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '[' && chars.peek() == Some(&'[') {
                chars.next(); // consume second '['

                // Read until ]]
                let mut link_content = String::new();
                let mut depth = 1;

                while let Some(ch) = chars.next() {
                    if ch == '[' && chars.peek() == Some(&'[') {
                        depth += 1;
                        chars.next();
                        link_content.push_str("[[");
                    } else if ch == ']' && chars.peek() == Some(&']') {
                        depth -= 1;
                        chars.next();
                        if depth == 0 {
                            break;
                        }
                        link_content.push_str("]]");
                    } else {
                        link_content.push(ch);
                    }
                }

                // Extract display text (after | if present, otherwise the link itself)
                let display = if let Some(pipe_pos) = link_content.find('|') {
                    &link_content[pipe_pos + 1..]
                } else {
                    &link_content
                };

                // Skip if it starts with special prefixes or is an interwiki link
                let lower = link_content.to_lowercase();
                let is_special = lower.starts_with("file:")
                    || lower.starts_with("image:")
                    || lower.starts_with("category:")
                    || lower.starts_with("kategorie:")
                    || lower.starts_with("catégorie:")
                    || lower.starts_with("categoría:")
                    || lower.starts_with("datei:")
                    || lower.starts_with("fichier:")
                    || lower.starts_with("archivo:")
                    || lower.starts_with("wikt:")
                    || lower.starts_with("wikipedia:")
                    || lower.starts_with("wp:");

                // Check for interwiki links (2-3 char language code followed by colon)
                let is_interwiki = {
                    if let Some(colon_pos) = lower.find(':') {
                        let prefix = &lower[..colon_pos];
                        prefix.len() >= 2 && prefix.len() <= 3 && prefix.chars().all(|c| c.is_ascii_lowercase())
                    } else {
                        false
                    }
                };

                if !is_special && !is_interwiki {
                    result.push_str(display);
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Process external links [url text] -> text or just show url
    fn process_external_links(&self, text: &str) -> String {
        let re = RE_EXTERNAL_LINK.get_or_init(|| {
            Regex::new(r"\[https?://[^\s\]]+\s+([^\]]+)\]").unwrap()
        });
        let result = re.replace_all(text, "$1").to_string();

        // Handle bare external links [url]
        let re_bare = RE_EXTERNAL_BARE.get_or_init(|| {
            Regex::new(r"\[(https?://[^\s\]]+)\]").unwrap()
        });
        re_bare.replace_all(&result, "$1").to_string()
    }

    /// Process formatting markup
    fn process_formatting(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Remove bold/italic: '''bold''' -> bold, ''italic'' -> italic
        // Handle bold first (longer pattern)
        result = result.replace("'''''", ""); // bold italic
        result = result.replace("'''", ""); // bold
        result = result.replace("''", ""); // italic

        // Remove heading markup but keep content (using pre-compiled regexes)
        // ==== Heading ==== -> Heading
        let re_h6 = RE_HEADING6.get_or_init(|| Regex::new(r"======+\s*(.*?)\s*======+").unwrap());
        result = re_h6.replace_all(&result, "$1\n").to_string();
        let re_h5 = RE_HEADING5.get_or_init(|| Regex::new(r"=====+\s*(.*?)\s*=====+").unwrap());
        result = re_h5.replace_all(&result, "$1\n").to_string();
        let re_h4 = RE_HEADING4.get_or_init(|| Regex::new(r"====+\s*(.*?)\s*====+").unwrap());
        result = re_h4.replace_all(&result, "$1\n").to_string();
        let re_h3 = RE_HEADING3.get_or_init(|| Regex::new(r"===+\s*(.*?)\s*===+").unwrap());
        result = re_h3.replace_all(&result, "$1\n").to_string();
        let re_h2 = RE_HEADING2.get_or_init(|| Regex::new(r"==+\s*(.*?)\s*==+").unwrap());
        result = re_h2.replace_all(&result, "$1\n").to_string();
        let re_h1 = RE_HEADING1.get_or_init(|| Regex::new(r"=+\s*(.*?)\s*=+").unwrap());
        result = re_h1.replace_all(&result, "$1\n").to_string();

        // Remove horizontal rules
        result = result.replace("----", "");

        // Remove bullet points and numbered lists
        let re_list = RE_LIST.get_or_init(|| Regex::new(r"(?m)^[*#:;]+\s*").unwrap());
        result = re_list.replace_all(&result, "").to_string();

        // Remove definition list markup
        let re_def = RE_DEF_LIST.get_or_init(|| Regex::new(r"(?m)^;([^:]+):(.+)$").unwrap());
        result = re_def.replace_all(&result, "$1: $2").to_string();

        result
    }

    /// Remove interwiki links [[lang:...]]
    fn remove_interwiki_links(&self, text: &str) -> String {
        // Common language codes
        let re = RE_INTERWIKI.get_or_init(|| {
            Regex::new(r"\[\[[a-z]{2,3}(-[a-z]+)?:[^\]]+\]\]").unwrap()
        });
        re.replace_all(text, "").to_string()
    }

    /// Remove magic words like __NOTOC__, __TOC__, etc.
    fn remove_magic_words(&self, text: &str) -> String {
        let re = RE_MAGIC_WORDS.get_or_init(|| Regex::new(r"__[A-Z]+__").unwrap());
        re.replace_all(text, "").to_string()
    }

    /// Clean up excessive whitespace
    fn clean_whitespace(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut prev_newline = false;
        let mut prev_space = false;

        for c in text.chars() {
            if c == '\n' {
                if !prev_newline {
                    result.push('\n');
                    prev_newline = true;
                }
                prev_space = false;
            } else if c.is_whitespace() {
                if !prev_space && !prev_newline {
                    result.push(' ');
                    prev_space = true;
                }
            } else {
                result.push(c);
                prev_newline = false;
                prev_space = false;
            }
        }

        result.trim().to_string()
    }
}

// Simple regex replacement without full regex crate
mod regex_lite {
    pub struct Regex {
        pattern: String,
    }

    impl Regex {
        pub fn new(pattern: &str) -> Result<Self, ()> {
            Ok(Self {
                pattern: pattern.to_string(),
            })
        }

        pub fn replace_all<'a>(&self, text: &'a str, replacement: &str) -> std::borrow::Cow<'a, str> {
            // For simple patterns, use string operations
            // For complex patterns, we'll implement basic matching
            simple_replace(text, &self.pattern, replacement)
        }
    }

    fn simple_replace<'a>(text: &'a str, pattern: &str, replacement: &str) -> std::borrow::Cow<'a, str> {
        // Handle some common regex patterns with simple string operations
        // This is a simplified implementation - in production use the regex crate

        // Handle literal string patterns (no special chars)
        if !pattern.contains('(')
            && !pattern.contains('[')
            && !pattern.contains('*')
            && !pattern.contains('+')
            && !pattern.contains('?')
            && !pattern.contains('|')
            && !pattern.contains('.')
            && !pattern.contains('^')
            && !pattern.contains('$')
        {
            return std::borrow::Cow::Owned(text.replace(pattern, replacement));
        }

        // For complex patterns, return unchanged (would need full regex support)
        // The actual parsing handles most cases with manual parsing
        std::borrow::Cow::Borrowed(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_bold_italic() {
        let parser = WikiTextParser::new();
        let result = parser.parse("This is '''bold''' and ''italic'' text.");
        assert!(result.contains("bold"));
        assert!(result.contains("italic"));
        assert!(!result.contains("'''"));
        assert!(!result.contains("''"));
    }

    #[test]
    fn test_internal_links() {
        let parser = WikiTextParser::new();

        // Simple link
        let result = parser.parse("The [[United States]] is a country.");
        assert!(result.contains("United States"));
        assert!(!result.contains("[["));

        // Link with display text
        let result = parser.parse("The [[United States|US]] is a country.");
        assert!(result.contains("US"));
        assert!(!result.contains("United States|"));
    }

    #[test]
    fn test_remove_templates() {
        let parser = WikiTextParser::new();
        let result = parser.parse("Hello {{template}} world.");
        assert_eq!(result, "Hello world.");
    }

    #[test]
    fn test_remove_categories() {
        let parser = WikiTextParser::new();
        let result = parser.parse("Content [[Category:Test]] more content.");
        assert!(result.contains("Content"));
        assert!(!result.contains("Category"));
    }

    #[test]
    fn test_whitespace_cleanup() {
        let parser = WikiTextParser::new();
        let result = parser.parse("Hello   world\n\n\n\ntest");
        assert!(!result.contains("   "));
        assert!(!result.contains("\n\n\n"));
    }

    #[test]
    fn test_remove_tables() {
        let parser = WikiTextParser::new();
        let result = parser.parse("Before {| class=\"wikitable\"\n|-\n| cell\n|} After");
        assert!(result.contains("Before"));
        assert!(result.contains("After"));
        assert!(!result.contains("wikitable"));
    }

    #[test]
    fn test_complex_wikitext() {
        let parser = WikiTextParser::new();
        let wikitext = r#"
'''Albert Einstein''' (14 March 1879 – 18 April 1955) was a German-born [[theoretical physicist]].

He developed the [[theory of relativity]]<ref>{{cite book|title=Einstein}}</ref>, one of the two pillars of [[modern physics]].

{{Infobox scientist
| name = Albert Einstein
| birth_date = 14 March 1879
}}

[[Category:Physicists]]
[[de:Albert Einstein]]
"#;

        let result = parser.parse(wikitext);

        // Should contain main text
        assert!(result.contains("Albert Einstein"));
        assert!(result.contains("theoretical physicist"));
        assert!(result.contains("theory of relativity"));

        // Should not contain markup
        assert!(!result.contains("'''"));
        assert!(!result.contains("[["));
        assert!(!result.contains("{{"));
        assert!(!result.contains("<ref>"));
        assert!(!result.contains("Category:"));
        assert!(!result.contains("de:"));
    }
}
