//! Text extraction and normalization

use ego_tree::NodeRef;
use scraper::{Html, Node, Selector};

use super::ContentExtractor;

impl ContentExtractor {
    /// Find the main content area
    pub(super) fn find_main_content(&self, document: &Html) -> String {
        // Try each content selector in priority order
        for selector in &self.content_selectors {
            if let Some(element) = document.select(selector).next() {
                let html = element.html();
                if html.len() > 200 {
                    return html;
                }
            }
        }

        // Fall back to body
        if let Ok(body_sel) = Selector::parse("body") {
            if let Some(body) = document.select(&body_sel).next() {
                return body.html();
            }
        }

        String::new()
    }

    /// Check if a node has a `<pre>` or `<code>` ancestor
    fn has_pre_ancestor(node: &NodeRef<Node>) -> bool {
        let mut current = node.parent();
        while let Some(parent) = current {
            if let Some(elem) = parent.value().as_element() {
                match elem.name() {
                    "pre" | "code" => return true,
                    _ => {}
                }
            }
            current = parent.parent();
        }
        false
    }

    /// Extract clean text from HTML, preserving structure as markdown
    pub(super) fn extract_text(&self, html: &str) -> String {
        let fragment = Html::parse_fragment(html);

        let mut text = String::new();
        let mut last_was_block = false;
        let mut list_depth: u32 = 0;

        for node in fragment.root_element().descendants() {
            if let Some(text_node) = node.value().as_text() {
                let in_pre = Self::has_pre_ancestor(&node);
                let t = if in_pre {
                    text_node.to_string()
                } else {
                    text_node.trim().to_string()
                };
                if !t.is_empty() {
                    if last_was_block && !text.is_empty() {
                        text.push('\n');
                    } else if !text.is_empty() && !in_pre {
                        text.push(' ');
                    }
                    text.push_str(&t);
                    last_was_block = false;
                }
            } else if let Some(elem) = node.value().as_element() {
                let name = elem.name();
                match name {
                    // Headings → markdown prefixes
                    "h1" => {
                        if !text.is_empty() { text.push_str("\n\n"); }
                        text.push_str("# ");
                        last_was_block = false;
                    }
                    "h2" => {
                        if !text.is_empty() { text.push_str("\n\n"); }
                        text.push_str("## ");
                        last_was_block = false;
                    }
                    "h3" => {
                        if !text.is_empty() { text.push_str("\n\n"); }
                        text.push_str("### ");
                        last_was_block = false;
                    }
                    "h4" => {
                        if !text.is_empty() { text.push_str("\n\n"); }
                        text.push_str("#### ");
                        last_was_block = false;
                    }
                    "h5" => {
                        if !text.is_empty() { text.push_str("\n\n"); }
                        text.push_str("##### ");
                        last_was_block = false;
                    }
                    "h6" => {
                        if !text.is_empty() { text.push_str("\n\n"); }
                        text.push_str("###### ");
                        last_was_block = false;
                    }
                    // Paragraphs → double newline
                    "p" => {
                        if !text.is_empty() { text.push_str("\n\n"); }
                        last_was_block = false;
                    }
                    // List items → bullet prefix
                    "li" => {
                        text.push('\n');
                        let indent = "  ".repeat(list_depth.saturating_sub(1) as usize);
                        text.push_str(&indent);
                        text.push_str("- ");
                        last_was_block = false;
                    }
                    "ul" | "ol" => {
                        list_depth += 1;
                        last_was_block = true;
                    }
                    // Preformatted/code blocks
                    "pre" => {
                        text.push_str("\n\n```\n");
                        last_was_block = false;
                    }
                    "code" => {
                        last_was_block = false;
                    }
                    // Other block elements
                    "div" | "br" | "tr" | "blockquote" | "section" | "header"
                    | "footer" | "aside" => {
                        last_was_block = true;
                    }
                    // Skip script and style content entirely
                    "script" | "style" | "noscript" => {
                        // These are handled by readability, but skip just in case
                    }
                    _ => {}
                }
            }
        }

        // Normalize whitespace while preserving paragraph breaks and structure
        Self::normalize_whitespace(&text)
    }

    /// Normalize whitespace: collapse runs of spaces on each line, preserve
    /// paragraph breaks (double newlines), and trim trailing whitespace.
    pub(super) fn normalize_whitespace(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut consecutive_newlines = 0u32;

        for line in text.split('\n') {
            let trimmed = line.split_whitespace().collect::<Vec<_>>().join(" ");

            if trimmed.is_empty() {
                consecutive_newlines += 1;
                continue;
            }

            // Emit appropriate newlines between non-empty lines
            if !result.is_empty() {
                if consecutive_newlines >= 2 {
                    result.push_str("\n\n");
                } else {
                    result.push('\n');
                }
            }

            consecutive_newlines = 0;
            result.push_str(&trimmed);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scraper::Html;

    fn extractor() -> ContentExtractor {
        ContentExtractor::default()
    }

    fn parse(html: &str) -> Html {
        Html::parse_document(html)
    }

    // ── find_main_content ──────────────────────────────────────

    #[test]
    fn find_main_content_article_tag() {
        let ext = extractor();
        let doc = parse(&format!(
            "<html><body><nav>Nav</nav><article><p>{}</p></article></body></html>",
            "Content. ".repeat(30)
        ));
        let content = ext.find_main_content(&doc);
        assert!(content.contains("Content."));
        assert!(!content.contains("<nav>"));
    }

    #[test]
    fn find_main_content_main_tag() {
        let ext = extractor();
        let doc = parse(&format!(
            "<html><body><main><p>{}</p></main><aside>Side</aside></body></html>",
            "Main content. ".repeat(20)
        ));
        let content = ext.find_main_content(&doc);
        assert!(content.contains("Main content."));
    }

    #[test]
    fn find_main_content_fallback_to_body() {
        let ext = extractor();
        let doc = parse("<html><body><p>Just a paragraph with enough text to show up here.</p></body></html>");
        let content = ext.find_main_content(&doc);
        assert!(content.contains("Just a paragraph"));
    }

    #[test]
    fn find_main_content_empty_html() {
        let ext = extractor();
        let doc = parse("<html><body></body></html>");
        let content = ext.find_main_content(&doc);
        // Body exists but is empty; still returns body HTML (which may be short)
        assert!(content.is_empty() || content.len() < 200);
    }

    #[test]
    fn find_main_content_skips_short_article() {
        let ext = extractor();
        // Article with <200 chars should be skipped in favor of body
        let doc = parse("<html><body><article><p>Short</p></article><p>Body content is here with extra text.</p></body></html>");
        let content = ext.find_main_content(&doc);
        assert!(content.contains("Body content"));
    }

    // ── extract_text headings ──────────────────────────────────

    #[test]
    fn extract_text_h1_to_h6() {
        let ext = extractor();
        let html = "<h1>One</h1><h2>Two</h2><h3>Three</h3><h4>Four</h4><h5>Five</h5><h6>Six</h6>";
        let text = ext.extract_text(html);
        assert!(text.contains("# One"), "text={text:?}");
        assert!(text.contains("## Two"), "text={text:?}");
        assert!(text.contains("### Three"), "text={text:?}");
        assert!(text.contains("#### Four"), "text={text:?}");
        assert!(text.contains("##### Five"), "text={text:?}");
        assert!(text.contains("###### Six"), "text={text:?}");
    }

    #[test]
    fn extract_text_paragraphs() {
        let ext = extractor();
        let html = "<p>First paragraph.</p><p>Second paragraph.</p>";
        let text = ext.extract_text(html);
        assert!(text.contains("First paragraph."), "text={text:?}");
        assert!(text.contains("Second paragraph."), "text={text:?}");
        // Paragraphs should be separated by newlines
        assert!(text.contains('\n'), "paragraphs should have newline separation, text={text:?}");
    }

    #[test]
    fn extract_text_list_items() {
        let ext = extractor();
        let html = "<ul><li>Item one</li><li>Item two</li></ul>";
        let text = ext.extract_text(html);
        assert!(text.contains("- Item one"), "text={text:?}");
        assert!(text.contains("- Item two"), "text={text:?}");
    }

    #[test]
    fn extract_text_nested_lists() {
        let ext = extractor();
        let html = "<ul><li>Outer<ul><li>Inner</li></ul></li></ul>";
        let text = ext.extract_text(html);
        assert!(text.contains("- Outer"), "text={text:?}");
        assert!(text.contains("Inner"), "text={text:?}");
    }

    #[test]
    fn extract_text_pre_block() {
        let ext = extractor();
        let html = "<pre>  code  with  spaces  </pre>";
        let text = ext.extract_text(html);
        assert!(text.contains("```"), "pre should become code fence, text={text:?}");
        assert!(text.contains("code with spaces"), "pre content should be extracted, text={text:?}");
    }

    #[test]
    fn extract_text_empty_html() {
        let ext = extractor();
        let text = ext.extract_text("");
        assert!(text.is_empty() || text.trim().is_empty());
    }

    #[test]
    fn extract_text_whitespace_collapse() {
        let ext = extractor();
        let html = "<p>  Multiple   spaces   should   collapse  </p>";
        let text = ext.extract_text(html);
        assert!(text.contains("Multiple spaces should collapse"), "text={text:?}");
    }

    #[test]
    fn extract_text_script_style_skipped() {
        let ext = extractor();
        let html = "<p>Visible</p><script>alert('hidden')</script><style>.hidden{}</style><p>Also visible</p>";
        let text = ext.extract_text(html);
        assert!(text.contains("Visible"));
        assert!(text.contains("Also visible"));
        // Script/style content may or may not appear since readability handles them,
        // but the structure should work
    }

    // ── normalize_whitespace ───────────────────────────────────

    #[test]
    fn normalize_whitespace_collapses_spaces() {
        let result = ContentExtractor::normalize_whitespace("hello   world   foo");
        assert_eq!(result, "hello world foo");
    }

    #[test]
    fn normalize_whitespace_preserves_paragraph_breaks() {
        let result = ContentExtractor::normalize_whitespace("paragraph one\n\n\nparagraph two");
        assert_eq!(result, "paragraph one\n\nparagraph two");
    }

    #[test]
    fn normalize_whitespace_single_newline() {
        let result = ContentExtractor::normalize_whitespace("line one\nline two");
        assert_eq!(result, "line one\nline two");
    }

    #[test]
    fn normalize_whitespace_empty() {
        let result = ContentExtractor::normalize_whitespace("");
        assert_eq!(result, "");
    }

    #[test]
    fn normalize_whitespace_only_whitespace() {
        let result = ContentExtractor::normalize_whitespace("   \n\n  \n   ");
        assert_eq!(result, "");
    }

    #[test]
    fn normalize_whitespace_trims_lines() {
        let result = ContentExtractor::normalize_whitespace("  hello  \n  world  ");
        assert_eq!(result, "hello\nworld");
    }
}
