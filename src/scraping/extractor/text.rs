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
    fn normalize_whitespace(text: &str) -> String {
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
