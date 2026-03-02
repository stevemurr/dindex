//! Metadata extraction: author, dates, language, JSON-LD, OpenGraph, Twitter Cards

use chrono::{DateTime, Utc};
use scraper::{Html, Selector};
use std::collections::HashMap;

use super::ContentExtractor;
use super::types::{ContentType, ExtractedMetadata};
use super::scoring;

impl ContentExtractor {
    /// Build `ExtractedMetadata` from an already-parsed `Html` document.
    ///
    /// Shared helper used by both `extract_all` and `extract_metadata` to
    /// avoid duplicating the metadata assembly logic.
    pub(super) fn build_metadata_from_document(
        &self,
        document: &Html,
        url: &url::Url,
        text_content: &str,
        word_count: usize,
        reading_time: u8,
        fallback_author: Option<String>,
        fallback_date: Option<DateTime<Utc>>,
        fallback_language: Option<String>,
    ) -> ExtractedMetadata {
        // Extract JSON-LD
        let json_ld = self.extract_json_ld(document);

        // Extract OpenGraph
        let og = self.extract_opengraph(document);

        // Extract Twitter cards
        let twitter = self.extract_twitter_cards(document);

        // Extract meta tags
        let meta = self.extract_meta_tags(document);

        // Extract title with fallback chain
        let title = json_ld
            .get("headline")
            .or_else(|| json_ld.get("name"))
            .or_else(|| og.get("title"))
            .or_else(|| twitter.get("title"))
            .or_else(|| meta.get("title"))
            .cloned()
            .unwrap_or_else(|| self.extract_title(document));

        // Extract description
        let description = json_ld
            .get("description")
            .or_else(|| og.get("description"))
            .or_else(|| twitter.get("description"))
            .or_else(|| meta.get("description"))
            .cloned();

        // Extract author
        let author = json_ld
            .get("author")
            .or_else(|| meta.get("author"))
            .cloned()
            .or(fallback_author);

        // Extract dates
        let published_date = json_ld
            .get("datePublished")
            .or_else(|| meta.get("date"))
            .and_then(|d| Self::parse_date(d))
            .or_else(|| {
                self.get_meta_content(document, "article:published_time")
                    .and_then(|d| Self::parse_date(&d))
            })
            .or(fallback_date);

        let modified_date = json_ld
            .get("dateModified")
            .and_then(|d| Self::parse_date(d));

        // Extract language
        let language = meta
            .get("language")
            .cloned()
            .or_else(|| self.extract_language(document))
            .or(fallback_language);

        // Determine content type
        let content_type = json_ld
            .get("@type")
            .map(|t| ContentType::from_schema_type(t))
            .unwrap_or_default();

        // Extract canonical URL
        let canonical_url = self.extract_canonical(document);

        let aggregator_score = scoring::compute_aggregator_score(document, text_content);
        let mut extra = HashMap::new();
        if aggregator_score > 0.0 {
            extra.insert(
                "aggregator_score".to_string(),
                format!("{:.3}", aggregator_score),
            );
        }

        ExtractedMetadata {
            url: url.to_string(),
            canonical_url,
            title,
            description,
            author,
            published_date,
            modified_date,
            language,
            content_type,
            word_count,
            reading_time_minutes: reading_time,
            domain: url.host_str().unwrap_or_default().to_string(),
            fetched_at: Utc::now(),
            extra,
        }
    }

    /// Extract page title
    pub(super) fn extract_title(&self, document: &Html) -> String {
        // Try og:title first
        if let Some(og_title) = self.get_meta_content(document, "og:title") {
            return og_title;
        }

        // Then title tag
        if let Ok(selector) = Selector::parse("title") {
            if let Some(title_elem) = document.select(&selector).next() {
                let title = title_elem.text().collect::<String>().trim().to_string();
                if !title.is_empty() {
                    return title;
                }
            }
        }

        // Try h1
        if let Ok(selector) = Selector::parse("h1") {
            if let Some(h1_elem) = document.select(&selector).next() {
                let title = h1_elem.text().collect::<String>().trim().to_string();
                if !title.is_empty() {
                    return title;
                }
            }
        }

        "Untitled".to_string()
    }

    /// Extract author
    pub(super) fn extract_author(&self, document: &Html) -> Option<String> {
        // Try meta author
        if let Some(author) = self.get_meta_content(document, "author") {
            return Some(author);
        }

        // Try schema.org author
        if let Ok(selector) = Selector::parse("[itemprop='author']") {
            if let Some(elem) = document.select(&selector).next() {
                let text = elem.text().collect::<String>().trim().to_string();
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }

        // Try common author class names
        for class in &[".author", ".byline", ".writer", "[rel='author']"] {
            if let Ok(selector) = Selector::parse(class) {
                if let Some(elem) = document.select(&selector).next() {
                    let text = elem.text().collect::<String>().trim().to_string();
                    if !text.is_empty() && text.len() < 100 {
                        return Some(text);
                    }
                }
            }
        }

        None
    }

    /// Extract publication date
    pub(super) fn extract_date(&self, document: &Html) -> Option<DateTime<Utc>> {
        // Try time element with datetime attribute
        if let Ok(selector) = Selector::parse("time[datetime]") {
            if let Some(elem) = document.select(&selector).next() {
                if let Some(datetime) = elem.value().attr("datetime") {
                    if let Some(dt) = Self::parse_date(datetime) {
                        return Some(dt);
                    }
                }
            }
        }

        // Try meta date
        for name in &[
            "article:published_time",
            "date",
            "DC.date",
            "datePublished",
        ] {
            if let Some(date_str) = self.get_meta_content(document, name) {
                if let Some(dt) = Self::parse_date(&date_str) {
                    return Some(dt);
                }
            }
        }

        None
    }

    /// Extract language
    pub(super) fn extract_language(&self, document: &Html) -> Option<String> {
        // Try html lang attribute
        if let Ok(selector) = Selector::parse("html") {
            if let Some(html_elem) = document.select(&selector).next() {
                if let Some(lang) = html_elem.value().attr("lang") {
                    return Some(lang.to_string());
                }
            }
        }

        // Try meta language
        self.get_meta_content(document, "language")
            .or_else(|| self.get_meta_content(document, "og:locale"))
    }

    /// Extract excerpt/description
    pub(super) fn extract_excerpt(&self, document: &Html, text_content: &str) -> Option<String> {
        // Try meta description first
        if let Some(desc) = self.get_meta_content(document, "description") {
            if !desc.is_empty() {
                return Some(desc);
            }
        }

        if let Some(desc) = self.get_meta_content(document, "og:description") {
            if !desc.is_empty() {
                return Some(desc);
            }
        }

        // Generate from first paragraph
        let words: Vec<&str> = text_content.split_whitespace().take(50).collect();
        if words.len() >= 10 {
            let excerpt = words.join(" ");
            return Some(if words.len() == 50 {
                format!("{}...", excerpt)
            } else {
                excerpt
            });
        }

        None
    }

    /// Extract canonical URL
    pub(super) fn extract_canonical(&self, document: &Html) -> Option<String> {
        if let Ok(selector) = Selector::parse("link[rel='canonical']") {
            if let Some(elem) = document.select(&selector).next() {
                return elem.value().attr("href").map(|s| s.to_string());
            }
        }
        self.get_meta_content(document, "og:url")
    }

    /// Get meta content by name or property, using pre-compiled selectors when available
    pub(super) fn get_meta_content(&self, document: &Html, name: &str) -> Option<String> {
        // Try cached selectors first
        if let Some((name_sel, prop_sel)) = self.meta_selectors.get(name) {
            // Try name attribute
            if let Some(selector) = name_sel {
                if let Some(elem) = document.select(selector).next() {
                    if let Some(content) = elem.value().attr("content") {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() {
                            return Some(trimmed.to_string());
                        }
                    }
                }
            }

            // Try property attribute (for OpenGraph)
            if let Some(selector) = prop_sel {
                if let Some(elem) = document.select(selector).next() {
                    if let Some(content) = elem.value().attr("content") {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() {
                            return Some(trimmed.to_string());
                        }
                    }
                }
            }

            return None;
        }

        // Fallback: parse selectors dynamically for uncached names
        let name_selector_str = format!("meta[name='{}']", name);
        if let Ok(selector) = Selector::parse(&name_selector_str) {
            if let Some(elem) = document.select(&selector).next() {
                if let Some(content) = elem.value().attr("content") {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
        }

        let prop_selector_str = format!("meta[property='{}']", name);
        if let Ok(selector) = Selector::parse(&prop_selector_str) {
            if let Some(elem) = document.select(&selector).next() {
                if let Some(content) = elem.value().attr("content") {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
        }

        None
    }

    /// Extract JSON-LD structured data
    fn extract_json_ld(&self, document: &Html) -> HashMap<String, String> {
        let mut data = HashMap::new();

        if let Ok(selector) = Selector::parse("script[type='application/ld+json']") {
            for script in document.select(&selector) {
                let json_text = script.text().collect::<String>();
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_text) {
                    Self::flatten_json_ld(&value, &mut data);
                }
            }
        }

        data
    }

    /// Flatten JSON-LD to simple key-value pairs
    fn flatten_json_ld(value: &serde_json::Value, data: &mut HashMap<String, String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, val) in map {
                    match val {
                        serde_json::Value::String(s) => {
                            data.insert(key.clone(), s.clone());
                        }
                        serde_json::Value::Object(nested) => {
                            // Handle nested author, etc.
                            if let Some(name) = nested.get("name") {
                                if let Some(s) = name.as_str() {
                                    data.insert(key.clone(), s.to_string());
                                }
                            }
                        }
                        serde_json::Value::Array(arr) => {
                            // Handle @graph arrays
                            for item in arr {
                                Self::flatten_json_ld(item, data);
                            }
                        }
                        _ => {}
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::flatten_json_ld(item, data);
                }
            }
            _ => {}
        }
    }

    /// Extract OpenGraph metadata
    fn extract_opengraph(&self, document: &Html) -> HashMap<String, String> {
        let mut data = HashMap::new();

        for prop in &["title", "description", "type", "url", "image", "locale"] {
            let key = format!("og:{}", prop);
            if let Some(value) = self.get_meta_content(document, &key) {
                data.insert(prop.to_string(), value);
            }
        }

        data
    }

    /// Extract Twitter Cards metadata
    fn extract_twitter_cards(&self, document: &Html) -> HashMap<String, String> {
        let mut data = HashMap::new();

        for prop in &["title", "description", "card", "site", "creator"] {
            let key = format!("twitter:{}", prop);
            if let Some(value) = self.get_meta_content(document, &key) {
                data.insert(prop.to_string(), value);
            }
        }

        data
    }

    /// Extract basic meta tags
    fn extract_meta_tags(&self, document: &Html) -> HashMap<String, String> {
        let mut data = HashMap::new();

        for name in &["title", "description", "author", "keywords", "date", "language"] {
            if let Some(value) = self.get_meta_content(document, name) {
                data.insert(name.to_string(), value);
            }
        }

        data
    }

    /// Parse a date string into DateTime
    pub(super) fn parse_date(date_str: &str) -> Option<DateTime<Utc>> {
        // Try ISO 8601 formats
        if let Ok(dt) = DateTime::parse_from_rfc3339(date_str) {
            return Some(dt.with_timezone(&Utc));
        }

        if let Ok(dt) = DateTime::parse_from_rfc2822(date_str) {
            return Some(dt.with_timezone(&Utc));
        }

        // Try common formats
        let formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
        ];

        for format in &formats {
            if let Ok(naive) = chrono::NaiveDate::parse_from_str(date_str, format) {
                if let Some(naive_dt) = naive.and_hms_opt(0, 0, 0) {
                    return Some(DateTime::from_naive_utc_and_offset(naive_dt, Utc));
                }
            }
            if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(date_str, format) {
                return Some(DateTime::from_naive_utc_and_offset(naive, Utc));
            }
        }

        None
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

    // ── extract_title ──────────────────────────────────────────

    #[test]
    fn title_from_og() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta property="og:title" content="OG Title"><title>HTML Title</title></head><body></body></html>"#);
        assert_eq!(ext.extract_title(&doc), "OG Title");
    }

    #[test]
    fn title_from_title_tag() {
        let ext = extractor();
        let doc = parse("<html><head><title>Page Title</title></head><body></body></html>");
        assert_eq!(ext.extract_title(&doc), "Page Title");
    }

    #[test]
    fn title_from_h1() {
        let ext = extractor();
        let doc = parse("<html><head></head><body><h1>Heading Title</h1></body></html>");
        assert_eq!(ext.extract_title(&doc), "Heading Title");
    }

    #[test]
    fn title_fallback_untitled() {
        let ext = extractor();
        let doc = parse("<html><head></head><body><p>No title here.</p></body></html>");
        assert_eq!(ext.extract_title(&doc), "Untitled");
    }

    #[test]
    fn title_empty_title_tag_falls_through() {
        let ext = extractor();
        let doc = parse("<html><head><title>   </title></head><body><h1>Real Title</h1></body></html>");
        assert_eq!(ext.extract_title(&doc), "Real Title");
    }

    // ── extract_author ─────────────────────────────────────────

    #[test]
    fn author_from_meta() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta name="author" content="Jane Doe"></head><body></body></html>"#);
        assert_eq!(ext.extract_author(&doc), Some("Jane Doe".to_string()));
    }

    #[test]
    fn author_from_itemprop() {
        let ext = extractor();
        let doc = parse(r#"<html><body><span itemprop="author">John Smith</span></body></html>"#);
        assert_eq!(ext.extract_author(&doc), Some("John Smith".to_string()));
    }

    #[test]
    fn author_from_class() {
        let ext = extractor();
        let doc = parse(r#"<html><body><span class="author">Author Name</span></body></html>"#);
        assert_eq!(ext.extract_author(&doc), Some("Author Name".to_string()));
    }

    #[test]
    fn author_from_byline() {
        let ext = extractor();
        let doc = parse(r#"<html><body><div class="byline">Byline Author</div></body></html>"#);
        assert_eq!(ext.extract_author(&doc), Some("Byline Author".to_string()));
    }

    #[test]
    fn author_none_when_missing() {
        let ext = extractor();
        let doc = parse("<html><body><p>No author info</p></body></html>");
        assert_eq!(ext.extract_author(&doc), None);
    }

    #[test]
    fn author_rejects_long_text() {
        let ext = extractor();
        let long_name = "A".repeat(150);
        let html = format!(r#"<html><body><span class="author">{}</span></body></html>"#, long_name);
        let doc = parse(&html);
        // author class text > 100 chars should be rejected
        assert_eq!(ext.extract_author(&doc), None);
    }

    // ── extract_date ───────────────────────────────────────────

    #[test]
    fn date_from_time_element() {
        let ext = extractor();
        let doc = parse(r#"<html><body><time datetime="2024-06-15T12:00:00Z">June 15</time></body></html>"#);
        let date = ext.extract_date(&doc);
        assert!(date.is_some());
        assert_eq!(date.unwrap().format("%Y-%m-%d").to_string(), "2024-06-15");
    }

    #[test]
    fn date_from_meta_article_published() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta property="article:published_time" content="2023-03-20T08:30:00Z"></head><body></body></html>"#);
        let date = ext.extract_date(&doc);
        assert!(date.is_some());
    }

    #[test]
    fn date_none_when_missing() {
        let ext = extractor();
        let doc = parse("<html><body><p>No date</p></body></html>");
        assert_eq!(ext.extract_date(&doc), None);
    }

    // ── parse_date ─────────────────────────────────────────────

    #[test]
    fn parse_date_rfc3339() {
        let dt = ContentExtractor::parse_date("2024-01-15T10:30:00+00:00");
        assert!(dt.is_some());
    }

    #[test]
    fn parse_date_rfc2822() {
        let dt = ContentExtractor::parse_date("Mon, 15 Jan 2024 10:30:00 +0000");
        assert!(dt.is_some());
    }

    #[test]
    fn parse_date_iso_date_only() {
        let dt = ContentExtractor::parse_date("2024-01-15");
        assert!(dt.is_some());
        assert_eq!(dt.unwrap().format("%Y-%m-%d").to_string(), "2024-01-15");
    }

    #[test]
    fn parse_date_slash_format() {
        let dt = ContentExtractor::parse_date("2024/06/20");
        assert!(dt.is_some());
    }

    #[test]
    fn parse_date_month_name() {
        let dt = ContentExtractor::parse_date("January 15, 2024");
        assert!(dt.is_some());
    }

    #[test]
    fn parse_date_abbreviated_month() {
        let dt = ContentExtractor::parse_date("Jan 15, 2024");
        assert!(dt.is_some());
    }

    #[test]
    fn parse_date_iso_datetime_no_tz() {
        let dt = ContentExtractor::parse_date("2024-01-15T10:30:00");
        assert!(dt.is_some());
    }

    #[test]
    fn parse_date_invalid() {
        assert!(ContentExtractor::parse_date("not a date").is_none());
        assert!(ContentExtractor::parse_date("").is_none());
    }

    // ── extract_language ───────────────────────────────────────

    #[test]
    fn language_from_html_attr() {
        let ext = extractor();
        let doc = parse(r#"<html lang="fr"><body></body></html>"#);
        assert_eq!(ext.extract_language(&doc), Some("fr".to_string()));
    }

    #[test]
    fn language_from_meta() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta name="language" content="de"></head><body></body></html>"#);
        assert_eq!(ext.extract_language(&doc), Some("de".to_string()));
    }

    #[test]
    fn language_from_og_locale() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta property="og:locale" content="en_US"></head><body></body></html>"#);
        assert_eq!(ext.extract_language(&doc), Some("en_US".to_string()));
    }

    #[test]
    fn language_none_when_missing() {
        let ext = extractor();
        let doc = parse("<html><body></body></html>");
        assert_eq!(ext.extract_language(&doc), None);
    }

    // ── extract_excerpt ────────────────────────────────────────

    #[test]
    fn excerpt_from_meta_description() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta name="description" content="A great article about Rust."></head><body></body></html>"#);
        assert_eq!(ext.extract_excerpt(&doc, ""), Some("A great article about Rust.".to_string()));
    }

    #[test]
    fn excerpt_from_og_description() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta property="og:description" content="OG excerpt here."></head><body></body></html>"#);
        assert_eq!(ext.extract_excerpt(&doc, ""), Some("OG excerpt here.".to_string()));
    }

    #[test]
    fn excerpt_generated_from_text() {
        let ext = extractor();
        let doc = parse("<html><body></body></html>");
        let text = "word ".repeat(30);
        let excerpt = ext.extract_excerpt(&doc, &text);
        assert!(excerpt.is_some());
    }

    #[test]
    fn excerpt_none_for_short_text() {
        let ext = extractor();
        let doc = parse("<html><body></body></html>");
        assert_eq!(ext.extract_excerpt(&doc, "too short"), None);
    }

    #[test]
    fn excerpt_adds_ellipsis_for_long_text() {
        let ext = extractor();
        let doc = parse("<html><body></body></html>");
        let text = "word ".repeat(100);
        let excerpt = ext.extract_excerpt(&doc, &text).unwrap();
        assert!(excerpt.ends_with("..."));
    }

    // ── extract_canonical ──────────────────────────────────────

    #[test]
    fn canonical_from_link_tag() {
        let ext = extractor();
        let doc = parse(r#"<html><head><link rel="canonical" href="https://example.com/canonical"></head><body></body></html>"#);
        assert_eq!(ext.extract_canonical(&doc), Some("https://example.com/canonical".to_string()));
    }

    #[test]
    fn canonical_from_og_url() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta property="og:url" content="https://example.com/og-url"></head><body></body></html>"#);
        assert_eq!(ext.extract_canonical(&doc), Some("https://example.com/og-url".to_string()));
    }

    #[test]
    fn canonical_none_when_missing() {
        let ext = extractor();
        let doc = parse("<html><head></head><body></body></html>");
        assert_eq!(ext.extract_canonical(&doc), None);
    }

    // ── get_meta_content ───────────────────────────────────────

    #[test]
    fn get_meta_content_by_name() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta name="author" content="Test Author"></head><body></body></html>"#);
        assert_eq!(ext.get_meta_content(&doc, "author"), Some("Test Author".to_string()));
    }

    #[test]
    fn get_meta_content_by_property() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta property="og:title" content="OG Title Value"></head><body></body></html>"#);
        assert_eq!(ext.get_meta_content(&doc, "og:title"), Some("OG Title Value".to_string()));
    }

    #[test]
    fn get_meta_content_trims_whitespace() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta name="description" content="  spaced content  "></head><body></body></html>"#);
        assert_eq!(ext.get_meta_content(&doc, "description"), Some("spaced content".to_string()));
    }

    #[test]
    fn get_meta_content_empty_returns_none() {
        let ext = extractor();
        let doc = parse(r#"<html><head><meta name="description" content=""></head><body></body></html>"#);
        assert_eq!(ext.get_meta_content(&doc, "description"), None);
    }

    #[test]
    fn get_meta_content_missing_returns_none() {
        let ext = extractor();
        let doc = parse("<html><head></head><body></body></html>");
        assert_eq!(ext.get_meta_content(&doc, "nonexistent"), None);
    }

    // ── JSON-LD extraction ─────────────────────────────────────

    #[test]
    fn json_ld_extracts_fields() {
        let ext = extractor();
        let doc = parse(r#"<html><head><script type="application/ld+json">{"@type": "Article", "headline": "Test Headline", "author": {"name": "Author Name"}}</script></head><body></body></html>"#);
        let json_ld = ext.extract_json_ld(&doc);
        assert_eq!(json_ld.get("headline"), Some(&"Test Headline".to_string()));
        assert_eq!(json_ld.get("author"), Some(&"Author Name".to_string()));
        assert_eq!(json_ld.get("@type"), Some(&"Article".to_string()));
    }

    #[test]
    fn json_ld_handles_graph_array() {
        let ext = extractor();
        let doc = parse(r#"<html><head><script type="application/ld+json">{"@graph": [{"@type": "Article", "headline": "Graph Headline"}]}</script></head><body></body></html>"#);
        let json_ld = ext.extract_json_ld(&doc);
        assert_eq!(json_ld.get("headline"), Some(&"Graph Headline".to_string()));
    }

    #[test]
    fn json_ld_empty_when_missing() {
        let ext = extractor();
        let doc = parse("<html><head></head><body></body></html>");
        let json_ld = ext.extract_json_ld(&doc);
        assert!(json_ld.is_empty());
    }

    #[test]
    fn json_ld_handles_invalid_json() {
        let ext = extractor();
        let doc = parse(r#"<html><head><script type="application/ld+json">not valid json</script></head><body></body></html>"#);
        let json_ld = ext.extract_json_ld(&doc);
        assert!(json_ld.is_empty());
    }

    // ── OpenGraph extraction ───────────────────────────────────

    #[test]
    fn opengraph_extracts_all_fields() {
        let ext = extractor();
        let doc = parse(r#"<html><head>
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG Desc">
            <meta property="og:type" content="article">
            <meta property="og:url" content="https://example.com">
        </head><body></body></html>"#);
        let og = ext.extract_opengraph(&doc);
        assert_eq!(og.get("title"), Some(&"OG Title".to_string()));
        assert_eq!(og.get("description"), Some(&"OG Desc".to_string()));
        assert_eq!(og.get("type"), Some(&"article".to_string()));
    }

    #[test]
    fn opengraph_empty_when_missing() {
        let ext = extractor();
        let doc = parse("<html><head></head><body></body></html>");
        let og = ext.extract_opengraph(&doc);
        assert!(og.is_empty());
    }

    // ── Twitter Cards extraction ───────────────────────────────

    #[test]
    fn twitter_cards_extracts_fields() {
        let ext = extractor();
        let doc = parse(r#"<html><head>
            <meta name="twitter:title" content="Tweet Title">
            <meta name="twitter:description" content="Tweet Desc">
            <meta name="twitter:card" content="summary_large_image">
        </head><body></body></html>"#);
        let tw = ext.extract_twitter_cards(&doc);
        assert_eq!(tw.get("title"), Some(&"Tweet Title".to_string()));
        assert_eq!(tw.get("description"), Some(&"Tweet Desc".to_string()));
        assert_eq!(tw.get("card"), Some(&"summary_large_image".to_string()));
    }

    // ── build_metadata_from_document ───────────────────────────

    #[test]
    fn build_metadata_title_fallback_chain() {
        let ext = extractor();
        let url = url::Url::parse("https://example.com/page").unwrap();

        // JSON-LD headline takes priority
        let doc = parse(r#"<html><head>
            <script type="application/ld+json">{"headline": "JSON-LD Title"}</script>
            <meta property="og:title" content="OG Title">
            <title>HTML Title</title>
        </head><body></body></html>"#);
        let meta = ext.build_metadata_from_document(&doc, &url, "text", 100, 1, None, None, None);
        assert_eq!(meta.title, "JSON-LD Title");

        // Without JSON-LD, OG title takes over
        let doc2 = parse(r#"<html><head>
            <meta property="og:title" content="OG Title">
            <title>HTML Title</title>
        </head><body></body></html>"#);
        let meta2 = ext.build_metadata_from_document(&doc2, &url, "text", 100, 1, None, None, None);
        assert_eq!(meta2.title, "OG Title");
    }

    #[test]
    fn build_metadata_uses_fallback_author() {
        let ext = extractor();
        let url = url::Url::parse("https://example.com").unwrap();
        let doc = parse("<html><body></body></html>");
        let meta = ext.build_metadata_from_document(
            &doc, &url, "text", 100, 1,
            Some("Fallback Author".to_string()), None, None,
        );
        assert_eq!(meta.author, Some("Fallback Author".to_string()));
    }

    #[test]
    fn build_metadata_domain_extracted() {
        let ext = extractor();
        let url = url::Url::parse("https://example.com/path").unwrap();
        let doc = parse("<html><body></body></html>");
        let meta = ext.build_metadata_from_document(&doc, &url, "text", 100, 1, None, None, None);
        assert_eq!(meta.domain, "example.com");
        assert_eq!(meta.url, "https://example.com/path");
    }

    #[test]
    fn build_metadata_content_type_from_json_ld() {
        let ext = extractor();
        let url = url::Url::parse("https://example.com").unwrap();
        let doc = parse(r#"<html><head><script type="application/ld+json">{"@type": "Recipe"}</script></head><body></body></html>"#);
        let meta = ext.build_metadata_from_document(&doc, &url, "text", 100, 1, None, None, None);
        assert_eq!(meta.content_type, ContentType::Recipe);
    }

    #[test]
    fn build_metadata_aggregator_score_stored() {
        let ext = extractor();
        let url = url::Url::parse("https://example.com").unwrap();
        // A page with feed links should get an aggregator score
        let doc = parse(r#"<html><head><link type="application/rss+xml" href="/feed"></head><body></body></html>"#);
        let text = "word ".repeat(50);
        let meta = ext.build_metadata_from_document(&doc, &url, &text, 50, 1, None, None, None);
        // Should have aggregator_score in extra if score > 0
        if let Some(score_str) = meta.extra.get("aggregator_score") {
            let score: f32 = score_str.parse().unwrap();
            assert!(score > 0.0);
        }
    }
}
