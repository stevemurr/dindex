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
