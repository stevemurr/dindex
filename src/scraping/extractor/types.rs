//! Content extraction types

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use thiserror::Error;

/// Errors during content extraction
#[derive(Debug, Error)]
pub enum ExtractError {
    #[error("No content found")]
    NoContent,
    #[error("Failed to parse HTML")]
    ParseError,
    #[error("Content too short: {0} chars")]
    TooShort(usize),
}

/// Extracted content from a web page
#[derive(Debug, Clone)]
pub struct ExtractedContent {
    /// Page title
    pub title: String,
    /// Main text content (clean, no HTML)
    pub text_content: String,
    /// Clean HTML (if available)
    pub clean_html: Option<String>,
    /// Author name
    pub author: Option<String>,
    /// Published date
    pub published_date: Option<DateTime<Utc>>,
    /// Page description/excerpt
    pub excerpt: Option<String>,
    /// Language
    pub language: Option<String>,
    /// Word count
    pub word_count: usize,
    /// Estimated reading time in minutes
    pub reading_time_minutes: u8,
}

/// Full extracted metadata from a page
#[derive(Debug, Clone)]
pub struct ExtractedMetadata {
    /// Original URL
    pub url: String,
    /// Canonical URL if specified
    pub canonical_url: Option<String>,
    /// Page title
    pub title: String,
    /// Description
    pub description: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Published date
    pub published_date: Option<DateTime<Utc>>,
    /// Modified date
    pub modified_date: Option<DateTime<Utc>>,
    /// Language code
    pub language: Option<String>,
    /// Content type (article, product, recipe, etc.)
    pub content_type: ContentType,
    /// Word count
    pub word_count: usize,
    /// Reading time in minutes
    pub reading_time_minutes: u8,
    /// Source domain
    pub domain: String,
    /// When scraped
    pub fetched_at: DateTime<Utc>,
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

/// Type of content on the page
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ContentType {
    #[default]
    Article,
    Product,
    Recipe,
    Video,
    Profile,
    Organization,
    Event,
    Other(String),
}

impl ContentType {
    pub(crate) fn from_schema_type(schema_type: &str) -> Self {
        match schema_type.to_lowercase().as_str() {
            "article" | "newsarticle" | "blogposting" | "technicalarticle" => ContentType::Article,
            "product" => ContentType::Product,
            "recipe" => ContentType::Recipe,
            "video" | "videoobject" => ContentType::Video,
            "person" | "profilepage" => ContentType::Profile,
            "organization" | "localbusiness" => ContentType::Organization,
            "event" => ContentType::Event,
            other => ContentType::Other(other.to_string()),
        }
    }
}

/// Configuration for content extraction
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Minimum content length in characters
    pub min_content_length: usize,
    /// Minimum word count
    pub min_word_count: usize,
    /// Average words per minute for reading time calculation
    pub words_per_minute: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            min_content_length: 100,
            min_word_count: 20,
            words_per_minute: 200,
        }
    }
}
