//! Content processing pipeline: deduplication and document conversion

use url::Url;

use crate::scraping::extractor::{ExtractedContent, ExtractedMetadata};
use crate::types::Document;

/// Convert extracted content to a Document for indexing
pub fn to_document(
    url: &Url,
    content: &ExtractedContent,
    metadata: &ExtractedMetadata,
) -> Document {
    let mut doc = Document::new(&content.text_content)
        .with_title(&content.title)
        .with_url(url.as_str());

    if let Some(author) = &metadata.author {
        doc.metadata.insert("author".to_string(), author.clone());
    }

    if let Some(date) = &metadata.published_date {
        doc.metadata.insert("published_date".to_string(), date.to_rfc3339());
    }

    if let Some(lang) = &metadata.language {
        doc.metadata.insert("language".to_string(), lang.clone());
    }

    doc.metadata.insert("word_count".to_string(), content.word_count.to_string());
    doc.metadata.insert("reading_time".to_string(), content.reading_time_minutes.to_string());
    doc.metadata.insert("domain".to_string(), metadata.domain.clone());

    // Propagate extra metadata from extraction (e.g., aggregator_score)
    for (key, value) in &metadata.extra {
        doc.metadata.insert(key.clone(), value.clone());
    }

    doc
}
