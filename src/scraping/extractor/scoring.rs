//! Aggregator detection scoring functions

use scraper::{Html, Selector};

/// Compute an aggregator score for a page (0.0 = original content, 1.0 = pure aggregator).
///
/// Uses three signals:
/// - RSS/Atom feed links (0.15 weight): aggregators often expose feeds
/// - External link density (0.50 weight): ratio of external link text to total text
/// - Content structure (0.35 weight): presence of `<article>`, heavy `<li>` lists, short content
pub fn compute_aggregator_score(document: &Html, text_content: &str) -> f32 {
    // Signal 1: RSS/Atom feed links (weight 0.15)
    let feed_score = detect_feed_links(document);

    // Signal 2: External link density (weight 0.50)
    let link_density_score = compute_external_link_density(document, text_content);

    // Signal 3: Content structure (weight 0.35)
    let structure_score = compute_structure_score(document, text_content);

    let score = feed_score * 0.15 + link_density_score * 0.50 + structure_score * 0.35;
    score.clamp(0.0, 1.0)
}

/// Check for RSS/Atom feed links in the document head.
pub fn detect_feed_links(document: &Html) -> f32 {
    let selectors = [
        "link[type='application/rss+xml']",
        "link[type='application/atom+xml']",
    ];
    for sel_str in &selectors {
        if let Ok(selector) = Selector::parse(sel_str) {
            if document.select(&selector).next().is_some() {
                return 1.0;
            }
        }
    }
    0.0
}

/// Compute external link density: ratio of external `<a>` text length to total text length.
/// Returns 0.0 for <5% density, scales linearly to 1.0 at >=50% density.
pub fn compute_external_link_density(document: &Html, text_content: &str) -> f32 {
    let total_text_len = text_content.len();
    if total_text_len == 0 {
        return 0.0;
    }

    let mut external_link_text_len = 0usize;
    if let Ok(selector) = Selector::parse("a[href]") {
        for element in document.select(&selector) {
            if let Some(href) = element.value().attr("href") {
                // Consider links starting with http:// or https:// as external
                if href.starts_with("http://") || href.starts_with("https://") {
                    let link_text: String = element.text().collect();
                    external_link_text_len += link_text.trim().len();
                }
            }
        }
    }

    let ratio = external_link_text_len as f32 / total_text_len as f32;

    // Scale: <5% → 0.0, >=50% → 1.0
    if ratio < 0.05 {
        0.0
    } else if ratio >= 0.50 {
        1.0
    } else {
        (ratio - 0.05) / 0.45
    }
}

/// Compute content structure score.
/// Presence of `<article>` reduces score; heavy `<li>` lists with links and
/// short content increase it.
pub fn compute_structure_score(document: &Html, text_content: &str) -> f32 {
    let word_count = text_content.split_whitespace().count();
    let mut score = 0.0f32;

    // Has <article> tag → likely real content (reduces score)
    if let Ok(selector) = Selector::parse("article") {
        if document.select(&selector).next().is_some() {
            score -= 0.4;
        }
    }

    // Count <li> elements that contain links
    let mut li_with_links = 0usize;
    let mut total_li = 0usize;
    if let Ok(li_sel) = Selector::parse("li") {
        if let Ok(a_sel) = Selector::parse("a") {
            for li in document.select(&li_sel) {
                total_li += 1;
                if li.select(&a_sel).next().is_some() {
                    li_with_links += 1;
                }
            }
        }
    }

    // Heavy link-lists: many <li> with links, and they dominate the page
    if total_li > 10 && li_with_links as f32 / total_li.max(1) as f32 > 0.7 {
        score += 0.5;
    }

    // Short content is more likely an aggregator page
    if word_count < 200 {
        score += 0.3;
    } else if word_count < 500 {
        score += 0.1;
    }

    score.clamp(0.0, 1.0)
}
