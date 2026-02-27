//! Scrape Job SSE Event Types
//!
//! Defines the real-time events emitted during web scrape jobs,
//! per-URL status tracking, and supporting types.

use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::protocol::JobStats;

/// How a URL entered the frontier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UrlSource {
    Seed,
    Discovered,
}

/// Per-URL processing status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UrlStatus {
    Queued,
    Fetching,
    Indexed,
    Failed,
    Skipped,
}

/// Per-URL tracking info stored in memory during a scrape job.
#[derive(Debug, Clone)]
pub struct UrlInfo {
    pub url: String,
    pub status: UrlStatus,
    pub depth: u8,
    pub title: Option<String>,
    pub error: Option<String>,
    pub chunks_created: usize,
    pub duration_ms: Option<u64>,
    pub updated_at: Instant,
}

/// SSE events emitted during scrape jobs.
///
/// Each variant is serialized as internally-tagged JSON (`"type": "variant_name"`)
/// and sent as an SSE `event:` with the matching name.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ScrapeEvent {
    /// Job has been created and seeds are queued.
    JobStarted {
        job_id: Uuid,
        seed_urls: Vec<String>,
        max_depth: u8,
        max_pages: usize,
    },

    /// A URL was added to the crawl frontier.
    UrlQueued {
        job_id: Uuid,
        url: String,
        depth: u8,
        source: UrlSource,
    },

    /// A URL fetch has begun.
    UrlFetching {
        job_id: Uuid,
        url: String,
    },

    /// A URL was successfully fetched, extracted, and indexed.
    UrlIndexed {
        job_id: Uuid,
        url: String,
        title: Option<String>,
        word_count: usize,
        chunks_created: usize,
        duration_ms: u64,
        discovered_urls: usize,
    },

    /// A URL failed to fetch or extract.
    UrlFailed {
        job_id: Uuid,
        url: String,
        error: String,
        duration_ms: u64,
    },

    /// A URL was skipped (duplicate, robots.txt, etc.).
    UrlSkipped {
        job_id: Uuid,
        url: String,
        reason: String,
    },

    /// Aggregate progress snapshot emitted after each URL.
    Progress {
        job_id: Uuid,
        urls_processed: u64,
        urls_succeeded: u64,
        urls_failed: u64,
        urls_skipped: u64,
        urls_queued: usize,
        chunks_indexed: usize,
        elapsed_ms: u64,
        rate: Option<f64>,
        eta_seconds: Option<u64>,
    },

    /// Job finished (completed, failed, or cancelled).
    JobCompleted {
        job_id: Uuid,
        status: String,
        stats: Option<JobStats>,
        error: Option<String>,
    },
}

impl ScrapeEvent {
    /// Returns the SSE `event:` field name for this event.
    pub fn event_name(&self) -> &'static str {
        match self {
            ScrapeEvent::JobStarted { .. } => "job_started",
            ScrapeEvent::UrlQueued { .. } => "url_queued",
            ScrapeEvent::UrlFetching { .. } => "url_fetching",
            ScrapeEvent::UrlIndexed { .. } => "url_indexed",
            ScrapeEvent::UrlFailed { .. } => "url_failed",
            ScrapeEvent::UrlSkipped { .. } => "url_skipped",
            ScrapeEvent::Progress { .. } => "progress",
            ScrapeEvent::JobCompleted { .. } => "job_completed",
        }
    }
}
