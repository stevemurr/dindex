//! Web scraping subsystem for the decentralized semantic search index
//!
//! This module implements a distributed, polite, LLM-optimized web scraping system
//! that feeds content into the decentralized semantic search index. The architecture
//! uses consistent hashing to partition domains across nodes, with gossip protocols
//! for URL exchange and SimHash for content deduplication.
//!
//! Key components:
//! - `DomainAssignment`: Consistent hashing for domain-to-node mapping
//! - `UrlFrontier`: Per-domain priority queues with URL scoring
//! - `PolitenessController`: robots.txt handling and rate limiting
//! - `FetchEngine`: HTTP + headless browser fetching
//! - `ContentExtractor`: HTML to clean text extraction
//! - `Deduplicator`: URL Bloom filter + SimHash content deduplication
//! - `ScrapingCoordinator`: Orchestrates the entire scraping pipeline

pub mod coordinator;
pub mod dedup;
pub mod domain_assignment;
pub mod extractor;
pub mod fetcher;
pub mod frontier;
pub mod politeness;

pub use coordinator::ScrapingCoordinator;
pub use dedup::{ContentDeduplicator, UrlDeduplicator};
pub use domain_assignment::DomainAssignment;
pub use extractor::ContentExtractor;
pub use fetcher::FetchEngine;
pub use frontier::{ScoredUrl, UrlFrontier};
pub use politeness::{FetchDecision, PolitenessController};
