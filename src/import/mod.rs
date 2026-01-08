//! Bulk import module for offline data dumps
//!
//! This module provides functionality for importing content from offline archives
//! and data dumps, specifically targeting Wikimedia dumps but designed to be
//! extensible to other formats.
//!
//! # Supported Formats
//!
//! - **Wikimedia XML dumps**: Compressed XML with MediaWiki markup (`.xml.bz2`)
//! - **ZIM files**: Kiwix format for offline Wikipedia (future)
//! - **WARC archives**: Web archive format (future)
//!
//! # Example Usage
//!
//! ```no_run
//! use dindex::import::{WikimediaSource, ImportCoordinatorBuilder};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Open a Wikipedia dump
//! let source = WikimediaSource::open("enwiki-latest-pages-articles.xml.bz2")?;
//!
//! // Create coordinator with configuration
//! let mut coordinator = ImportCoordinatorBuilder::new(".dindex")
//!     .with_batch_size(100)
//!     .with_dedup(true)
//!     .build()?;
//!
//! // Run the import
//! let stats = coordinator.import(source)?;
//! println!("Imported {} documents", stats.documents_imported);
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                         Import Coordinator                          │
//! │                   (progress, resume, batching)                      │
//! └─────────────────────────────────────────────────────────────────────┘
//!                                    │
//!                                    ▼
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        DumpSource Trait                             │
//! │           fn iter_documents() -> impl Iterator<DumpDocument>        │
//! └─────────────────────────────────────────────────────────────────────┘
//!          │                        │                        │
//!          ▼                        ▼                        ▼
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │ WikimediaSource │    │   ZimSource     │    │   WarcSource    │
//! │                 │    │   (future)      │    │   (future)      │
//! │ - XML streaming │    │ - ZIM reader    │    │ - WARC parser   │
//! │ - bz2 decompress│    │ - HTML extract  │    │ - CDX lookup    │
//! │ - WikiText parse│    │                 │    │                 │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!                                    │
//!                                    ▼
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Existing Index Pipeline                          │
//! │              (embedding → vector index → storage)                   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

pub mod coordinator;
pub mod progress;
pub mod source;
pub mod wikimedia;
pub mod wikitext;

// Re-export main types
pub use coordinator::{ImportCoordinator, ImportCoordinatorBuilder};
pub use progress::ImportProgress;
pub use source::{
    DumpDocument, DumpFormat, DumpSource, ImportCheckpoint, ImportConfig, ImportError, ImportStats,
};
pub use wikimedia::WikimediaSource;
pub use wikitext::WikiTextParser;
