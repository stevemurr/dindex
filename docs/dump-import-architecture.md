# Offline Dump Import Architecture

## Overview

Add support for importing content from offline archives and data dumps, specifically targeting Wikimedia dumps (dumps.wikimedia.org) but designed to be extensible to other formats.

This is a **separate module** from the web scraping subsystem - it handles bulk offline data import rather than live crawling.

## Goals

1. Stream-process large compressed dumps (20+ GB) with bounded memory
2. Convert source formats (WikiText, ZIM) to clean plaintext
3. Feed content directly into existing embedding/indexing pipeline
4. Support resumable imports for crash recovery
5. Provide progress reporting for long-running imports

## Supported Formats

### Priority 1: Wikimedia XML Dumps
- Source: `dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2`
- Format: Compressed XML with MediaWiki markup
- Size: ~20GB compressed, ~80GB uncompressed for English Wikipedia

### Priority 2: ZIM Files (Kiwix)
- Source: `download.kiwix.org/zim/`
- Format: Compressed archive with pre-rendered HTML
- Use case: Wikipedia, Stack Exchange, Project Gutenberg, etc.

### Priority 3: Common Crawl WARC
- Source: `commoncrawl.org`
- Format: WARC (Web ARChive) with CDX index
- Use case: Bulk web archive import

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Import Coordinator                          │
│                   (progress, resume, batching)                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DumpSource Trait                             │
│           fn iter_documents() -> impl Iterator<DumpDocument>        │
└─────────────────────────────────────────────────────────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ WikimediaSource │    │   ZimSource     │    │   WarcSource    │
│                 │    │                 │    │                 │
│ - XML streaming │    │ - libzim/zim-rs │    │ - warc crate    │
│ - bz2 decompress│    │ - HTML extract  │    │ - CDX lookup    │
│ - WikiText parse│    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Content Processor                              │
│              (text cleaning, chunking, dedup)                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Existing Index Pipeline                          │
│              (embedding → vector index → storage)                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/import/
├── mod.rs              # Module exports
├── coordinator.rs      # Import orchestration, progress, resume
├── source.rs           # DumpSource trait definition
├── wikimedia.rs        # Wikimedia XML dump parser
├── zim.rs              # ZIM file reader
├── warc.rs             # WARC archive reader (future)
├── wikitext.rs         # WikiText → plaintext converter
└── progress.rs         # Progress tracking and checkpointing
```

## Core Types

```rust
/// A document extracted from a dump
pub struct DumpDocument {
    /// Unique identifier within the dump (e.g., Wikipedia page ID)
    pub id: String,
    /// Document title
    pub title: String,
    /// Clean plaintext content
    pub content: String,
    /// Source URL (reconstructed for Wikipedia, actual for WARC)
    pub url: Option<String>,
    /// Last modification timestamp
    pub modified: Option<chrono::DateTime<chrono::Utc>>,
    /// Source-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Trait for dump sources
#[async_trait]
pub trait DumpSource: Send {
    /// Iterate over documents in the dump
    fn iter_documents(&mut self) -> Box<dyn Iterator<Item = Result<DumpDocument, ImportError>> + '_>;

    /// Get total document count if known (for progress)
    fn document_count_hint(&self) -> Option<u64>;

    /// Get current byte position (for resume)
    fn byte_position(&self) -> u64;

    /// Seek to byte position (for resume)
    fn seek_to(&mut self, position: u64) -> Result<(), ImportError>;
}

/// Import configuration
pub struct ImportConfig {
    /// Batch size for indexing
    pub batch_size: usize,
    /// Enable content deduplication
    pub deduplicate: bool,
    /// Checkpoint interval (documents)
    pub checkpoint_interval: usize,
    /// Checkpoint file path
    pub checkpoint_path: Option<PathBuf>,
    /// Filter: minimum content length
    pub min_content_length: usize,
    /// Filter: namespace allowlist (Wikipedia-specific)
    pub allowed_namespaces: Option<Vec<i32>>,
}
```

## Implementation Steps

### Phase 1: Core Infrastructure

#### 1.1 Add Dependencies to Cargo.toml
```toml
# Compression
bzip2 = "0.4"
flate2 = "1.0"

# XML parsing (streaming)
quick-xml = "0.31"

# WikiText parsing
parse_wiki_text = "0.1"  # or custom parser

# ZIM support (Phase 2)
# zim = "0.1"  # if available, else libzim bindings

# Progress reporting
indicatif = "0.17"
```

#### 1.2 Create Module Structure
- Create `src/import/mod.rs` with module exports
- Define `DumpSource` trait in `src/import/source.rs`
- Define `DumpDocument` and `ImportConfig` types

#### 1.3 Implement Import Coordinator
```rust
impl ImportCoordinator {
    /// Run import from a dump source
    pub async fn import<S: DumpSource>(&mut self, source: S) -> Result<ImportStats, ImportError>;

    /// Resume import from checkpoint
    pub async fn resume<S: DumpSource>(&mut self, source: S) -> Result<ImportStats, ImportError>;
}
```

### Phase 2: Wikimedia XML Support

#### 2.1 WikiText Parser
Create `src/import/wikitext.rs`:
- Strip `[[internal links]]` → keep display text
- Remove `{{templates}}`
- Strip `{| tables |}`
- Remove `<ref>` tags and references section
- Remove categories, interwiki links
- Handle `'''bold'''` and `''italic''` markup
- Preserve paragraph structure

Example transformations:
```
Input:  "The [[United States|US]] is a {{country}}."
Output: "The US is a country."

Input:  "Population: 300 million<ref>Census 2020</ref>"
Output: "Population: 300 million"
```

#### 2.2 Wikimedia XML Stream Parser
Create `src/import/wikimedia.rs`:
```rust
pub struct WikimediaSource {
    reader: BufReader<BzDecoder<File>>,
    parser: quick_xml::Reader<...>,
    current_page: Option<PartialPage>,
    bytes_read: u64,
}

impl WikimediaSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ImportError>;
}

impl DumpSource for WikimediaSource {
    fn iter_documents(&mut self) -> ... {
        // Stream XML, yield documents one at a time
        // Parse <page> elements:
        //   <title>Article Title</title>
        //   <ns>0</ns>  <!-- namespace: 0 = main articles -->
        //   <id>12345</id>
        //   <revision>
        //     <text>WikiText content here</text>
        //   </revision>
    }
}
```

#### 2.3 Namespace Filtering
Wikipedia namespaces to include by default:
- `0` - Main articles
- `100` - Portal (optional)
- `118` - Draft (optional)

Exclude:
- `1` - Talk
- `2` - User
- `6` - File
- `10` - Template
- `14` - Category

### Phase 3: ZIM Support

#### 3.1 ZIM Reader
Create `src/import/zim.rs`:
- Use `zim` crate or FFI to libzim
- Iterate over articles
- Extract HTML content
- Convert HTML → plaintext (reuse `src/scraping/extractor.rs`)

### Phase 4: CLI Integration

#### 4.1 Add CLI Commands to main.rs
```rust
#[derive(Subcommand)]
enum Commands {
    // ... existing commands ...

    /// Import content from offline dumps
    Import {
        /// Path to dump file
        #[arg(required = true)]
        path: PathBuf,

        /// Dump format (auto-detected if not specified)
        #[arg(short, long)]
        format: Option<DumpFormat>,

        /// Batch size for indexing
        #[arg(long, default_value = "100")]
        batch_size: usize,

        /// Resume from checkpoint
        #[arg(long)]
        resume: bool,

        /// Checkpoint file path
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        /// Skip content deduplication
        #[arg(long)]
        no_dedup: bool,
    },

    /// Show import progress/stats
    ImportStatus,
}

#[derive(Clone, Copy, ValueEnum)]
enum DumpFormat {
    WikimediaXml,
    Zim,
    Warc,
}
```

#### 4.2 Progress Reporting
Use `indicatif` for terminal progress:
```
Importing enwiki-latest-pages-articles.xml.bz2
[████████████████░░░░░░░░░░░░░░] 55% (3,245,123 / 5,900,000 articles)
Speed: 1,234 articles/sec | ETA: 35 minutes
Last: "History of computing"
```

### Phase 5: Integration with Existing Pipeline

#### 5.1 Connect to Embedding Pipeline
```rust
// In coordinator.rs
async fn process_batch(&mut self, docs: Vec<DumpDocument>) -> Result<(), ImportError> {
    // Convert to existing Document type
    let documents: Vec<Document> = docs.into_iter()
        .map(|d| Document {
            id: d.id,
            content: d.content,
            metadata: DocumentMetadata {
                title: Some(d.title),
                url: d.url,
                ..Default::default()
            },
        })
        .collect();

    // Chunk documents
    let chunks = self.chunker.chunk_documents(&documents);

    // Generate embeddings
    let embeddings = self.embedder.embed_batch(&chunks).await?;

    // Index
    self.index.add_batch(&chunks, &embeddings)?;

    Ok(())
}
```

#### 5.2 Deduplication Integration
- Reuse `src/scraping/dedup.rs` SimHash for content dedup
- Skip exact duplicates (same content hash)
- Flag near-duplicates (SimHash distance < threshold)

## Testing Strategy

### Unit Tests
- WikiText parser: test all markup patterns
- XML streaming: test with small sample dumps
- Checkpoint/resume: verify state persistence

### Integration Tests
```rust
#[test]
fn test_wikimedia_sample_import() {
    // Use a small sample dump (e.g., Simple English Wikipedia)
    let source = WikimediaSource::open("tests/fixtures/simplewiki-sample.xml.bz2")?;
    let mut coordinator = ImportCoordinator::new(config);
    let stats = coordinator.import(source).await?;
    assert!(stats.documents_imported > 0);
}
```

### Sample Data
- Create `tests/fixtures/` with small sample dumps
- Simple English Wikipedia dump (~200MB) for CI
- Synthetic test files for edge cases

## Configuration Extension

Add to `src/config.rs`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConfig {
    /// Default batch size
    pub batch_size: usize,
    /// Enable checkpointing
    pub enable_checkpoints: bool,
    /// Checkpoint directory
    pub checkpoint_dir: PathBuf,
    /// Default namespace filter for Wikipedia
    pub wikipedia_namespaces: Vec<i32>,
    /// Minimum content length to import
    pub min_content_length: usize,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            enable_checkpoints: true,
            checkpoint_dir: PathBuf::from(".dindex/checkpoints"),
            wikipedia_namespaces: vec![0], // Main namespace only
            min_content_length: 100,
        }
    }
}
```

## Error Handling

```rust
#[derive(Debug, Error)]
pub enum ImportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XML parse error: {0}")]
    XmlParse(#[from] quick_xml::Error),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Invalid dump format: {0}")]
    InvalidFormat(String),

    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    #[error("Index error: {0}")]
    Index(#[from] crate::index::IndexError),

    #[error("Embedding error: {0}")]
    Embedding(#[from] crate::embedding::EmbeddingError),
}
```

## Performance Considerations

1. **Streaming**: Never load full dump into memory
2. **Batch Processing**: Group documents for efficient embedding (batch size 64-128)
3. **Parallel Processing**: Consider `rayon` for CPU-bound WikiText parsing
4. **Async I/O**: Use `tokio::fs` for non-blocking file reads
5. **Memory Budget**: Target < 2GB RAM for import process

## Future Extensions

1. **Incremental Updates**: Import only changed articles using dump diffs
2. **Multi-file Support**: Handle split dumps (enwiki-*-pages-articles1.xml.bz2, etc.)
3. **Remote Streaming**: Stream directly from HTTP without full download
4. **Format Auto-detection**: Detect format from file extension/magic bytes
5. **WARC/CDX Support**: Import from Common Crawl archives

## References

- Wikimedia dump format: https://meta.wikimedia.org/wiki/Data_dumps
- MediaWiki XML schema: https://www.mediawiki.org/wiki/Help:Export
- ZIM format: https://wiki.openzim.org/wiki/ZIM_file_format
- WARC format: https://iipc.github.io/warc-specifications/
