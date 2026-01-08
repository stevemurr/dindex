# Unified Document Identity and Deduplication System

## Overview

Implement content-based document identity using SimHash, with persistent deduplication that works uniformly across all input sources (Wikipedia import, web scraping, local files).

## Key Design Decisions

1. **DocumentId = SimHash(normalized_content)** - 64-bit fingerprint as 16-char hex
2. **SHA256 for exact-match detection** - distinguish "same content" from "similar content"  
3. **Near-duplicate threshold: Hamming distance ≤ 3** - same document, update it
4. **URL is metadata** - stored for attribution, multiple URLs can map to same content

## Update Semantics

| Condition | Action |
|-----------|--------|
| No match | Insert new document, generate chunks, index |
| Exact SHA256 match | Update metadata only (timestamp, URL list) |
| SimHash distance ≤ 3 | Remove old chunks, re-chunk new content, update registry |

---

## Files to Create

### 1. `src/index/registry.rs` - Persistent Document Registry

Core dedup index that persists to `document_registry.json`:

```rust
pub struct DocumentEntry {
    pub content_id: ContentId,      // SimHash hex
    pub content_hash: ContentHash,  // SHA256 hex
    pub simhash: u64,               // Raw value for Hamming distance
    pub urls: HashSet<String>,      // All URLs mapping to this content
    pub title: Option<String>,
    pub first_seen: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub source_type: String,        // "wikipedia", "web", "local"
    pub source_ids: HashMap<String, String>,  // e.g., {"wikipedia_id": "12345"}
    pub chunk_ids: Vec<String>,
}

pub enum DuplicateCheckResult {
    New,
    ExactMatch { entry: DocumentEntry },
    NearDuplicate { entry: DocumentEntry, hamming_distance: u32 },
}

pub struct DocumentRegistry {
    entries: HashMap<String, DocumentEntry>,      // content_id -> entry
    simhash_index: HashMap<u64, String>,          // simhash -> content_id
    simhash_buckets: HashMap<u8, HashSet<String>>, // LSH buckets (top 8 bits)
    url_index: HashMap<String, String>,           // url -> content_id
}
```

### 2. `src/index/processor.rs` - Unified Document Processor

Single entry point for all document ingestion:

```rust
pub enum ProcessingResult {
    Indexed { content_id: String, chunks_created: usize },
    MetadataUpdated { content_id: String },
    ContentUpdated { content_id: String, chunks_created: usize, chunks_removed: usize },
    Skipped { content_id: String, reason: String },
}

pub struct DocumentProcessor {
    registry: Arc<DocumentRegistry>,
    indexer: Arc<HybridIndexer>,
    splitter: TextSplitter,
}

impl DocumentProcessor {
    pub fn process(
        &self,
        content: &str,
        url: Option<String>,
        title: Option<String>,
        source_type: &str,
        source_id: Option<(&str, &str)>,
        embedding_fn: impl Fn(&str) -> Embedding,
    ) -> Result<ProcessingResult>;
}
```

---

## Files to Modify

### 3. `src/types.rs` - Add Identity Types

```rust
/// Content-based document ID (SimHash as 16-char hex)
pub struct ContentId(pub String);

/// Exact content hash (SHA256 as 64-char hex)
pub struct ContentHash(pub String);

/// Combined identity for dedup decisions
pub struct DocumentIdentity {
    pub content_id: ContentId,
    pub content_hash: ContentHash,
    pub simhash: u64,
}

impl DocumentIdentity {
    pub fn compute(content: &str) -> Self {
        let normalized = content.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
        let simhash = SimHash::compute(&normalized).value();
        Self {
            content_id: ContentId::from_simhash(simhash),
            content_hash: ContentHash::compute(&normalized),
            simhash,
        }
    }
}
```

### 4. `src/retrieval/hybrid.rs` - Add Removal Methods

```rust
impl HybridIndexer {
    /// Remove a chunk from all indices
    pub fn remove_chunk(&self, chunk_id: &ChunkId) -> Result<()>;
    
    /// Remove all chunks for a document
    pub fn remove_document(&self, document_id: &str) -> Result<usize>;
}
```

### 5. `src/import/coordinator.rs` - Integrate DocumentProcessor

- Add `DocumentRegistry` and `DocumentProcessor` fields
- Replace direct chunking/indexing with `processor.process()`
- Save registry on completion

### 6. `src/scraping/coordinator.rs` - Integrate DocumentProcessor

- Use `DocumentProcessor` for scraped content
- Remove separate `ContentDeduplicator` usage (registry handles it)

### 7. `src/index/mod.rs` - Export New Modules

```rust
mod registry;
mod processor;
pub use registry::{DocumentRegistry, DocumentEntry, DuplicateCheckResult};
pub use processor::{DocumentProcessor, ProcessingResult};
```

### 8. `src/config.rs` - Add DedupConfig

```rust
pub struct DedupConfig {
    pub enabled: bool,
    pub simhash_distance_threshold: u32,  // default: 3
    pub normalize_content: bool,          // default: true
}
```

### 9. `src/main.rs` - Add Migration Command

```rust
Commands::MigrateRegistry => {
    // Rebuild registry from existing chunks.json
}
```

---

## Data Directory Layout

```
.dindex/
  document_registry.json  # NEW - persistent dedup index
  documents.json
  chunks.json
  vector.index
  vector.mappings.json
  bm25/
```

---

## Implementation Phases

### Phase 1: Core Types (1-2 hours)
- Add `ContentId`, `ContentHash`, `DocumentIdentity` to `src/types.rs`

### Phase 2: Document Registry (3-4 hours)
- Create `src/index/registry.rs` with persistence
- Unit tests for SimHash lookup, near-duplicate detection

### Phase 3: Document Processor (2-3 hours)
- Create `src/index/processor.rs`
- Add `remove_chunk`/`remove_document` to `HybridIndexer`

### Phase 4: Integration (3-4 hours)
- Update `ImportCoordinator`
- Update `ScrapingCoordinator`
- Update local file indexing in `main.rs`

### Phase 5: Migration & Testing (2-3 hours)
- Add `migrate-registry` CLI command
- Integration tests with Wikipedia re-import
- Test exact match vs near-duplicate behavior

---

## Testing Checklist

- [ ] Import same Wikipedia article twice → metadata update only
- [ ] Import article with minor edit → content update, re-chunk
- [ ] Import completely different article → new document
- [ ] Scrape URL, then import same content from Wikipedia → recognized as duplicate
- [ ] Registry persists across restarts
- [ ] Migration rebuilds registry from existing chunks
