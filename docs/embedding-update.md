# Embedding Integration Plan

## Problem Statement

The codebase has a **fully implemented ONNX embedding engine** that is not being used. Instead, four different commands use hash-based "fake" embeddings that produce incompatible vectors:

| Command | Location | Embedding Source |
|---------|----------|------------------|
| `index` | `main.rs:467-474` | Hash-based (scheme A) |
| `search` | `main.rs:523-530` | Hash-based (scheme B) |
| `import` | `coordinator.rs:209-221` | Hash-based (scheme C) |
| `scrape` | `main.rs:853-860` | Hash-based (scheme A) |

**Result**: Vector similarity between documents indexed by different commands is essentially random. BM25 text search works, but semantic/vector search is broken.

## Current State

### What Already Exists (and works)

```
src/embedding/
├── mod.rs        # Module exports
├── engine.rs     # Full ONNX Runtime integration with batch processing
├── model.rs      # ModelManager with HuggingFace downloading
└── quantize.rs   # INT8/binary quantization utilities
```

**Features already implemented:**
- ONNX Runtime inference (CPU-optimized, multi-threaded)
- Batch embedding with `embed_batch()`
- Single text embedding with `embed()`
- Matryoshka dimension truncation (768 → 256)
- Model downloading from HuggingFace with progress bars
- Model registry with 4 pre-configured models
- INT8 quantization support
- Tokenizer integration

**The `dindex download` command already works:**
```bash
dindex download nomic-embed-text-v1.5  # Downloads model
```

### What's Broken

1. **Feature flag disabled**: `onnx` feature not in default features
2. **Engine not instantiated**: Commands don't create `EmbeddingEngine`
3. **Fake embeddings inline**: Hash-based code scattered across commands

## Implementation Plan

---

### Phase 1: Enable ONNX Feature Flag

**File**: `Cargo.toml`

```toml
# Before
[features]
default = []
onnx = ["ort"]

# After
[features]
default = ["onnx"]
onnx = ["ort"]
```

**Verification**: `cargo build --release` should compile with ONNX support.

---

### Phase 2: Create Shared Embedding Service

Create a helper to initialize the embedding engine consistently across commands.

**New file**: `src/embedding/service.rs`

```rust
use crate::config::Config;
use crate::embedding::{EmbeddingEngine, ModelManager};
use anyhow::{Context, Result};
use std::sync::Arc;

/// Initialize embedding engine from config
/// 
/// Downloads model if not present, then creates engine.
pub async fn init_embedding_engine(config: &Config) -> Result<Arc<EmbeddingEngine>> {
    let cache_dir = config.node.data_dir.join("models");
    let manager = ModelManager::new(&cache_dir)?;
    
    // Ensure model is downloaded
    let model_name = &config.embedding.model_name;
    if !manager.model_exists(model_name)? {
        eprintln!("Model '{}' not found. Downloading...", model_name);
        manager.download_model(model_name).await
            .context("Failed to download embedding model")?;
    }
    
    // Create embedding config with paths
    let embedding_config = manager.create_config(model_name).await?;
    
    // Create engine
    let engine = EmbeddingEngine::new(&embedding_config)
        .context("Failed to initialize embedding engine")?;
    
    Ok(Arc::new(engine))
}

/// Check if model exists without downloading
pub fn check_model_exists(config: &Config) -> Result<bool> {
    let cache_dir = config.node.data_dir.join("models");
    let manager = ModelManager::new(&cache_dir)?;
    manager.model_exists(&config.embedding.model_name)
}
```

**Update**: `src/embedding/mod.rs`
```rust
mod service;
pub use service::{init_embedding_engine, check_model_exists};
```

---

### Phase 3: Update Index Command

**File**: `src/main.rs` (around line 440-490)

**Before:**
```rust
// Generate dummy embeddings (in production, use embedding engine)
let chunks_with_embeddings: Vec<_> = chunks
    .into_iter()
    .map(|c| {
        let embedding: Vec<f32> = (0..config.embedding.dimensions)
            .map(|i| {
                let hash = xxhash_rust::xxh3::xxh3_64(c.content.as_bytes());
                ((hash.wrapping_add(i as u64) % 1000) as f32 / 500.0) - 1.0
            })
            .collect();
        (c, embedding)
    })
    .collect();
```

**After:**
```rust
// Initialize embedding engine
let engine = init_embedding_engine(&config).await
    .context("Failed to initialize embedding engine")?;

// Extract texts for batch embedding
let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();

// Generate real embeddings
let embeddings = engine.embed_batch(&texts)
    .context("Failed to generate embeddings")?;

// Pair chunks with embeddings
let chunks_with_embeddings: Vec<_> = chunks
    .into_iter()
    .zip(embeddings.into_iter())
    .collect();
```

---

### Phase 4: Update Search Command

**File**: `src/main.rs` (around line 520-540)

**Before:**
```rust
// Generate query embedding (dummy for now)
let query_embedding: Vec<f32> = (0..config.embedding.dimensions)
    .map(|i| {
        let hash = xxhash_rust::xxh3::xxh3_64(query_text.as_bytes());
        ((hash.wrapping_add(i as u64) % 1000) as f32 / 500.0) - 1.0
    })
    .collect();
```

**After:**
```rust
// Initialize embedding engine
let engine = init_embedding_engine(&config).await
    .context("Failed to initialize embedding engine")?;

// Generate query embedding
let query_embedding = engine.embed(&query_text)
    .context("Failed to embed query")?;
```

---

### Phase 5: Update Import Coordinator

**File**: `src/import/coordinator.rs`

#### 5.1 Add Engine to Struct

```rust
pub struct ImportCoordinator {
    config: ImportConfig,
    processor: DocumentProcessor,
    vector_index: Arc<VectorIndex>,
    embedding_engine: Arc<EmbeddingEngine>,  // ADD THIS
    embedding_dims: usize,
    data_dir: PathBuf,
    quiet: bool,
}
```

#### 5.2 Update Constructor

```rust
impl ImportCoordinator {
    pub fn new(
        config: ImportConfig,
        data_dir: impl AsRef<Path>,
        chunking_config: ChunkingConfig,
        index_config: IndexConfig,
        dedup_config: DedupConfig,
        embedding_engine: Arc<EmbeddingEngine>,  // ADD PARAMETER
    ) -> Result<Self, ImportError> {
        let embedding_dims = embedding_engine.dimensions();
        
        // ... existing initialization ...
        
        Ok(Self {
            config,
            processor,
            vector_index,
            embedding_engine,  // STORE IT
            embedding_dims,
            data_dir,
            quiet: false,
        })
    }
}
```

#### 5.3 Replace Fake Embedding Function

**Remove:**
```rust
/// Generate embedding for text (placeholder - uses hash-based pseudo-embedding)
fn generate_embedding(text: &str, dims: usize) -> Embedding {
    // ... hash-based code ...
}
```

**Replace usage in `import()` method:**
```rust
// Before
|text| Self::generate_embedding(text, embedding_dims),

// After  
|text| self.embedding_engine.embed(text).unwrap_or_else(|_| vec![0.0; self.embedding_dims]),
```

Or better, batch the embeddings for performance:
```rust
// Collect texts first, then batch embed
let texts: Vec<&str> = documents_batch.iter().map(|d| d.content.as_str()).collect();
let embeddings = self.embedding_engine.embed_batch(&texts)?;
```

#### 5.4 Update Builder

```rust
impl ImportCoordinatorBuilder {
    embedding_engine: Option<Arc<EmbeddingEngine>>,  // Add field
    
    pub fn with_embedding_engine(mut self, engine: Arc<EmbeddingEngine>) -> Self {
        self.embedding_engine = Some(engine);
        self
    }
    
    pub fn build(self) -> Result<ImportCoordinator, ImportError> {
        let engine = self.embedding_engine
            .ok_or_else(|| ImportError::Config("Embedding engine required".into()))?;
            
        ImportCoordinator::new(
            self.config,
            self.data_dir,
            self.chunking_config,
            self.index_config,
            self.dedup_config,
            engine,
        )
    }
}
```

---

### Phase 6: Update Scrape Command

**File**: `src/main.rs` (around line 850-865)

Same pattern as index command:

```rust
// Initialize embedding engine once at start
let engine = init_embedding_engine(&config).await?;

// In the scraping loop, use:
let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
let embeddings = engine.embed_batch(&texts)?;
```

---

### Phase 7: Update Main.rs Import Command Call Site

**File**: `src/main.rs` (around line 950-1010)

```rust
Commands::Import { source, ... } => {
    // Initialize embedding engine
    let engine = init_embedding_engine(&config).await
        .context("Failed to initialize embedding engine. Run 'dindex download' first.")?;
    
    let coordinator = ImportCoordinatorBuilder::new(&config.node.data_dir)
        .with_embedding_engine(engine)  // ADD THIS
        .with_batch_size(batch_size)
        // ... other options ...
        .build()?;
    
    // ... rest of import logic ...
}
```

---

### Phase 8: Add Model Check to Commands

Add early validation that model exists before starting long operations:

```rust
// At start of index/import/search/scrape commands
if !check_model_exists(&config)? {
    eprintln!("Error: Embedding model not found.");
    eprintln!("Run: dindex download {}", config.embedding.model_name);
    std::process::exit(1);
}
```

---

## Testing Plan

### Unit Tests

1. **Engine initialization**: Test `init_embedding_engine()` with valid config
2. **Embedding consistency**: Same text produces same embedding
3. **Batch vs single**: `embed_batch([text])` == `[embed(text)]`

### Integration Tests

```rust
#[tokio::test]
async fn test_index_and_search_use_same_embeddings() {
    // Index a document
    // Search for it
    // Verify it's found with high similarity
}

#[tokio::test]
async fn test_import_produces_searchable_content() {
    // Import Wikipedia entries
    // Search for known content
    // Verify results are from imported content
}
```

### Manual Testing

```bash
# 1. Download model
dindex download nomic-embed-text-v1.5

# 2. Index a document
dindex index ./test.pdf --title "Test"

# 3. Search for content from that document
dindex search "content from test pdf"
# Should return results from test.pdf

# 4. Import Wikipedia
dindex import ./wiki.xml.bz2 --max-docs 100

# 5. Search for Wikipedia content
dindex search "content from wikipedia"
# Should return results from Wikipedia
```

---

## File Changes Summary

### Modified Files

| File | Changes |
|------|---------|
| `Cargo.toml` | Enable `onnx` in default features |
| `src/embedding/mod.rs` | Export new service module |
| `src/main.rs` | Update index, search, scrape commands |
| `src/import/coordinator.rs` | Add engine field, remove fake embeddings |

### New Files

| File | Purpose |
|------|---------|
| `src/embedding/service.rs` | Shared engine initialization helper |

### Removed Code

| Location | Code Removed |
|----------|--------------|
| `main.rs:467-476` | Hash-based embedding in index |
| `main.rs:523-531` | Hash-based embedding in search |
| `main.rs:853-860` | Hash-based embedding in scrape |
| `coordinator.rs:209-223` | `generate_embedding()` function |

---

## Performance Considerations

### Batch Size

The embedding engine supports batching. For optimal performance:

- **Index/Import**: Batch 32-64 chunks at a time
- **Search**: Single query, no batching needed
- **Scrape**: Batch per-page chunks together

### Memory Usage

- Model loads ~500MB into RAM
- Keep engine as `Arc<EmbeddingEngine>` and reuse
- Don't create new engine per batch

### Threading

The engine uses `config.embedding.num_threads` (defaults to CPU cores). This is already configured correctly.

---

## Migration for Existing Indexes

**Important**: Existing indexes created with fake embeddings are **incompatible** with real embeddings.

Users must re-index their content:

```bash
# Remove old index
rm -rf ~/.local/share/dindex/vector.index
rm -rf ~/.local/share/dindex/bm25

# Re-download model (if needed)
dindex download nomic-embed-text-v1.5

# Re-index content
dindex index ./documents/
```

Add a warning in the release notes about this breaking change.

---

## Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 1: Feature flag | 5 min |
| Phase 2: Service helper | 30 min |
| Phase 3: Index command | 20 min |
| Phase 4: Search command | 15 min |
| Phase 5: Import coordinator | 1 hour |
| Phase 6: Scrape command | 20 min |
| Phase 7: Import call site | 15 min |
| Phase 8: Model checks | 15 min |
| Testing | 1 hour |
| **Total** | **~4 hours** |

The embedding engine is already built and tested. This is primarily a wiring exercise.
