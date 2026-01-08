# PDF Support Plan

## Overview

Add PDF text extraction for both local files and remote URLs (e.g., `https://arxiv.org/pdf/2407.21075`).

## Current State

- `dindex index ./file.pdf` uses `std::fs::read_to_string()` → fails on binary
- Web scraping only accepts `text/html`, `application/xhtml`, `text/plain`
- No PDF parsing dependencies in Cargo.toml

## Requirements

1. **Local PDF**: `dindex index ./paper.pdf --title "Paper"`
2. **Remote PDF**: `dindex index https://arxiv.org/pdf/2407.21075`
3. **Scraping**: When crawler encounters PDF links, extract and index

---

## Recommended Approach

### Dependency: `pdf-extract`

```toml
# Cargo.toml
pdf-extract = "0.7"
```

Pure Rust, no system dependencies, extracts text from PDFs. Handles most academic papers well.

Alternative: `lopdf` + custom text extraction (more control, more work).

---

## Files to Create

### 1. `src/content/mod.rs` - Content extraction module

```rust
mod pdf;
mod text;

pub use pdf::PdfExtractor;
pub use text::TextExtractor;

/// Unified content extraction result
pub struct ExtractedDocument {
    pub content: String,
    pub title: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Detect content type and extract
pub fn extract_from_path(path: &Path) -> Result<ExtractedDocument>;
pub fn extract_from_url(url: &Url, bytes: &[u8], content_type: &str) -> Result<ExtractedDocument>;
```

### 2. `src/content/pdf.rs` - PDF extraction

```rust
use pdf_extract::extract_text;

pub struct PdfExtractor;

impl PdfExtractor {
    /// Extract text from PDF bytes
    pub fn extract(bytes: &[u8]) -> Result<ExtractedDocument> {
        let text = extract_text_from_mem(bytes)?;
        
        // Clean up extracted text
        let cleaned = Self::clean_text(&text);
        
        Ok(ExtractedDocument {
            content: cleaned,
            title: Self::extract_title(&text),
            metadata: HashMap::new(),
        })
    }
    
    /// Clean common PDF extraction artifacts
    fn clean_text(text: &str) -> String {
        text.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
    
    /// Try to extract title from first lines
    fn extract_title(text: &str) -> Option<String> {
        text.lines()
            .find(|l| l.len() > 10 && l.len() < 200)
            .map(|s| s.to_string())
    }
}
```

### 3. `src/content/text.rs` - Plain text extraction

```rust
pub struct TextExtractor;

impl TextExtractor {
    pub fn extract(content: String) -> ExtractedDocument {
        ExtractedDocument {
            content,
            title: None,
            metadata: HashMap::new(),
        }
    }
}
```

---

## Files to Modify

### 4. `src/main.rs` - Update `index_document()`

```rust
use crate::content::{extract_from_path, extract_from_url};

async fn index_document(
    config: Config,
    path: PathBuf,  // Can be file path OR URL
    title: Option<String>,
    url: Option<String>,
) -> Result<()> {
    // Detect if path is URL
    let extracted = if path.to_string_lossy().starts_with("http") {
        // Fetch and extract from URL
        let url = Url::parse(&path.to_string_lossy())?;
        let bytes = fetch_url_bytes(&url).await?;
        let content_type = detect_content_type(&bytes, &url);
        extract_from_url(&url, &bytes, &content_type)?
    } else if path.is_file() {
        // Extract from local file
        extract_from_path(&path)?
    } else if path.is_dir() {
        // Existing directory logic...
    };
    
    // Use extracted content
    let mut doc = Document::new(extracted.content);
    if let Some(t) = title.or(extracted.title) {
        doc = doc.with_title(t);
    }
    // ... rest of indexing pipeline
}

async fn fetch_url_bytes(url: &Url) -> Result<Vec<u8>> {
    let client = reqwest::Client::new();
    let resp = client.get(url.as_str()).send().await?;
    Ok(resp.bytes().await?.to_vec())
}
```

### 5. `src/scraping/fetcher.rs` - Accept PDF content type

```rust
// Update content type check (around line 170)
fn is_supported_content_type(content_type: &str) -> bool {
    content_type.contains("text/html")
        || content_type.contains("application/xhtml")
        || content_type.contains("text/plain")
        || content_type.contains("application/pdf")  // NEW
}

// Return bytes instead of text for binary types
pub enum FetchedContent {
    Html(String),
    Pdf(Vec<u8>),
    Text(String),
}
```

### 6. `src/scraping/coordinator.rs` - Route PDF content

```rust
// When processing fetched content
match fetched {
    FetchedContent::Html(html) => {
        // Existing HTML extraction
        let extracted = extractor.extract(&html, &url)?;
    }
    FetchedContent::Pdf(bytes) => {
        // NEW: PDF extraction
        let extracted = PdfExtractor::extract(&bytes)?;
    }
    FetchedContent::Text(text) => {
        // Plain text
        let extracted = TextExtractor::extract(text);
    }
}
```

### 7. `src/lib.rs` - Export content module

```rust
pub mod content;
```

---

## Implementation Phases

### Phase 1: Core PDF Extraction (1-2 hours)
- Add `pdf-extract` dependency
- Create `src/content/mod.rs`, `pdf.rs`, `text.rs`
- Unit test PDF extraction

### Phase 2: Local File Support (1 hour)
- Update `index_document()` in `src/main.rs`
- Detect file type by extension
- Route to appropriate extractor

### Phase 3: URL Support (1-2 hours)
- Add URL detection in `index_document()`
- Fetch bytes from URL
- Detect content type and extract

### Phase 4: Scraping Integration (1-2 hours)
- Update `fetcher.rs` to accept PDF content type
- Update `coordinator.rs` to route PDF content
- Test with arxiv URLs

---

## Testing

```bash
# Local PDF
dindex index ./paper.pdf --title "My Paper"

# Remote PDF
dindex index https://arxiv.org/pdf/2407.21075

# Search indexed content
dindex search "transformer architecture"
```

---

## Edge Cases to Handle

1. **Encrypted PDFs** - Skip with warning
2. **Image-only PDFs** (scanned docs) - No text, skip with warning  
3. **Very large PDFs** - Stream/chunk processing or size limit
4. **Malformed PDFs** - Graceful error handling
5. **PDF redirects** - Follow redirects (arxiv → actual PDF)

---

## File Summary

| File | Action |
|------|--------|
| `Cargo.toml` | Add `pdf-extract = "0.7"` |
| `src/content/mod.rs` | NEW - Content extraction dispatch |
| `src/content/pdf.rs` | NEW - PDF text extraction |
| `src/content/text.rs` | NEW - Plain text wrapper |
| `src/main.rs` | Modify `index_document()` |
| `src/scraping/fetcher.rs` | Accept PDF content-type |
| `src/scraping/coordinator.rs` | Route PDF to extractor |
| `src/lib.rs` | Export `content` module |
