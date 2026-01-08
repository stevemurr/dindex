# Web Scraping Add-On for Decentralized LLM Search Index

## Executive Summary

This document specifies a **distributed, polite, LLM-optimized web scraping subsystem** that feeds content into the decentralized semantic search index. The architecture embraces the same principles as the parent system: Rust-native, CPU-only inference, tolerant of node churn, and designed for eventual consistency.

The key insight is that **scraping coordination in a P2P network is fundamentally different from centralized crawlers**. Rather than a single frontier queue, we use consistent hashing to partition domains across nodes, with gossip protocols for URL exchange and SimHash for content deduplication across the network.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCRAPING NODE                                        │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐   │
│  │   URL Frontier   │  │  Fetch Engine    │  │   Content Pipeline      │   │
│  │  (Per-domain     │  │  (reqwest +      │  │   (Readability →        │   │
│  │   priority heap) │  │  chromiumoxide)  │  │    Chunking →           │   │
│  │                  │  │                  │  │    Embedding)           │   │
│  └────────┬─────────┘  └────────┬─────────┘  └───────────┬─────────────┘   │
│           │                     │                        │                  │
│  ┌────────┴─────────────────────┴────────────────────────┴───────────────┐  │
│  │                     Politeness Controller                              │  │
│  │   robots.txt cache | Per-domain rate limits | 429 backoff             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                     Deduplication Layer                                 │  │
│  │   URL Bloom filter (local) | SimHash DHT (network-wide content)        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ────────────────┼────────────────
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                     P2P COORDINATION LAYER                                   │
│                                                                              │
│   Domain Assignment (consistent hashing on hostname)                         │
│   URL Exchange (batched gossip between nodes)                                │
│   SimHash Queries (DHT lookup for content deduplication)                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Domain Assignment via Consistent Hashing

The cornerstone of distributed crawling coordination is **assigning domains to nodes without centralized coordination**. Following the UbiCrawler architecture, we use consistent hashing on hostnames.

### Why Hostname-Based Partitioning

- **Politeness is local**: All crawl-delay and rate-limiting decisions for a domain stay on one node
- **robots.txt caching**: Each domain's robots.txt is cached exactly once
- **Reduces cross-node chatter**: Most extracted URLs stay local (same-site links)
- **Natural load distribution**: Domains are roughly uniformly distributed by hash

### Implementation

```rust
use std::collections::BTreeMap;
use blake3::hash;

struct DomainAssignment {
    ring: BTreeMap<u64, PeerId>,
    virtual_nodes: usize,  // 150 vnodes per physical node recommended
}

impl DomainAssignment {
    fn assign_domain(&self, hostname: &str) -> PeerId {
        let key = hash(hostname.as_bytes());
        let key_u64 = u64::from_be_bytes(key.as_bytes()[..8].try_into().unwrap());
        
        // Find first node >= key on the ring
        self.ring.range(key_u64..)
            .next()
            .or_else(|| self.ring.iter().next())  // Wrap around
            .map(|(_, peer)| *peer)
            .unwrap()
    }
    
    fn on_node_join(&mut self, peer: PeerId) {
        // Add virtual nodes to ring
        for i in 0..self.virtual_nodes {
            let vnode_key = hash(format!("{}:{}", peer, i).as_bytes());
            let key_u64 = u64::from_be_bytes(vnode_key.as_bytes()[..8].try_into().unwrap());
            self.ring.insert(key_u64, peer);
        }
        // Trigger reassignment of affected domains
    }
    
    fn on_node_leave(&mut self, peer: PeerId) {
        self.ring.retain(|_, p| *p != peer);
        // Domains reassign automatically to next node on ring
    }
}
```

### Handling Node Churn

When a node leaves (gracefully or crashed):

1. **Immediate**: URLs for affected domains route to next node on ring
2. **Background**: Orphaned in-progress crawls may produce duplicates (acceptable with dedup)
3. **Recovery**: If node returns quickly, it can resume its assigned domains

When a node joins:

1. **Gradual migration**: New node starts accepting its assigned domains
2. **No explicit handoff**: Previous owner naturally stops receiving new URLs for those domains
3. **Parallel crawling window**: Brief period where both nodes may crawl (dedup handles this)

---

## URL Frontier Design

Each node maintains a **per-domain priority queue** for URLs it's responsible for crawling.

### Data Structure

```rust
struct UrlFrontier {
    // Per-domain heaps, keyed by hostname
    domain_queues: HashMap<String, BinaryHeap<ScoredUrl>>,
    
    // Global seen-URL filter (probabilistic, allows some false positives)
    seen_urls: ScalableBloomFilter,
    
    // Per-domain last-crawl timestamps for politeness
    domain_last_fetch: HashMap<String, Instant>,
    
    // robots.txt cache
    robots_cache: LruCache<String, RobotsTxt>,
}

struct ScoredUrl {
    url: Url,
    priority: f32,      // Higher = crawl sooner
    depth: u8,          // Hops from seed
    discovered_at: Instant,
}

impl Ord for ScoredUrl {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then older discovery time
        self.priority.partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.discovered_at.cmp(&self.discovered_at))
    }
}
```

### Priority Scoring

URLs are prioritized by a combination of signals:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Depth from seed | -0.1 per hop | Breadth-first bias yields higher quality pages |
| Inlink count | +0.2 per inlink | More-linked pages are more important |
| Domain authority | +0.3 if known good | Prefer established sources |
| Freshness hint | +0.5 if news/blog | Time-sensitive content first |
| Content type | +0.2 for HTML | Skip binaries unless configured |

### URL Exchange Between Nodes

When a node extracts URLs pointing to domains assigned to other nodes, it batches them for exchange:

```rust
struct UrlExchangeBatch {
    from_node: PeerId,
    urls: Vec<(String, ScoredUrl)>,  // (hostname, url)
    timestamp: Instant,
}

// Batch and send every 5 seconds or 1000 URLs, whichever first
impl UrlFrontier {
    fn add_discovered_url(&mut self, url: Url) {
        let hostname = url.host_str().unwrap_or_default();
        let assigned_node = self.domain_assignment.assign_domain(hostname);
        
        if assigned_node == self.local_peer_id {
            self.add_local_url(url);
        } else {
            self.outbound_batches
                .entry(assigned_node)
                .or_default()
                .push(url);
        }
    }
}
```

---

## Fetch Engine

### Two-Tier Fetching Strategy

**Tier 1: Fast HTTP (reqwest)**
- First attempt for all URLs
- ~50ms latency, minimal resource usage
- Works for 70-80% of web pages

**Tier 2: Headless Browser (chromiumoxide)**
- Fallback for JS-rendered content
- ~2-5s latency, higher resource usage
- Triggered when Tier 1 produces insufficient content

### Recommended Rust Crates

| Component | Crate | Notes |
|-----------|-------|-------|
| HTTP client | `reqwest` | Async, connection pooling, HTTP/2 |
| Headless browser | `chromiumoxide` | CDP protocol, async, well-maintained |
| robots.txt | `texting_robots` | Battle-tested against 34M real robots.txt |
| HTML parsing | `scraper` + `lol_html` | DOM queries + streaming rewrite |
| URL normalization | `url` | Standard library quality |

### Headless Browser Detection Heuristics

Trigger browser rendering when:

```rust
fn needs_browser_rendering(response: &Response, body: &str) -> bool {
    // Very little text content
    let text_ratio = estimate_text_content_ratio(body);
    if text_ratio < 0.1 { return true; }
    
    // Heavy JS framework indicators
    let js_framework_hints = [
        "window.__NEXT_DATA__",     // Next.js
        "window.__NUXT__",          // Nuxt.js  
        "ng-app", "ng-controller",  // Angular
        "<div id=\"root\"></div>",  // React SPA
    ];
    if js_framework_hints.iter().any(|h| body.contains(h)) {
        return true;
    }
    
    // Explicit JS loading patterns
    if body.contains("Loading...") && body.len() < 5000 {
        return true;
    }
    
    false
}
```

### Connection Pooling Configuration

```rust
let client = reqwest::Client::builder()
    .pool_max_idle_per_host(10)
    .pool_idle_timeout(Duration::from_secs(90))
    .timeout(Duration::from_secs(30))
    .connect_timeout(Duration::from_secs(10))
    .user_agent("DecentralizedSearchBot/1.0 (+https://your-project.org/bot)")
    .build()?;
```

---

## Politeness Controller

### robots.txt Handling

Using `texting_robots` for robust parsing:

```rust
use texting_robots::{Robot, get_robots_url};

struct PolitenessController {
    robots_cache: LruCache<String, CachedRobots>,
    domain_state: HashMap<String, DomainState>,
}

struct CachedRobots {
    robot: Robot,
    fetched_at: Instant,
    crawl_delay: Option<Duration>,
}

struct DomainState {
    last_fetch: Instant,
    consecutive_429s: u32,
    backoff_until: Option<Instant>,
}

impl PolitenessController {
    async fn can_fetch(&mut self, url: &Url) -> FetchDecision {
        let hostname = url.host_str().unwrap_or_default();
        
        // Check robots.txt (fetch if not cached)
        let robots = self.get_or_fetch_robots(hostname).await;
        if !robots.robot.allowed(url.as_str()) {
            return FetchDecision::Disallowed;
        }
        
        // Check rate limiting
        let state = self.domain_state.entry(hostname.to_string()).or_default();
        
        if let Some(backoff_until) = state.backoff_until {
            if Instant::now() < backoff_until {
                return FetchDecision::RateLimited(backoff_until);
            }
        }
        
        let min_delay = robots.crawl_delay
            .unwrap_or(Duration::from_millis(1000));  // Default 1s
        
        let elapsed = state.last_fetch.elapsed();
        if elapsed < min_delay {
            return FetchDecision::WaitFor(min_delay - elapsed);
        }
        
        FetchDecision::Allowed
    }
    
    fn record_429(&mut self, hostname: &str, retry_after: Option<Duration>) {
        let state = self.domain_state.entry(hostname.to_string()).or_default();
        state.consecutive_429s += 1;
        
        // Exponential backoff: 30s, 60s, 120s, 240s, max 10min
        let backoff = retry_after.unwrap_or_else(|| {
            Duration::from_secs(30 * 2u64.pow(state.consecutive_429s.min(4)))
        });
        
        state.backoff_until = Some(Instant::now() + backoff);
    }
}
```

### Crawl Rate Guidelines

| Scenario | Delay | Rationale |
|----------|-------|-----------|
| No robots.txt | 1000ms | Conservative default |
| robots.txt Crawl-delay | As specified | Respect site preference |
| After 429 response | Retry-After or exponential | Required by HTTP spec |
| Small/personal sites | 2000ms minimum | Extra politeness |
| Large CDN-backed sites | 500ms acceptable | Can handle more load |

---

## Content Extraction Pipeline

### Stage 1: HTML → Clean Text

**Primary: Mozilla Readability (via `readability-js` crate)**

This wraps the actual Firefox Reader Mode algorithm via QuickJS, ensuring extraction quality matches what billions of users experience.

```rust
use readability_js::Readability;

struct ContentExtractor {
    reader: Readability,
}

impl ContentExtractor {
    fn extract(&self, html: &str, url: &str) -> Result<ExtractedContent> {
        let article = self.reader.parse_with_url(html, url)?;
        
        Ok(ExtractedContent {
            title: article.title,
            author: article.byline,
            published_date: article.published_time,
            clean_html: article.content,
            text_content: article.text_content,
            excerpt: article.excerpt,
        })
    }
}
```

**Fallback: fast_html2md for simpler pages**

When Readability fails (char_threshold not met), fall back to simpler HTML→Markdown conversion.

### Stage 2: Chunking for Embeddings

Following the main index architecture's chunking strategy:

```rust
struct ChunkingConfig {
    base_size: usize,           // 512 tokens
    overlap_ratio: f32,         // 0.15 (15%)
    min_chunk_size: usize,      // 100 tokens
    max_chunk_size: usize,      // 1024 tokens
}

struct ContentChunk {
    chunk_id: String,
    document_id: String,
    text: String,
    position_in_doc: f32,       // 0.0 to 1.0
    preceding_chunk_id: Option<String>,
    following_chunk_id: Option<String>,
}

fn chunk_content(text: &str, config: &ChunkingConfig) -> Vec<ContentChunk> {
    let tokens = tokenize(text);  // Use tiktoken-rs for accurate counts
    let overlap_tokens = (config.base_size as f32 * config.overlap_ratio) as usize;
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < tokens.len() {
        let end = (start + config.base_size).min(tokens.len());
        
        // Try to break at sentence boundary
        let adjusted_end = find_sentence_boundary(&tokens[start..end])
            .map(|b| start + b)
            .unwrap_or(end);
        
        let chunk_text = detokenize(&tokens[start..adjusted_end]);
        chunks.push(ContentChunk {
            chunk_id: generate_chunk_id(&chunk_text),
            text: chunk_text,
            position_in_doc: start as f32 / tokens.len() as f32,
            ..Default::default()
        });
        
        start = adjusted_end.saturating_sub(overlap_tokens);
    }
    
    // Link chunks together
    for i in 0..chunks.len() {
        if i > 0 {
            chunks[i].preceding_chunk_id = Some(chunks[i-1].chunk_id.clone());
        }
        if i < chunks.len() - 1 {
            chunks[i].following_chunk_id = Some(chunks[i+1].chunk_id.clone());
        }
    }
    
    chunks
}
```

### Stage 3: Embedding Generation

Using the same embedding infrastructure as the main index (nomic-embed or e5-small via ONNX Runtime):

```rust
struct EmbeddingPipeline {
    model: OrtSession,
    tokenizer: Tokenizer,
}

impl EmbeddingPipeline {
    async fn embed_chunks(&self, chunks: &[ContentChunk]) -> Vec<EmbeddedChunk> {
        // Batch for efficiency
        let batch_size = 32;
        let mut results = Vec::with_capacity(chunks.len());
        
        for batch in chunks.chunks(batch_size) {
            let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
            let embeddings = self.model.run_batch(&texts)?;
            
            for (chunk, embedding) in batch.iter().zip(embeddings) {
                results.push(EmbeddedChunk {
                    chunk: chunk.clone(),
                    embedding,
                    embedding_model: "nomic-embed-text-v1.5".to_string(),
                });
            }
        }
        
        results
    }
}
```

---

## Deduplication Strategy

### Level 1: URL Deduplication (Local Bloom Filter)

```rust
use probabilistic_collections::bloom::ScalableBloomFilter;

struct UrlDeduplicator {
    // Target 1% false positive rate
    filter: ScalableBloomFilter<String>,
}

impl UrlDeduplicator {
    fn is_new_url(&mut self, url: &Url) -> bool {
        let normalized = normalize_url(url);
        if self.filter.contains(&normalized) {
            false  // Already seen (or false positive)
        } else {
            self.filter.insert(&normalized);
            true
        }
    }
}

fn normalize_url(url: &Url) -> String {
    // Remove fragments, normalize case, sort query params
    let mut normalized = url.clone();
    normalized.set_fragment(None);
    
    // Sort query parameters for consistent hashing
    if let Some(query) = normalized.query() {
        let mut params: Vec<_> = query.split('&').collect();
        params.sort();
        normalized.set_query(Some(&params.join("&")));
    }
    
    normalized.as_str().to_lowercase()
}
```

### Level 2: Content Deduplication (SimHash via DHT)

For detecting near-duplicate content across different URLs and nodes:

```rust
use simhash::simhash;

struct ContentDeduplicator {
    // Local cache of recent SimHashes
    local_cache: LruCache<u64, DocumentId>,
    
    // DHT client for network-wide queries
    dht: KademliaClient,
}

impl ContentDeduplicator {
    fn compute_simhash(&self, text: &str) -> u64 {
        // Extract features (3-grams)
        let features: Vec<&str> = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .windows(3)
            .map(|w| w.join(" "))
            .collect();
        
        simhash(&features)
    }
    
    async fn is_duplicate(&mut self, text: &str) -> Option<DocumentId> {
        let hash = self.compute_simhash(text);
        
        // Check local cache first
        if let Some(doc_id) = self.local_cache.get(&hash) {
            return Some(doc_id.clone());
        }
        
        // Query DHT for similar hashes (Hamming distance ≤ 3)
        let similar = self.query_dht_near_duplicates(hash, 3).await;
        
        if let Some(existing) = similar.first() {
            return Some(existing.clone());
        }
        
        // Not a duplicate - register in DHT
        self.dht.put(hash, self.local_doc_id.clone()).await;
        self.local_cache.put(hash, self.local_doc_id.clone());
        
        None
    }
    
    async fn query_dht_near_duplicates(&self, hash: u64, max_distance: u32) -> Vec<DocumentId> {
        // Query all permutations with up to max_distance bit flips
        // For k=3, this is ~41,664 queries - batch and parallelize
        
        let mut candidates = Vec::new();
        
        // Simplified: query the exact hash and nearby
        if let Some(doc_id) = self.dht.get(hash).await {
            candidates.push(doc_id);
        }
        
        // In practice, use the table-based approach from Google's paper
        // Partition 64-bit hash into blocks, query by block prefix
        
        candidates
    }
}
```

### SimHash DHT Integration

Store SimHash → DocumentId mappings in the same Kademlia DHT used for search routing:

```
Key: SHA256(simhash_value) → routes to appropriate DHT nodes
Value: (document_id, source_node, timestamp)
```

Google's research shows that for 8B documents with 64-bit SimHash, Hamming distance k=3 achieves:
- **Precision**: ~75% (most flagged duplicates are true duplicates)  
- **Recall**: Sufficient for practical deduplication

---

## Metadata Extraction

Beyond main content, extract structured metadata for LLM consumption:

```rust
struct ExtractedMetadata {
    // Core identifiers
    url: String,
    canonical_url: Option<String>,
    
    // Content metadata
    title: String,
    description: Option<String>,
    author: Option<String>,
    published_date: Option<DateTime<Utc>>,
    modified_date: Option<DateTime<Utc>>,
    language: Option<String>,
    
    // Structural hints
    content_type: ContentType,  // Article, Product, Recipe, etc.
    word_count: usize,
    reading_time_minutes: u8,
    
    // Source signals
    domain: String,
    domain_authority: Option<f32>,  // If known from previous crawls
    
    // Scraped timestamps
    fetched_at: DateTime<Utc>,
    indexed_at: DateTime<Utc>,
}

fn extract_metadata(html: &str, url: &Url) -> ExtractedMetadata {
    let doc = Html::parse_document(html);
    
    // Priority: JSON-LD > OpenGraph > Twitter Cards > meta tags > heuristics
    let json_ld = extract_json_ld(&doc);
    let og = extract_opengraph(&doc);
    let twitter = extract_twitter_cards(&doc);
    let meta = extract_meta_tags(&doc);
    
    ExtractedMetadata {
        title: json_ld.title
            .or(og.title)
            .or(twitter.title)
            .or(meta.title)
            .or_else(|| extract_title_tag(&doc))
            .unwrap_or_default(),
        
        author: json_ld.author
            .or(meta.author)
            .or_else(|| extract_author_heuristic(&doc)),
        
        published_date: json_ld.date_published
            .or(meta.date)
            .or_else(|| extract_date_heuristic(&doc)),
        
        // ... etc
    }
}
```

---

## Integration with Semantic Index

After extraction, embedding, and deduplication, content flows into the main index:

```rust
struct IndexedDocument {
    // Identity
    document_id: String,
    source_url: String,
    
    // Content
    chunks: Vec<EmbeddedChunk>,
    
    // Metadata
    metadata: ExtractedMetadata,
    
    // Routing info
    content_centroid: Vec<f32>,  // Mean of chunk embeddings
    simhash: u64,
    
    // Provenance
    scraped_by: PeerId,
    scraped_at: DateTime<Utc>,
}

impl ScrapingNode {
    async fn process_and_index(&mut self, url: Url) -> Result<()> {
        // Fetch
        let html = self.fetch_engine.fetch(&url).await?;
        
        // Extract
        let content = self.extractor.extract(&html, url.as_str())?;
        
        // Deduplicate
        if let Some(existing) = self.dedup.is_duplicate(&content.text_content).await {
            log::info!("Skipping duplicate of {}", existing);
            return Ok(());
        }
        
        // Chunk & embed
        let chunks = chunk_content(&content.text_content, &self.chunk_config);
        let embedded = self.embedder.embed_chunks(&chunks).await?;
        
        // Compute centroid for routing
        let centroid = compute_centroid(&embedded);
        
        // Create document
        let doc = IndexedDocument {
            document_id: generate_doc_id(&url),
            source_url: url.to_string(),
            chunks: embedded,
            metadata: extract_metadata(&html, &url),
            content_centroid: centroid,
            simhash: self.dedup.compute_simhash(&content.text_content),
            scraped_by: self.local_peer_id,
            scraped_at: Utc::now(),
        };
        
        // Add to local vector index
        self.vector_index.add_document(&doc)?;
        
        // Update centroid advertisements
        self.update_routing_centroids().await;
        
        Ok(())
    }
}
```

---

## Configuration Recommendations

### Resource Budgets per Node

| Resource | Recommended | Notes |
|----------|-------------|-------|
| Concurrent fetches | 10-50 | Per node, across all domains |
| Browser instances | 2-4 | Only for JS rendering |
| URL frontier size | 1M URLs | ~50MB with efficient encoding |
| Bloom filter | 10M URLs | ~12MB at 1% FP rate |
| robots.txt cache | 10K domains | ~100MB |
| Embedding batch | 32 chunks | Optimal for CPU inference |

### Crawl Rate Targets

For a modest homelab node:

| Metric | Target | Notes |
|--------|--------|-------|
| Pages/hour | 1,000-5,000 | Depends on politeness delays |
| New content/day | 10K-50K pages | After deduplication |
| Domains/node | 100-1,000 | Depends on network size |

### Startup Sequence

1. **Bootstrap**: Connect to P2P network, join DHT
2. **Domain assignment**: Receive assigned domains via consistent hashing
3. **Seed URLs**: Load from configured seeds or receive from network
4. **robots.txt prefetch**: Fetch robots.txt for top-priority domains
5. **Begin crawling**: Start politeness-limited fetching

---

## Complete Crate Dependencies

```toml
[dependencies]
# Networking
reqwest = { version = "0.12", features = ["gzip", "brotli", "http2"] }
chromiumoxide = "0.7"
tokio = { version = "1", features = ["full"] }

# HTML Processing
scraper = "0.19"
lol_html = "1.2"
readability-js = "0.2"
fast_html2md = "0.1"

# robots.txt
texting_robots = "0.2"

# URL Handling
url = "2"

# Deduplication
probabilistic-collections = "0.5"
simhash = "0.1"

# Embedding (from main index)
ort = "2"
tokenizers = "0.19"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Hashing
blake3 = "1"

# Time
chrono = { version = "0.4", features = ["serde"] }

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"
```

---

## Summary

This scraping add-on integrates seamlessly with the decentralized semantic index by:

1. **Using the same P2P infrastructure** (libp2p, Kademlia DHT)
2. **Partitioning work via consistent hashing** (no central coordinator)
3. **Producing embeddings with the same models** (nomic-embed, ONNX Runtime)
4. **Storing content in the same vector index** (USearch)
5. **Leveraging DHT for network-wide deduplication** (SimHash queries)

The architecture prioritizes politeness, handles node churn gracefully, and produces LLM-optimized content with rich metadata for downstream RAG applications.
