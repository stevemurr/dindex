# Cache Layer Implementation Plan

## Overview

This document describes the implementation of a **query cache layer** for DIndex. The cache allows consumer nodes (nodes that query more than they index) to temporarily store chunks from query results, reducing load on producer nodes and improving query latency for repeated/similar queries.

## Design Goals

1. **Bounded resource usage**: Cache has a configurable max size, ensuring lightweight nodes stay lightweight
2. **Transparent caching**: Cached chunks participate in local search (both vector and BM25)
3. **Self-balancing**: Popular content naturally spreads across more caches
4. **Simple freshness model**: TTL-based expiry, no complex coordination
5. **Opt-in routing advertisement**: Cached content can optionally be advertised to peers

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Query Flow                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Query arrives                                                               │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      QueryCoordinator                                │    │
│  │  1. Check local primary index                                        │    │
│  │  2. Check cache index (chunks where now - cached_at < TTL)          │    │
│  │  3. Fan out to network (unless --local-only)                        │    │
│  │  4. Merge all results via RRF                                        │    │
│  │  5. Cache new chunks from network responses                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  Results returned to user                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Cache Storage Model

### What Gets Cached

When a query returns results from remote peers, we cache the **full chunk data**:

```rust
/// A cached chunk from a remote peer
pub struct CachedChunk {
    /// The chunk data (content, metadata, embedding)
    pub chunk: Chunk,
    /// The embedding vector for search
    pub embedding: Vec<f32>,
    /// Peer we received this from (origin authority)
    pub source_peer: PeerId,
    /// When we cached this chunk
    pub cached_at: DateTime<Utc>,
    /// Last time this chunk was accessed in a query
    pub last_accessed: DateTime<Utc>,
    /// Number of times accessed (for stats/smarter eviction)
    pub access_count: u32,
}
```

### Why Chunk-Level Caching

Caching at the chunk level (vs query-result level) provides:

1. **Reusability**: A cached chunk can match *new* queries, not just the original query
2. **Full search participation**: Cached chunks are searchable via both dense (vector) and BM25 (text)
3. **Finer-grained eviction**: LRU operates on individual chunks, not entire query results

## Configuration

Add a new `[cache]` section to `dindex.toml`:

```toml
[cache]
# Enable the query cache
enabled = true

# Maximum cache size (bytes). LRU eviction when exceeded.
# Examples: "100MB", "1GB", "500MB"
max_size = "500MB"

# Time-to-live for cached chunks. After this duration, chunks are
# considered stale and will be evicted or refreshed on next access.
# Examples: "1h", "24h", "7d"
ttl = "24h"

# Include cached content in routing advertisements (centroids/bloom filters).
# Allows other peers to route queries to you based on your cache contents.
include_in_routing = true

# Only advertise cached chunks after N accesses (reduces routing churn)
routing_access_threshold = 3

# Path to cache storage (defaults to {data_dir}/cache/)
# cache_dir = "/path/to/cache"
```

### Config Struct

```rust
/// Cache layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable caching of remote query results
    pub enabled: bool,
    /// Maximum cache size in bytes
    pub max_size: u64,
    /// TTL for cached chunks
    pub ttl_secs: u64,
    /// Include cache contents in routing advertisements
    pub include_in_routing: bool,
    /// Access count threshold before advertising a cached chunk
    pub routing_access_threshold: u32,
    /// Cache directory (optional override)
    pub cache_dir: Option<PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 500 * 1024 * 1024, // 500MB
            ttl_secs: 24 * 60 * 60,       // 24 hours
            include_in_routing: true,
            routing_access_threshold: 3,
            cache_dir: None,
        }
    }
}
```

## Implementation Components

### 1. CacheIndex (`src/cache/mod.rs`)

The main cache structure, holding cached chunks in an LRU-evicting store:

```rust
pub struct CacheIndex {
    /// Cached chunks by chunk_id
    chunks: RwLock<HashMap<ChunkId, CachedChunk>>,
    /// LRU order tracking (most recent at back)
    lru_order: RwLock<VecDeque<ChunkId>>,
    /// Current size in bytes
    current_size: AtomicU64,
    /// Vector index for cached chunks (separate from primary)
    vector_index: VectorIndex,
    /// BM25 index for cached chunks (separate from primary)
    bm25_index: Bm25Index,
    /// Configuration
    config: CacheConfig,
}

impl CacheIndex {
    /// Create new cache index
    pub fn new(config: &CacheConfig, data_dir: &Path) -> Result<Self>;
    
    /// Load existing cache from disk
    pub fn load(config: &CacheConfig, data_dir: &Path) -> Result<Self>;
    
    /// Save cache to disk
    pub fn save(&self) -> Result<()>;
    
    /// Insert chunks from a query response
    pub fn insert_from_response(&self, response: &QueryResponse) -> Result<usize>;
    
    /// Search the cache (returns valid, non-expired chunks)
    pub fn search(&self, query: &Query, embedding: &Embedding, top_k: usize) 
        -> Result<Vec<SearchResult>>;
    
    /// Get a chunk by ID (updates access time/count)
    pub fn get(&self, chunk_id: &ChunkId) -> Option<CachedChunk>;
    
    /// Evict expired chunks (called periodically)
    pub fn evict_expired(&self) -> usize;
    
    /// Evict LRU chunks until under max_size
    fn evict_lru(&self, target_size: u64);
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats;
    
    /// Get embeddings for routing advertisement (chunks above access threshold)
    pub fn advertisable_embeddings(&self) -> Vec<(ChunkId, Embedding)>;
    
    /// Clear all cached data
    pub fn clear(&self);
}
```

### 2. Integration with QueryCoordinator (`src/query/coordinator.rs`)

Modify `QueryCoordinator::execute()` to include cache in the search path:

```rust
impl QueryCoordinator {
    pub async fn execute(&self, query: &Query) -> Result<AggregatedResults> {
        let embedding = self.embed_query(&query.text)?;
        
        // 1. Search primary local index
        let mut all_results = Vec::new();
        if let Some(retriever) = &self.local_retriever {
            let local = retriever.search(query, Some(&embedding))?;
            all_results.push(("local".to_string(), local));
        }
        
        // 2. Search cache index (NEW)
        if let Some(cache) = &self.cache_index {
            let cached = cache.search(query, &embedding, query.top_k)?;
            if !cached.is_empty() {
                all_results.push(("cache".to_string(), cached));
            }
        }
        
        // 3. Fan out to network
        if let Some(network) = &self.network {
            let candidates = self.router.find_candidates(&embedding, None);
            let responses = network.query(&candidates, query, &embedding).await?;
            
            for response in &responses {
                all_results.push((
                    response.responder_peer.clone().unwrap_or_default(),
                    response.results.clone()
                ));
                
                // 4. Cache the response chunks (NEW)
                if let Some(cache) = &self.cache_index {
                    cache.insert_from_response(response)?;
                }
            }
        }
        
        // 5. Merge via RRF
        let final_results = self.merge_results(all_results, query.top_k)?;
        
        Ok(AggregatedResults { results: final_results, ... })
    }
}
```

### 3. Integration with Routing (`src/routing/router.rs`)

When building node advertisements, optionally include cached chunk embeddings:

```rust
impl AdvertisementBuilder {
    /// Include cached embeddings in the advertisement
    pub fn with_cache(mut self, cache: &CacheIndex) -> Self {
        if cache.config.include_in_routing {
            let cached_embeddings = cache.advertisable_embeddings();
            self.additional_embeddings.extend(cached_embeddings);
        }
        self
    }
}
```

### 4. Cache Warming Command

Add a CLI command to pre-populate the cache:

```bash
# Warm cache from a specific peer's content
dindex cache-from <peer_id> --topic "rust programming" --limit 1000

# Warm cache by running queries
dindex cache-warm --queries-file ./queries.txt
```

Implementation in `src/commands/cache.rs`:

```rust
pub async fn cache_from_peer(
    peer_id: &str,
    topic: Option<&str>,
    limit: usize,
) -> Result<()> {
    // 1. Connect to peer
    // 2. If topic specified, search for topic-related chunks
    // 3. Otherwise, request sample of peer's content
    // 4. Insert into cache
}

pub async fn cache_warm_queries(queries_file: &Path) -> Result<()> {
    // 1. Read queries from file
    // 2. Execute each query (which populates cache as side effect)
    // 3. Report stats
}
```

## File Structure

```
src/
├── cache/
│   ├── mod.rs           # CacheIndex, CachedChunk
│   ├── config.rs        # CacheConfig
│   ├── storage.rs       # Persistence (save/load cache to disk)
│   └── stats.rs         # CacheStats, metrics
├── config.rs            # Add CacheConfig to main Config
├── query/
│   └── coordinator.rs   # Integrate cache into query path
├── routing/
│   └── router.rs        # Include cache in advertisements
└── commands/
    └── cache.rs         # cache-from, cache-warm commands
```

## Eviction Strategy

### LRU with Size Limit

```rust
fn evict_lru(&self, target_size: u64) {
    let mut chunks = self.chunks.write();
    let mut lru = self.lru_order.write();
    
    while self.current_size.load(Ordering::Relaxed) > target_size {
        if let Some(chunk_id) = lru.pop_front() {
            if let Some(cached) = chunks.remove(&chunk_id) {
                let chunk_size = Self::estimate_size(&cached);
                self.current_size.fetch_sub(chunk_size, Ordering::Relaxed);
                
                // Remove from vector and BM25 indexes
                self.vector_index.remove(&chunk_id);
                self.bm25_index.remove(&chunk_id);
            }
        } else {
            break;
        }
    }
}
```

### TTL Expiry

```rust
fn evict_expired(&self) -> usize {
    let now = Utc::now();
    let ttl = Duration::seconds(self.config.ttl_secs as i64);
    let mut evicted = 0;
    
    let mut chunks = self.chunks.write();
    let expired: Vec<ChunkId> = chunks
        .iter()
        .filter(|(_, c)| now - c.cached_at > ttl)
        .map(|(id, _)| id.clone())
        .collect();
    
    for chunk_id in expired {
        if let Some(cached) = chunks.remove(&chunk_id) {
            let size = Self::estimate_size(&cached);
            self.current_size.fetch_sub(size, Ordering::Relaxed);
            self.vector_index.remove(&chunk_id);
            self.bm25_index.remove(&chunk_id);
            evicted += 1;
        }
    }
    
    evicted
}
```

### Background Eviction Task

Run periodic eviction in the daemon:

```rust
// In daemon startup
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 min
    loop {
        interval.tick().await;
        let evicted = cache_index.evict_expired();
        if evicted > 0 {
            info!("Evicted {} expired cache entries", evicted);
        }
    }
});
```

## Size Estimation

```rust
impl CacheIndex {
    fn estimate_size(cached: &CachedChunk) -> u64 {
        let mut size = 0u64;
        
        // Chunk content
        size += cached.chunk.content.len() as u64;
        
        // Embedding (4 bytes per f32)
        size += (cached.embedding.len() * 4) as u64;
        
        // Metadata (rough estimate)
        size += 500; // chunk_id, document_id, urls, etc.
        
        // CachedChunk overhead
        size += 100; // timestamps, access_count, source_peer
        
        size
    }
}
```

## Search Integration

Cached chunks participate in hybrid search:

```rust
impl CacheIndex {
    pub fn search(
        &self,
        query: &Query,
        embedding: &Embedding,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        // Filter to non-expired chunks
        let now = Utc::now();
        let ttl = Duration::seconds(self.config.ttl_secs as i64);
        
        // Dense search on cache vector index
        let dense_results = self.vector_index.search(embedding, top_k * 2)?;
        
        // BM25 search on cache text index  
        let bm25_results = self.bm25_index.search(&query.text, top_k * 2)?;
        
        // RRF fusion
        let fused = reciprocal_rank_fusion(&[dense_results, bm25_results], 60);
        
        // Filter expired and build results
        let chunks = self.chunks.read();
        let results: Vec<SearchResult> = fused
            .into_iter()
            .filter_map(|scored| {
                let cached = chunks.get(&scored.chunk_id)?;
                if now - cached.cached_at > ttl {
                    return None; // Expired
                }
                Some(SearchResult {
                    chunk: cached.chunk.clone(),
                    relevance_score: scored.score,
                    node_id: Some(cached.source_peer.to_string()),
                    matched_by: vec!["cache".to_string()],
                })
            })
            .take(top_k)
            .collect();
        
        // Update access times for returned chunks
        drop(chunks);
        for result in &results {
            self.touch(&result.chunk.metadata.chunk_id);
        }
        
        Ok(results)
    }
    
    fn touch(&self, chunk_id: &ChunkId) {
        if let Some(cached) = self.chunks.write().get_mut(chunk_id) {
            cached.last_accessed = Utc::now();
            cached.access_count += 1;
        }
        // Update LRU order
        let mut lru = self.lru_order.write();
        lru.retain(|id| id != chunk_id);
        lru.push_back(chunk_id.clone());
    }
}
```

## Persistence

Cache is persisted to disk for fast startup:

```rust
// src/cache/storage.rs

impl CacheIndex {
    pub fn save(&self) -> Result<()> {
        let cache_dir = self.cache_dir();
        std::fs::create_dir_all(&cache_dir)?;
        
        // Save chunk data
        let chunks_path = cache_dir.join("cache_chunks.json");
        let chunks = self.chunks.read();
        let data = serde_json::to_string(&*chunks)?;
        std::fs::write(&chunks_path, data)?;
        
        // Save vector index
        self.vector_index.save(&cache_dir.join("cache_vectors"))?;
        
        // Save BM25 index (tantivy handles its own persistence)
        self.bm25_index.commit()?;
        
        Ok(())
    }
    
    pub fn load(config: &CacheConfig, data_dir: &Path) -> Result<Self> {
        let cache_dir = config.cache_dir.clone()
            .unwrap_or_else(|| data_dir.join("cache"));
        
        if !cache_dir.exists() {
            return Self::new(config, data_dir);
        }
        
        // Load chunks
        let chunks_path = cache_dir.join("cache_chunks.json");
        let chunks: HashMap<ChunkId, CachedChunk> = if chunks_path.exists() {
            let data = std::fs::read_to_string(&chunks_path)?;
            serde_json::from_str(&data)?
        } else {
            HashMap::new()
        };
        
        // Rebuild LRU order from last_accessed times
        let mut lru_vec: Vec<_> = chunks.iter()
            .map(|(id, c)| (id.clone(), c.last_accessed))
            .collect();
        lru_vec.sort_by_key(|(_, t)| *t);
        let lru_order: VecDeque<_> = lru_vec.into_iter().map(|(id, _)| id).collect();
        
        // Load vector index
        let vector_index = VectorIndex::load(&cache_dir.join("cache_vectors"))?;
        
        // Load BM25 index
        let bm25_index = Bm25Index::open(&cache_dir.join("cache_bm25"))?;
        
        // Calculate current size
        let current_size: u64 = chunks.values()
            .map(Self::estimate_size)
            .sum();
        
        Ok(Self {
            chunks: RwLock::new(chunks),
            lru_order: RwLock::new(lru_order),
            current_size: AtomicU64::new(current_size),
            vector_index,
            bm25_index,
            config: config.clone(),
        })
    }
}
```

## CLI Commands

### Cache Status

```bash
$ dindex cache status

Cache Status:
  Enabled:     true
  Size:        127.3 MB / 500.0 MB (25.5%)
  Chunks:      1,847
  TTL:         24h
  
  Oldest:      2h 15m ago
  Newest:      3m ago
  
  Sources:
    peer_abc123:  847 chunks
    peer_def456:  623 chunks
    peer_ghi789:  377 chunks
  
  Routing:     Advertising 423 chunks (access_count >= 3)
```

### Cache Clear

```bash
$ dindex cache clear
Cleared 1,847 cached chunks (127.3 MB)

$ dindex cache clear --older-than 12h
Cleared 892 cached chunks older than 12 hours
```

### Cache From

```bash
$ dindex cache-from 12D3KooWAbCd... --topic "machine learning" --limit 500
Fetching chunks from peer 12D3KooWAbCd...
  Querying: "machine learning"
  Received: 50 results
  Cached:   47 new chunks (3 duplicates)
  
  Querying: "neural networks"  
  Received: 50 results
  Cached:   41 new chunks (9 duplicates)
  
  ...
  
Total: 423 chunks cached (12.4 MB)
```

## Implementation Order

### Phase 1: Core Cache Structure
1. Add `CacheConfig` to `src/config.rs`
2. Create `src/cache/mod.rs` with `CacheIndex`, `CachedChunk`
3. Implement basic insert/get/evict operations
4. Add size tracking and LRU eviction

### Phase 2: Search Integration
5. Create cache-specific vector and BM25 indexes
6. Implement `CacheIndex::search()`
7. Integrate cache search into `QueryCoordinator::execute()`
8. Cache chunks from network responses

### Phase 3: Persistence
9. Implement `save()` and `load()` for cache
10. Add cache loading to daemon startup
11. Add periodic save (every N minutes or on shutdown)

### Phase 4: Routing Integration
12. Add `advertisable_embeddings()` to CacheIndex
13. Modify `AdvertisementBuilder` to include cache
14. Add `include_in_routing` config logic

### Phase 5: CLI Commands
15. Add `cache status` command
16. Add `cache clear` command
17. Add `cache-from` command for warming

### Phase 6: Testing
18. Unit tests for CacheIndex
19. Integration tests for cache in query path
20. Test LRU eviction behavior
21. Test TTL expiry
22. Test persistence round-trip

## Open Questions / Future Work

1. **Deduplication with primary index**: If a cached chunk's source later gets indexed locally, should we detect and remove the cached copy?

2. **Cache coherence signals**: Should we listen for "content updated" messages from source peers and invalidate affected cache entries?

3. **Selective caching**: Should we add filters for what to cache (e.g., only cache chunks above a relevance threshold)?

4. **Cache sharing protocol**: Could nodes proactively share their cache contents with peers who are querying similar topics?

5. **Metrics/observability**: Add cache hit rate, eviction rate, and size metrics for monitoring.
