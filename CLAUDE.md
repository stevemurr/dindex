# CLAUDE.md

## Project Overview

DIndex is a decentralized semantic search index for LLM consumption. It combines P2P networking (libp2p), pluggable HTTP embedding backends (OpenAI, vLLM, Ollama, etc.), and hybrid retrieval (dense vectors + BM25) into a federated search system.

## Build & Run

```bash
# Build
cargo build --release

# Run tests (477 unit + 23 integration + 1 doc = 501 total)
cargo test

# Run specific integration test
cargo test integration_test

# Run Docker cluster tests (requires Docker, ~2-3 min each, run sequentially)
cargo test --test p2p_cluster -- --ignored --nocapture
```

## Key Commands

```bash
dindex init                                    # Initialize config
dindex index ./documents/ --title "Docs"       # Index documents
dindex search "query" --format json --top-k 10 # Search
dindex start --listen /ip4/0.0.0.0/udp/4001/quic-v1  # Start P2P node
dindex import ./wiki.xml.bz2 --batch-size 100  # Import Wikipedia dump
dindex scrape https://example.com --depth 2    # Web scraping
dindex stats                                   # Show index stats
```

## Project Structure

```
src/
├── main.rs              # CLI entry point (clap)
├── lib.rs               # Library root, module declarations
├── types.rs             # Core data types (Chunk, Query, SearchResult, ScoreBreakdown)
├── util.rs              # Shared utilities (SimHash, URL normalization, hashing)
├── config/              # TOML configuration and validation
├── embedding/           # HTTP embedding backends (OpenAI-compatible APIs)
├── index/               # USearch HNSW vector index, IndexStack factory, sled chunk storage
│   └── stack.rs         # IndexStack: open(), open_read_only(), create() factories
├── retrieval/           # Hybrid search (dense + BM25 + RRF fusion)
│   ├── scoring.rs       # Composable ScoringStage pipeline, ScoreBreakdown provenance
│   └── bm25.rs          # Tantivy BM25 index (read-write + read-only modes)
├── chunking/            # Document → chunks (token-aware, sentence-granular)
│   └── tokenizer/       # Pluggable tokenizers (BPE via tiktoken-rs, heuristic)
├── network/             # libp2p (Kademlia DHT, GossipSub, QUIC)
├── routing/             # Semantic query routing (centroids, LSH, bloom filters)
├── query/               # Query coordination, score-based merge, adaptive fan-out
├── scraping/            # Web crawling (URL frontier, fetch engine, crawl trap detection)
├── import/              # Bulk import (Wikipedia XML streaming)
├── content/             # Content extraction (HTML, PDF, plain text)
├── daemon/              # Background daemon (IPC protocol, HTTP API, job management)
│   └── protocol.rs      # JSON-based IPC (length-prefixed, over Unix socket)
├── commands/            # CLI command implementations
└── client/              # IPC client library (Unix socket communication)
```

## Architecture Highlights

- **Hybrid Retrieval**: Dense vectors (HNSW) + BM25 lexical search, combined via Reciprocal Rank Fusion (k=60)
- **Composable Scoring**: Pluggable `ScoringStage` pipeline — aggregator demotion, overlap reranker with stop words, `ScoreBreakdown` provenance tracking
- **HTTP Embeddings**: Pluggable HTTP backends — works with any OpenAI-compatible API (vLLM, Ollama, LM Studio, etc.)
- **Token-Aware Chunking**: Sentence-granular splitting using BPE token counts (tiktoken-rs), tokenizer auto-resolved from model name
- **IPC Protocol**: Daemon-client communication uses JSON over Unix socket (length-prefixed). Storage and P2P use bincode.
- **Read-Only Index Mode**: CLI search fallback uses `IndexStack::open_read_only()` — opens BM25 without write lock, falls back to in-memory ChunkStorage if sled is locked by daemon
- **P2P Network**: Kademlia DHT for peer discovery, GossipSub for messaging, QUIC transport
- **Semantic Routing**: Nodes advertise content centroids + multi-band LSH (8x8-bit) signatures; queries route to relevant nodes via targeted delivery
- **Adaptive Fan-Out**: Query coordinator expands to additional peers when initial results are low quality
- **Scraping**: Multi-tier fetching (HTTP first, headless browser fallback), URL frontier dedup, crawl trap detection

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| libp2p | P2P networking (Kademlia, GossipSub, QUIC) |
| usearch | HNSW vector index |
| tantivy | BM25 full-text search |
| sled | Embedded database (chunk storage) |
| axum | HTTP API server |
| tiktoken-rs | BPE tokenization for chunking |
| reqwest | HTTP client (embedding backends, scraping) |
| tokio | Async runtime |
| serde_json | JSON serialization (IPC protocol) |
| bincode | Binary serialization (storage, P2P network) |
| quick-xml | XML streaming (Wikipedia import) |

## Docker

```bash
docker build -t dindex .
docker compose --profile init run --rm dindex-init  # Initialize
docker compose up -d                                 # Start node
```

## Configuration

Config lives in `dindex.toml`. Key settings:
- Embedding: HTTP backend (e.g., vLLM serving bge-m3), configured via `backend = "http"` + `endpoint`
- Index: HNSW M=16, EF=200/100 (construction/search)
- Chunking: 512 tokens, 15% overlap
- Routing: 100 centroids, 128-bit LSH, 5 candidate nodes
