# CLAUDE.md

## Project Overview

DIndex is a decentralized semantic search index for LLM consumption. It combines P2P networking (libp2p), pluggable HTTP embedding backends (OpenAI, vLLM, Ollama, etc.), and hybrid retrieval (dense vectors + BM25) into a federated search system.

## Build & Run

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run specific integration test
cargo test integration_test
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
├── main.rs              # CLI entry point
├── types.rs             # Core data types (Chunk, Query, SearchResult)
├── config.rs            # TOML configuration
├── embedding/           # HTTP embedding backends (OpenAI-compatible APIs)
├── index/               # USearch HNSW vector index
├── retrieval/           # Hybrid search (dense + BM25 + RRF fusion)
├── chunking/            # Document → chunks (512 tokens, 15% overlap)
├── network/             # libp2p (Kademlia DHT, GossipSub, QUIC)
├── routing/             # Semantic query routing (centroids, LSH, bloom filters)
├── scraping/            # Distributed web crawling
└── import/              # Bulk import (Wikimedia XML)
```

## Architecture Highlights

- **Hybrid Retrieval**: Dense vectors (HNSW) + BM25 lexical search, combined via Reciprocal Rank Fusion (k=60)
- **HTTP Embeddings**: Pluggable HTTP backends — works with any OpenAI-compatible API (vLLM, Ollama, LM Studio, etc.)
- **P2P Network**: Kademlia DHT for peer discovery, GossipSub for messaging, QUIC transport
- **Semantic Routing**: Nodes advertise content centroids + LSH signatures; queries route to relevant nodes via targeted delivery
- **Adaptive Fan-Out**: Query coordinator expands to additional peers when initial results are low quality
- **Scraping**: Multi-tier fetching (HTTP first, headless browser fallback), consistent hashing for domain assignment

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| libp2p | P2P networking |
| usearch | HNSW vector index |
| reqwest | HTTP client (embedding backends, scraping) |
| tantivy | BM25 full-text search |
| tokio | Async runtime |
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
