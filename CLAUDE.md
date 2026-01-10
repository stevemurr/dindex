# CLAUDE.md

## Project Overview

DIndex is a decentralized semantic search index for LLM consumption. It combines P2P networking (libp2p), CPU/GPU-efficient embeddings (embed_anything + candle), and hybrid retrieval (dense vectors + BM25) into a federated search system.

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
dindex download bge-m3                         # Download embedding model (optional, auto-downloads)
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
├── embedding/           # Embedding inference via embed_anything (bge-m3, etc.)
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
- **GPU/CPU Flexible**: Uses embed_anything (candle backend) with CUDA/Metal support, falls back to CPU
- **P2P Network**: Kademlia DHT for peer discovery, GossipSub for messaging, QUIC transport
- **Semantic Routing**: Nodes advertise content centroids + LSH signatures; queries route to relevant nodes
- **Scraping**: Multi-tier fetching (HTTP first, headless browser fallback), consistent hashing for domain assignment

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| libp2p | P2P networking |
| usearch | HNSW vector index |
| embed_anything | Embedding inference (candle backend) |
| tantivy | BM25 full-text search |
| tokio | Async runtime |
| reqwest | HTTP client |
| quick-xml | XML streaming (Wikipedia import) |

## Docker

```bash
docker build -t dindex .
docker compose --profile init run --rm dindex-init  # Initialize
docker compose up -d                                 # Start node
```

## Configuration

Config lives in `dindex.toml`. Key settings:
- Embedding: `bge-m3` (default), 1024 dims, multilingual, Matryoshka support
- GPU: Auto-detected (CUDA/Metal), build with `--features cuda` or `--features metal`
- Index: HNSW M=16, EF=200/100 (construction/search)
- Chunking: 512 tokens, 15% overlap
- Routing: 100 centroids, 128-bit LSH, 5 candidate nodes
