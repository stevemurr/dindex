# DIndex

**Decentralized Semantic Search Index for LLM Consumption**

Search engines return links. DIndex returns answers — semantically relevant text chunks ready for LLM context windows, served from a federated P2P network with no central authority.

## Technical Highlights

- ~28k lines of Rust, 403 tests
- Hybrid retrieval: dense vectors (HNSW) + BM25 + Reciprocal Rank Fusion
- Multi-band LSH (8x8-bit) semantic routing with bloom filters
- Adaptive fan-out query coordination across peers
- INT8 scalar quantization (4x compression, 99% recall)
- Crawl trap detection, URL frontier dedup
- Decomposed module architecture with type-safe enums throughout

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                DIndex Node                   │
                    │                                              │
  Documents ──────► │  Chunk ──► Embed ──► Index ──► Retrieve     │
                    │  (512tok)  (HTTP)   (HNSW)   (Dense+BM25)   │
                    │                       │          │           │
                    │                       ▼          ▼           │
                    │                    Centroids   RRF Fusion    │
                    │                       │          │           │
                    └───────────────────────┼──────────┼───────────┘
                                            │          │
                    ┌───────────────────────┼──────────┼───────────┐
                    │           P2P Layer   │          │           │
                    │                       ▼          ▼           │
                    │  Kademlia DHT ◄── Semantic ──► Query        │
                    │  GossipSub        Routing    Coordinator    │
                    │  QUIC Transport   (LSH+Bloom) (Fan-out)     │
                    └─────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Rust |
| P2P Networking | libp2p (Kademlia, GossipSub, QUIC) |
| Vector Index | USearch (HNSW, INT8 quantized) |
| Lexical Search | Tantivy (BM25) |
| HTTP API | Axum |
| Async Runtime | Tokio |
| Metadata Store | sled |
| HTTP Client | reqwest |

## Features

- **Hybrid Retrieval** — Dense vectors + BM25 lexical search, combined via RRF (k=60) without score calibration
- **Semantic Routing** — Content centroids, multi-band LSH signatures, and bloom filters route queries to relevant nodes
- **Adaptive Fan-Out** — Query coordinator expands to additional peers when initial results are low quality
- **Pluggable Embeddings** — Any OpenAI-compatible HTTP API (vLLM, Ollama, LM Studio, etc.)
- **P2P Networking** — Kademlia DHT for peer discovery, GossipSub for messaging, QUIC transport
- **Vector Search** — USearch HNSW index with INT8 scalar quantization and memory-mapped storage
- **Metadata Filtering** — Category-based search with exact match, contains, and URL prefix filters
- **Web Scraping** — Multi-URL crawling with depth control, domain restriction, and headless browser fallback
- **Bulk Import** — Wikipedia XML dumps with resumable checkpointing

## Quick Start

### Local Binary

```bash
cargo install --path .
dindex init
dindex index ./documents/ --title "My Docs"
dindex search "your query" --top-k 10
dindex start  # launch P2P node
```

### Docker Compose

```bash
docker compose up -d        # starts vLLM + DIndex
# wait ~2 min for model load, then:
curl -s http://localhost:8081/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 10}' | jq
```

See [docs/USAGE.md](docs/USAGE.md) for the full CLI reference and [docs/DOCKER.md](docs/DOCKER.md) for Docker details.

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/status` | Daemon status |
| `GET` | `/api/v1/stats` | Index statistics |
| `POST` | `/api/v1/search` | Search with optional filters |
| `POST` | `/api/v1/index` | Index documents |
| `POST` | `/api/v1/index/commit` | Force commit pending writes |
| `POST` | `/api/v1/index/clear` | Clear all index entries |
| `DELETE` | `/api/v1/documents` | Delete documents by IDs |

See [docs/API.md](docs/API.md) for curl examples and filter documentation.

## Configuration

```toml
[embedding]
backend = "http"
endpoint = "http://localhost:8002/v1/embeddings"
model = "BAAI/bge-m3"
dimensions = 1024

[node]
listen_addr = "/ip4/0.0.0.0/udp/0/quic-v1"
data_dir = ".dindex"

[retrieval]
enable_dense = true
enable_bm25 = true
rrf_k = 60

[http]
enabled = true
listen_addr = "0.0.0.0:8080"
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full config reference.

## Project Structure

```
src/
├── main.rs          # CLI entry point (clap)
├── types.rs         # Core data types (Chunk, Query, SearchResult)
├── config/          # TOML configuration and validation
├── embedding/       # HTTP embedding backends (OpenAI-compatible)
├── index/           # USearch HNSW vector index + IndexStack factory
├── retrieval/       # Hybrid search (dense + BM25 + RRF fusion)
├── chunking/        # Document chunking (512 tokens, 15% overlap)
├── network/         # libp2p (Kademlia DHT, GossipSub, QUIC)
├── routing/         # Semantic query routing (centroids, LSH, bloom)
├── query/           # Query coordination and adaptive fan-out
├── scraping/        # Web crawling (URL frontier, fetch engine)
├── import/          # Bulk import (Wikipedia XML streaming)
├── content/         # Content extraction and processing
├── daemon/          # Background daemon management
├── commands/        # CLI command implementations
└── client/          # Swift client library
```

## Clients

- **Swift** (iOS, macOS, visionOS) — see [docs/SWIFT_CLIENT.md](docs/SWIFT_CLIENT.md)

## License

MIT License — see [LICENSE](LICENSE) for details.
