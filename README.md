# DIndex

> **Experimental**: This project is a work in progress and not ready for production use.

**Decentralized Semantic Search Index for LLM Consumption**

A federated semantic search system designed for LLM consumption, featuring pluggable embedding backends, lightweight vector indices, P2P networking resilient to node churn, and intelligent semantic routing.

## Features

- **P2P Networking**: Built on rust-libp2p with Kademlia DHT, GossipSub, and QUIC transport
- **Vector Search**: USearch HNSW index with INT8 scalar quantization
- **Hybrid Retrieval**: Combines dense vector search + BM25 lexical search with RRF fusion
- **Semantic Routing**: Content centroids and LSH signatures for efficient query routing
- **Pluggable Embeddings**: HTTP backends (OpenAI, vLLM, Ollama, LM Studio)
- **HTTP API**: REST API server for programmatic access with optional auth and CORS
- **Metadata Filtering**: Category-based search with exact match and contains filters
- **Bulk Import**: Wikipedia dumps with resumable checkpointing (ZIM and WARC support planned)
- **Web Scraping**: Multi-URL crawling with depth control and domain restriction
- **LLM-Ready**: Rich metadata structure for retrieved chunks

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DECENTRALIZED NODE                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │  rust-libp2p    │  │ Embedding Engine │  │  Vector Index         │  │
│  │  - Kademlia DHT │  │ (HTTP backends)  │  │  (USearch HNSW)       │  │
│  │  - GossipSub    │  │ - OpenAI, vLLM   │  │  - INT8 quantized     │  │
│  │  - QUIC         │  │ - Ollama, etc.   │  │  - Memory-mapped      │  │
│  │  - AutoNAT      │  └────────┬─────────┘  │  - 50-200 centroids   │  │
│  └────────┬────────┘           │            └───────────┬───────────┘  │
│           │                    │                        │              │
│  ┌────────┴────────┐          │            ┌───────────┴───────────┐  │
│  │  HTTP API       │          │            │  Metadata Filtering   │  │
│  │  (Axum)         │          │            │  - Exact match        │  │
│  │  - REST /api/v1 │          │            │  - Contains           │  │
│  │  - Auth / CORS  │          │            │  - URL prefix         │  │
│  └────────┬────────┘          │            └───────────┬───────────┘  │
│           └───────────────────┼────────────────────────┘              │
│                               │                                        │
│  ┌────────────────────────────┼──────────────────────────────────────┐ │
│  │                     Hybrid Retrieval Engine                        │ │
│  │  Dense (HNSW) + BM25 → RRF Fusion                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/stevemurr/dindex
cd dindex

# Build
cargo build --release

# Install directly
cargo install --path .
```

## Quick Start (Docker Compose)

The fastest way to get running. Requires Docker and an NVIDIA GPU for the embedding server:

```bash
# Clone and start the stack (vLLM embeddings + DIndex)
git clone https://github.com/stevemurr/dindex
cd dindex
docker compose up -d

# Wait for services to be healthy (~2 min for vLLM to load the model)
docker compose logs -f

# Index documents via the HTTP API
curl -X POST http://localhost:8081/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "content": "Your document text here...",
      "title": "My Document",
      "metadata": {"category": "example"}
    }]
  }'

# Search
curl -s http://localhost:8081/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "top_k": 10}' | jq

# Stop
docker compose down
```

The API is available at `http://localhost:8081` and P2P at port `4001`.

## Quick Start (Local Binary)

```bash
# Initialize configuration
dindex init

# Index a document
dindex index ./document.txt --title "My Document"

# Search
dindex search "your query here" --top-k 10

# Start P2P node
dindex start --listen /ip4/0.0.0.0/udp/4001/quic-v1
```

## Usage

### Indexing Documents

```bash
# Index a single file
dindex index ./path/to/document.txt

# Index with metadata
dindex index ./paper.pdf --title "Research Paper" --url "https://example.com/paper"

# Index a directory
dindex index ./documents/
```

### Searching

```bash
# Basic search
dindex search "semantic search concepts"

# JSON output for LLM consumption
dindex search "semantic search" --format json --top-k 20

# Export results
dindex search "query" --format json > results.json
```

### Web Scraping

```bash
# Scrape a site with depth control
dindex scrape https://example.com --depth 2 --stay-on-domain

# Scrape multiple seeds with rate limiting
dindex scrape https://site1.com https://site2.com --max-pages 1000 --delay-ms 1000

# View scraping statistics
dindex scrape-stats
```

### Bulk Import

```bash
# Import Wikipedia dump (auto-detects format)
dindex import ./wiki.xml.bz2 --batch-size 100

# Import ZIM file (Kiwix) — coming soon
# dindex import ./file.zim --format zim

# Import WARC web archive — coming soon
# dindex import ./archive.warc --format warc

# Resume an interrupted import
dindex import ./wiki.xml.bz2 --resume --checkpoint ./checkpoint.json

# Check import progress
dindex import-status ./checkpoint.json
```

### P2P Network

```bash
# Start node (daemonizes by default)
dindex start

# Start in foreground
dindex start --foreground

# Start with custom listen address
dindex start --listen /ip4/0.0.0.0/udp/4001/quic-v1

# Connect to bootstrap peers
dindex start --bootstrap /ip4/1.2.3.4/udp/4001/quic-v1/p2p/QmPeerId

# Daemon management
dindex daemon status
dindex daemon stop
dindex daemon restart
```

### Statistics & Export

```bash
# Show index statistics
dindex stats

# Show document registry statistics
dindex registry-stats

# Export for LLM consumption
dindex export ./output.jsonl --format jsonl
```

## HTTP API

DIndex includes a REST API server for programmatic access. Enable it in your config:

```toml
[http]
enabled = true
listen_addr = "0.0.0.0:8080"
api_keys = []       # Empty = no auth required; add keys to require Bearer token
cors_enabled = true
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/status` | Daemon status |
| `GET` | `/api/v1/stats` | Index statistics |
| `POST` | `/api/v1/search` | Search with optional filters |
| `POST` | `/api/v1/index` | Index documents |
| `POST` | `/api/v1/index/commit` | Force commit pending writes |
| `POST` | `/api/v1/index/clear` | Clear all entries from the index |
| `DELETE` | `/api/v1/documents` | Delete documents by IDs |

### Search with Metadata Filtering

```bash
curl -X POST http://localhost:8081/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 10,
    "filters": {
      "source_url_prefix": "https://arxiv.org",
      "metadata_equals": {"source": "arxiv"},
      "metadata_contains": {"category": ["ml", "ai"]}
    }
  }'
```

- **`metadata_equals`**: All specified key-value pairs must match exactly
- **`metadata_contains`**: Value must appear in the metadata field (supports comma-separated values in stored metadata)

### Index Documents via API

```bash
curl -X POST http://localhost:8081/api/v1/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "content": "Document text here...",
      "title": "My Document",
      "url": "https://example.com/doc",
      "metadata": {"category": "tech", "author": "Jane"}
    }]
  }'
```

### Delete Documents

```bash
# Delete specific documents by ID
curl -X DELETE http://localhost:8081/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"document_ids": ["doc123", "doc456"]}'

# Clear the entire index
curl -X POST http://localhost:8081/api/v1/index/clear
```

## Embedding Backends

DIndex supports pluggable embedding backends. The HTTP backend is the default and works with any OpenAI-compatible API.

### HTTP Backend (Default)

Works with OpenAI, vLLM, Ollama, LM Studio, text-embeddings-inference, and more:

```toml
[embedding]
backend = "http"
endpoint = "http://localhost:8002/v1/embeddings"
model = "BAAI/bge-m3"
dimensions = 1024
timeout_secs = 30
max_batch_size = 100
# api_key = "sk-..."  # Or set OPENAI_API_KEY env var
```

**Provider examples:**

| Provider | Endpoint |
|----------|----------|
| OpenAI | `https://api.openai.com/v1/embeddings` |
| vLLM | `http://localhost:8002/v1/embeddings` |
| Ollama | `http://localhost:11434/v1/embeddings` |
| LM Studio | `http://localhost:1234/v1/embeddings` |

## Swift Client

A native Swift client library is available for iOS, macOS, and visionOS apps.

**Add to your `Package.swift`:**

```swift
dependencies: [
    .package(path: "../DIndexClient")  // or use a URL
]
```

**Usage:**

```swift
import DIndexClient

let client = DIndexClient(baseURL: URL(string: "http://localhost:8081")!)

// Search
let results = try await client.search(query: "machine learning", topK: 10)

// Search with filters
let filters = SearchFilters(
    metadataEquals: ["source": "arxiv"],
    metadataContains: ["category": ["ml"]]
)
let filtered = try await client.search(query: "transformers", topK: 5, filters: filters)

// Index a document
try await client.index(content: "Document text...", title: "My Doc", url: "https://example.com")

// Delete documents
try await client.deleteDocuments(ids: ["doc123", "doc456"])

// Clear the entire index
try await client.clearAll()

// Health check
let healthy = try await client.health()
```

## Docker

### Quick Start with Docker Compose

The default `docker-compose.yml` runs DIndex with a vLLM embedding server (requires NVIDIA GPU):

```bash
# Start the full stack (vLLM embeddings + DIndex)
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

This starts:
- **vLLM** serving BGE-M3 embeddings on port 8002 (GPU-accelerated)
- **DIndex** with HTTP API on port 8081 (mapped from container port 8080) and P2P on port 4001

### Building the Image

```bash
# Build the Docker image
docker build -t dindex .
```

### Standalone Docker Usage

```bash
# Initialize configuration
docker run --rm -v dindex-data:/data dindex init --data-dir /data

# Start P2P node
docker run -d --name dindex \
  -p 4001:4001/udp \
  -p 4001:4001/tcp \
  -p 8081:8080 \
  -v dindex-data:/data \
  dindex start --listen /ip4/0.0.0.0/udp/4001/quic-v1

# Search
docker run --rm -v dindex-data:/data dindex search "your query" --top-k 10

# Index local documents
docker run --rm \
  -v dindex-data:/data \
  -v ./documents:/documents:ro \
  dindex index /documents --title "My Documents"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level (trace, debug, info, warn, error) |
| `DINDEX_DATA_DIR` | `/data` | Data directory inside container |
| `HF_TOKEN` | — | HuggingFace token (for gated models) |

### Exposed Ports

| Port | Protocol | Description |
|------|----------|-------------|
| 4001 | UDP | P2P QUIC transport (primary) |
| 4001 | TCP | P2P TCP fallback |
| 8081 | TCP | HTTP API (mapped from container 8080) |

## Configuration

Create `dindex.toml`:

```toml
[node]
listen_addr = "/ip4/0.0.0.0/udp/0/quic-v1"
data_dir = ".dindex"
enable_mdns = true
replication_factor = 3
query_timeout_secs = 10

[embedding]
backend = "http"
endpoint = "http://localhost:8002/v1/embeddings"
model = "BAAI/bge-m3"
dimensions = 1024

[index]
hnsw_m = 16
hnsw_ef_construction = 200
hnsw_ef_search = 100
memory_mapped = true
max_capacity = 1000000

[chunking]
chunk_size = 512
overlap_fraction = 0.15
min_chunk_size = 50
max_chunk_size = 2048

[retrieval]
enable_dense = true
enable_bm25 = true
rrf_k = 60
candidate_count = 50
enable_reranking = true

[routing]
num_centroids = 100
lsh_bits = 128
lsh_num_hashes = 8
bloom_bits_per_item = 10
candidate_nodes = 5

[http]
enabled = true
listen_addr = "0.0.0.0:8080"
api_keys = []
cors_enabled = true
```

## Output Format for LLM Consumption

Retrieved chunks include rich metadata:

```json
{
  "chunk_id": "doc123_0",
  "document_id": "doc123",
  "content": "The chunk content...",
  "source_url": "https://example.com/doc",
  "source_title": "Document Title",
  "timestamp": "2025-01-08T00:00:00Z",
  "position_in_doc": 0.35,
  "section_hierarchy": ["Chapter 2", "Section 2.1"],
  "preceding_chunk_id": "doc123_prev",
  "following_chunk_id": "doc123_next",
  "relevance_score": 0.89,
  "matched_by": ["dense", "bm25"]
}
```

## Technical Details

### Hybrid Retrieval

DIndex uses three-way hybrid retrieval:
1. **Dense vectors** (BGE-M3/custom embeddings) for semantic similarity
2. **BM25** (Tantivy) for exact lexical matching
3. **RRF fusion** (k=60) to combine rankings without score calibration

### Semantic Routing

Each node advertises:
- **50-200 content centroids** (k-means clustered, truncated to 256 dims)
- **LSH signatures** (128-bit) for fast similarity estimation
- **Bloom filters** for negative filtering

Query routing:
1. Compute query embedding + LSH signature
2. Filter nodes via bloom filter
3. Rank by centroid similarity
4. Query top 3-5 candidate nodes in parallel

### Quantization

- **INT8 scalar quantization**: 4x compression, 99% recall retention
- **Matryoshka truncation**: 256 of 768 dims for routing (98%+ retention)
- Combined: up to 12x reduction in routing bandwidth

## License

MIT License - see LICENSE file for details.
