# DIndex

**Decentralized Semantic Search Index for LLM Consumption**

A federated semantic search system designed for LLM consumption, featuring CPU-efficient embeddings, lightweight vector indices, P2P networking resilient to node churn, and intelligent semantic routing.

## Features

- **P2P Networking**: Built on rust-libp2p with Kademlia DHT, GossipSub, and QUIC transport
- **Vector Search**: USearch HNSW index with INT8 scalar quantization
- **Hybrid Retrieval**: Combines dense vector search + BM25 lexical search with RRF fusion
- **Semantic Routing**: Content centroids and LSH signatures for efficient query routing
- **CPU-Optimized**: Designed for nodes without GPU access, using ONNX Runtime for inference
- **LLM-Ready**: Rich metadata structure for retrieved chunks

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DECENTRALIZED NODE                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐  │
│  │  rust-libp2p    │  │ Embedding Engine │  │  Vector Index         │  │
│  │  - Kademlia DHT │  │ (ort + ONNX)     │  │  (USearch HNSW)       │  │
│  │  - GossipSub    │  │ - nomic-embed    │  │  - INT8 quantized     │  │
│  │  - QUIC         │  │ - INT8 inference │  │  - Memory-mapped      │  │
│  │  - AutoNAT      │  └────────┬─────────┘  │  - 50-200 centroids   │  │
│  └────────┬────────┘           │            └───────────┬───────────┘  │
│           │                    │                        │              │
│           └────────────────────┼────────────────────────┘              │
│                                │                                        │
│  ┌─────────────────────────────┼─────────────────────────────────────┐ │
│  │                     Hybrid Retrieval Engine                        │ │
│  │  Dense (HNSW) + BM25 → RRF Fusion                                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dindex
cd dindex

# Build
cargo build --release

# Or install directly
cargo install --path .
```

## Quick Start

```bash
# Initialize configuration
dindex init

# Download embedding model
dindex download nomic-embed-text-v1.5

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

### P2P Network

```bash
# Start node with default settings
dindex start

# Start with custom listen address
dindex start --listen /ip4/0.0.0.0/udp/4001/quic-v1

# Connect to bootstrap peers
dindex start --bootstrap /ip4/1.2.3.4/udp/4001/quic-v1/p2p/QmPeerId
```

### Statistics & Export

```bash
# Show index statistics
dindex stats

# Export for LLM consumption
dindex export ./output.jsonl --format jsonl
```

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
model_name = "nomic-embed-text-v1.5"
dimensions = 768
truncated_dimensions = 256
max_sequence_length = 8192
quantize_int8 = true

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
max_chunk_size = 1024

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
```

## Supported Embedding Models

| Model | Parameters | Dimensions | CPU Latency | Best For |
|-------|------------|------------|-------------|----------|
| nomic-embed-text-v1.5 | 137M | 768 | ~105ms | Matryoshka support |
| e5-small-v2 | 33M | 384 | ~16ms | Speed-critical |
| bge-base-en-v1.5 | 109M | 768 | ~82ms | General-purpose |
| all-MiniLM-L6-v2 | 22.7M | 384 | ~10ms | Lightweight |

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
1. **Dense vectors** (nomic/e5 embeddings) for semantic similarity
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
