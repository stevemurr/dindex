# Configuration

DIndex is configured via `dindex.toml`. Run `dindex init` to generate a default config.

## Full Reference

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
timeout_secs = 30
max_batch_size = 100
# api_key = "sk-..."  # Or set OPENAI_API_KEY env var

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

## Section Reference

### `[node]`

| Field | Default | Description |
|-------|---------|-------------|
| `listen_addr` | `/ip4/0.0.0.0/udp/0/quic-v1` | libp2p multiaddr for P2P listening |
| `data_dir` | `.dindex` | Directory for index data, metadata, and state |
| `enable_mdns` | `true` | Auto-discover peers on the local network |
| `replication_factor` | `3` | Number of nodes that store each chunk |
| `query_timeout_secs` | `10` | Timeout for distributed queries |

### `[embedding]`

| Field | Default | Description |
|-------|---------|-------------|
| `backend` | `"http"` | Embedding backend type (currently `http` only) |
| `endpoint` | — | URL of the OpenAI-compatible embeddings API |
| `model` | — | Model name to pass to the API |
| `dimensions` | — | Output embedding dimensions |
| `timeout_secs` | `30` | HTTP request timeout |
| `max_batch_size` | `100` | Max texts per embedding request |
| `api_key` | — | API key (or set `OPENAI_API_KEY` env var) |

Compatible providers:

| Provider | Endpoint |
|----------|----------|
| OpenAI | `https://api.openai.com/v1/embeddings` |
| vLLM | `http://localhost:8002/v1/embeddings` |
| Ollama | `http://localhost:11434/v1/embeddings` |
| LM Studio | `http://localhost:1234/v1/embeddings` |

### `[index]`

| Field | Default | Description |
|-------|---------|-------------|
| `hnsw_m` | `16` | HNSW graph connectivity (higher = more accurate, more memory) |
| `hnsw_ef_construction` | `200` | Search depth during index building |
| `hnsw_ef_search` | `100` | Search depth during queries |
| `memory_mapped` | `true` | Use mmap for the vector index |
| `max_capacity` | `1000000` | Maximum number of vectors |

### `[chunking]`

| Field | Default | Description |
|-------|---------|-------------|
| `chunk_size` | `512` | Target chunk size in tokens |
| `overlap_fraction` | `0.15` | Overlap between adjacent chunks (15%) |
| `min_chunk_size` | `50` | Minimum tokens per chunk |
| `max_chunk_size` | `2048` | Maximum tokens per chunk |

### `[retrieval]`

| Field | Default | Description |
|-------|---------|-------------|
| `enable_dense` | `true` | Enable HNSW vector search |
| `enable_bm25` | `true` | Enable BM25 lexical search |
| `rrf_k` | `60` | Reciprocal Rank Fusion parameter |
| `candidate_count` | `50` | Number of candidates from each retriever |
| `enable_reranking` | `true` | Enable word-boundary reranking |

### `[routing]`

| Field | Default | Description |
|-------|---------|-------------|
| `num_centroids` | `100` | Number of content centroids per node |
| `lsh_bits` | `128` | LSH signature bit width |
| `lsh_num_hashes` | `8` | Number of LSH hash bands |
| `bloom_bits_per_item` | `10` | Bloom filter bits per indexed item |
| `candidate_nodes` | `5` | Max nodes to query per search |

### `[http]`

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Enable the HTTP API server |
| `listen_addr` | `0.0.0.0:8080` | Address and port for the HTTP server |
| `api_keys` | `[]` | API keys for Bearer auth (empty = no auth) |
| `cors_enabled` | `true` | Enable CORS headers |
