# Optimal stack for a decentralized LLM search index

**Building a federated semantic search system for LLM consumption requires a careful balance of CPU-efficient embeddings, lightweight vector indices, P2P networking resilient to node churn, and intelligent semantic routing.** The recommended architecture uses Rust with rust-libp2p for networking, USearch for embedded vector indexing, and a hybrid retrieval pipeline combining dense embeddings with sparse search—all designed for nodes without GPU access and tolerant of 5-10 second query latencies.

This report synthesizes current research across vector databases, embedding models, decentralized architectures, and RAG-optimized retrieval to provide specific, implementable recommendations for a future-looking system.

---

## The case for Rust with USearch and libp2p

The implementation stack should center on **Rust** as the primary language, paired with **rust-libp2p** for peer-to-peer networking and **USearch** as the embedded vector index. This combination offers the best performance characteristics for CPU-bound semantic search in a distributed environment.

Rust outperforms Go by **1.5-2x** in high-throughput scenarios due to zero-cost abstractions and absence of garbage collection pauses—critical when performing vector similarity computations. Major vector databases validate this choice: Qdrant, LanceDB, and Pinecone (after rewriting from C++) all use Rust. The rust-libp2p implementation is production-proven across Ethereum (Lighthouse), IPFS, and Polkadot, with continuous interoperability testing against go-libp2p.

For the embedded vector index, **USearch** is the clear winner for edge deployment. This single-header C++11 library with native Rust bindings offers:

- **Sub-megabyte footprint** versus multi-GB dependencies for alternatives
- **Memory-mapped file access** allowing indices to be served from disk without full RAM loading
- **10x faster** than FAISS at equivalent recall on benchmarks
- **User-defined metrics** and automatic type casting between f32, f16, and i8 representations
- Production usage by Google, ClickHouse, and DuckDB

**LanceDB** serves as an alternative for scenarios requiring columnar storage benefits, offering query over billions of vectors in under 100ms on a MacBook through its Lance format (100x faster than Parquet for vector queries).

The networking layer should use **libp2p's native QUIC transport**, which provides 20-40ms latency improvement over HTTP/2, zero-RTT connection establishment, and built-in NAT traversal—essential for nodes operating behind home routers. For serialization, **FlatBuffers** offers zero-copy access to embedding vectors (100x faster deserialization than Protocol Buffers), critical for high-throughput query handling.

---

## CPU-optimized embedding models achieve near-GPU quality

The constraint of CPU-only inference no longer significantly impacts retrieval quality. Modern small embedding models deliver excellent results with sub-100ms latency on commodity hardware.

**Recommended embedding models ranked by use case:**

| Model | Parameters | Dimensions | CPU Latency | Best For |
|-------|------------|------------|-------------|----------|
| **e5-small-v2** | 33M | 384 | ~16ms | Speed-critical deployments |
| **nomic-embed-text-v1.5** | 137M | 768 | ~105ms | Matryoshka (variable dimension) support |
| **bge-base-en-v1.5** | 109M | 768 | ~82ms | General-purpose with query prefixes |
| **BGE-M3** | 568M | 1024 | Higher | Multilingual and hybrid retrieval |

**Matryoshka embeddings** deserve special attention for P2P networks. These models are trained so that earlier dimensions contain more important information, allowing truncation without retraining. At just 33% of original dimensions (256 of 768), Matryoshka-trained models retain **98%+ of retrieval performance**. This enables a powerful optimization: nodes advertise content using compact 256-dimensional embeddings for routing, while storing full 768-dimensional embeddings for final retrieval.

For inference, **ONNX Runtime via the `ort` Rust crate** delivers 3-5x speedup over Python equivalents with 60-80% less memory usage. All major embedding models convert cleanly to ONNX, and INT8 quantization preserves 99%+ accuracy with additional 2-3x speedup on CPUs with VNNI instructions.

**Quantization strategy for embeddings:**

- **Scalar quantization (INT8)**: 4x compression with 99% performance retention—use as default
- **Binary quantization**: 32x compression with 92-95% retention—viable for high-dimensional (1024+) models like Cohere embed-v3
- **Matryoshka truncation**: 4-8x compression by reducing dimensions—combine with INT8 for up to 32x total reduction

---

## Semantic routing through content centroids replaces domain tags

Traditional federated search routes queries based on explicit metadata or domain classifications. A semantic-first system routes based on **learned representations of node content**, enabling discovery without manual categorization.

**Content centroid advertising architecture:**

Each node generates **50-200 cluster centroids** representing its indexed content using k-means clustering with cosine similarity. For efficient bandwidth usage, these centroids undergo Matryoshka truncation to 128-256 dimensions and INT8 quantization, resulting in approximately **6-25KB of routing metadata** per node versus 150-600KB for full-precision vectors.

The routing process works hierarchically:

1. **Coarse routing**: Query embedding compared against cached node centroids to identify candidate nodes (top 3-5)
2. **Fine routing**: Selected nodes perform local k-NN search across their full indices
3. **Result aggregation**: Reciprocal Rank Fusion combines results without score calibration

**Locality-Sensitive Hashing (LSH)** provides an additional optimization layer. LSH generates compact **64-256 bit signatures** from embeddings such that similar vectors map to similar signatures with high probability. Nodes advertise these signatures via the DHT, and query routers can quickly eliminate irrelevant nodes before computing full centroid similarities.

**Bloom filters** complement LSH for negative filtering. Each node maintains a Bloom filter of its content's LSH signatures (~10 bits per item for 1% false positive rate). Query routers check these filters first, eliminating irrelevant nodes before more expensive centroid comparison.

---

## Kademlia DHT handles node churn with proven patterns

IPFS demonstrates that Kademlia-based DHTs can operate reliably at **250,000+ nodes** with significant churn. The key parameters proven in production:

- **k-bucket size**: 20 (number of peers tracked per distance bucket)
- **Replication factor**: k=20 for provider records
- **Provider record TTL**: 48 hours
- **Republish interval**: Every 22 hours

For a semantic search index, this translates to:

**Replication strategy**: Store index shards across k=3-5 replicas for high availability without excessive storage overhead. With k=3 replicas and 80% individual node availability, system availability reaches **99.2%**; with k=5, it reaches **99.97%**.

**Consistency model**: Eventual consistency is acceptable for search indices—users tolerate slightly stale results far better than unavailable results. Index updates propagate asynchronously via gossip protocols, with periodic synchronization reconciling differences.

**Graceful degradation**: The system returns partial results when some nodes are unavailable. Quality estimation tracks which semantic regions have reduced coverage, allowing the LLM consumer to understand result completeness. Timeout-based queries (5-10 seconds given the latency tolerance) gather results from available nodes rather than waiting for stragglers.

**Hybrid architecture combining Kademlia with semantic overlay:**

```
Query → Semantic signature (LSH) → DHT lookup for candidate nodes
     → Centroid similarity filtering → Parallel queries to top-N nodes  
     → Local HNSW search per node → RRF aggregation → Reranking → Results
```

---

## Hybrid retrieval outperforms pure semantic search by 15-30%

While the system prioritizes vector-first retrieval, **hybrid search combining dense embeddings with sparse retrieval** consistently improves recall, particularly for exact terminology, acronyms, and domain-specific jargon.

**Three-way hybrid configuration (optimal):**

1. **Dense vectors**: E5 or nomic embeddings for semantic similarity
2. **SPLADE sparse vectors**: Neural learned sparse representations that expand and weight terms
3. **BM25 full-text**: Traditional lexical matching for exact phrases

This combination captures semantic meaning (dense), handles precise recall for specialized terms (SPLADE), and ensures exact match capability (BM25). Research shows three-way retrieval outperforms two-way by **15-30%** on diverse benchmarks.

**SPLADE for CPU deployment**: Use document-only expansion mode—expand documents at index time rather than queries at search time. This achieves ~12.7% improvement over BM25 with similar query latency.

**Result fusion via Reciprocal Rank Fusion (RRF)**:

```
score(d) = Σ 1/(k + rank_r(d)) for all rankers r, where k=60
```

RRF works on ranks rather than scores, requiring no calibration across heterogeneous nodes. It's robust to outlier scores and rewards consensus when multiple retrieval methods rank a document highly.

---

## Chunking and metadata structure for LLM consumption

How content is chunked and what metadata accompanies results dramatically impacts LLM utility.

**Late chunking preserves context**: Rather than chunking documents then embedding chunks independently, late chunking applies the transformer to the full document (up to 8192 tokens with models like jina-embeddings-v2), then mean-pools per chunk boundary. This preserves contextual dependencies and improves retrieval by **5-15%** over naive chunking.

**Optimal chunk parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base chunk size | 512 tokens | Balances context preservation with precision |
| Overlap | 15% (~75 tokens) | Empirically optimal on financial/legal benchmarks |
| Factoid queries | 128-256 tokens | Tighter chunks for precise answers |
| Complex reasoning | 1024 tokens | Broader context for multi-hop queries |

**Essential metadata structure for LLM consumption:**

```json
{
  "chunk_id": "unique-identifier",
  "document_id": "parent-doc-id", 
  "source_url": "https://...",
  "source_title": "Document Title",
  "timestamp": "2025-01-08T00:00:00Z",
  "position_in_doc": 0.35,
  "section_hierarchy": ["Chapter 2", "Section 2.1"],
  "preceding_chunk_id": "...",
  "following_chunk_id": "...",
  "relevance_score": 0.89,
  "node_id": "originating-peer"
}
```

**Cross-encoder reranking on CPU**: After aggregating results from multiple nodes, rerank the top 20-50 candidates using **ms-marco-MiniLM-L-6-v2** (22.7M parameters). This model offers the best efficiency/quality tradeoff for CPU deployment, running in ONNX Runtime with 2-3x speedup over PyTorch.

---

## Complete architecture recommendation

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
│  │  Dense (HNSW) + SPLADE (sparse) + BM25 → RRF Fusion               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    ──────────────┼──────────────
                                  │
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY COORDINATION                               │
│  1. Query embedding + LSH signature                                      │
│  2. DHT lookup → candidate nodes (via centroid similarity)               │
│  3. Parallel queries to top 3-5 nodes                                    │
│  4. Global RRF aggregation                                               │
│  5. Cross-encoder reranking (MiniLM-L-6)                                │
│  6. Return results with metadata for LLM consumption                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**Critical implementation parameters:**

| Component | Specification |
|-----------|---------------|
| Language | Rust |
| P2P networking | rust-libp2p with QUIC transport |
| Vector index | USearch with INT8 scalar quantization |
| Embedding model | nomic-embed-text-v1.5 (Matryoshka-capable) or e5-small-v2 |
| Inference runtime | ONNX Runtime via ort crate |
| Serialization | FlatBuffers for queries/embeddings, Protobuf for coordination |
| DHT replication | k=3-5 replicas per shard |
| Centroid advertising | 50-200 centroids per node, 256-dim truncated, INT8 |
| Chunk size | 512 tokens, 15% overlap, late chunking when possible |
| Result fusion | RRF with k=60 |
| Reranker | ms-marco-MiniLM-L-6-v2 via ONNX |

---

## Conclusion

A decentralized semantic search index optimized for LLM consumption is technically feasible today using mature, production-proven components. The key insight is that **CPU-only inference is no longer a significant limitation**—modern embedding models with INT8 quantization achieve near-parity with full-precision GPU inference while enabling deployment on commodity hardware.

The architecture should embrace eventual consistency and partial results rather than fighting for strong consistency across unreliable nodes. Semantic routing via content centroids and LSH signatures enables discovery without centralized indexing, while Kademlia DHT patterns handle node churn gracefully.

Three implementation choices will most impact success: selecting Rust for its performance in vector operations and alignment with the vector database ecosystem; using hybrid retrieval (dense + sparse + BM25) for robust recall across query types; and designing metadata structure from the start to maximize LLM utility of retrieved content.

The 5-10 second latency tolerance provides significant headroom—parallel queries across nodes, comprehensive reranking, and quality estimation can all fit within this budget while delivering substantially better results than speed-optimized single-source retrieval.