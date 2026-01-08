//! DIndex: Decentralized Semantic Search Index for LLM Consumption
//!
//! A federated semantic search system optimized for LLM consumption, featuring:
//! - P2P networking via libp2p (Kademlia DHT, GossipSub, QUIC)
//! - CPU-optimized embeddings via ONNX Runtime
//! - Vector indexing with USearch (HNSW)
//! - Hybrid retrieval (Dense + BM25 + RRF fusion)
//! - Semantic routing via content centroids and LSH
//! - Distributed web scraping with polite crawling
//! - Bulk import from offline dumps (Wikipedia, etc.)

pub mod chunking;
pub mod config;
pub mod embedding;
pub mod import;
pub mod index;
pub mod network;
pub mod query;
pub mod retrieval;
pub mod routing;
pub mod scraping;
pub mod types;

pub use config::Config;
pub use types::*;
