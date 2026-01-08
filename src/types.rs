//! Core types for the DIndex system

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a document
pub type DocumentId = String;

/// Unique identifier for a chunk
pub type ChunkId = String;

/// Unique identifier for a peer/node
pub type NodeId = String;

/// Embedding vector type
pub type Embedding = Vec<f32>;

/// Quantized embedding (INT8)
pub type QuantizedEmbedding = Vec<i8>;

/// LSH signature for fast similarity estimation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LshSignature {
    pub bits: Vec<u64>,
    pub num_bits: usize,
}

impl LshSignature {
    pub fn new(bits: Vec<u64>, num_bits: usize) -> Self {
        Self { bits, num_bits }
    }

    /// Calculate Hamming distance between two signatures
    pub fn hamming_distance(&self, other: &LshSignature) -> usize {
        assert_eq!(self.num_bits, other.num_bits);
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    /// Calculate similarity (1 - normalized hamming distance)
    pub fn similarity(&self, other: &LshSignature) -> f32 {
        1.0 - (self.hamming_distance(other) as f32 / self.num_bits as f32)
    }
}

/// Metadata for a document chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub source_url: Option<String>,
    pub source_title: Option<String>,
    pub timestamp: DateTime<Utc>,
    /// Position in document as fraction (0.0 to 1.0)
    pub position_in_doc: f32,
    /// Section hierarchy (e.g., ["Chapter 2", "Section 2.1"])
    pub section_hierarchy: Vec<String>,
    pub preceding_chunk_id: Option<ChunkId>,
    pub following_chunk_id: Option<ChunkId>,
    pub node_id: Option<NodeId>,
    /// Additional custom metadata
    pub extra: HashMap<String, String>,
}

impl ChunkMetadata {
    pub fn new(chunk_id: ChunkId, document_id: DocumentId) -> Self {
        Self {
            chunk_id,
            document_id,
            source_url: None,
            source_title: None,
            timestamp: Utc::now(),
            position_in_doc: 0.0,
            section_hierarchy: Vec::new(),
            preceding_chunk_id: None,
            following_chunk_id: None,
            node_id: None,
            extra: HashMap::new(),
        }
    }
}

/// A document chunk with its content and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub metadata: ChunkMetadata,
    pub content: String,
    /// Token count for this chunk
    pub token_count: usize,
}

/// An indexed chunk with embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedChunk {
    pub chunk: Chunk,
    /// Dense embedding vector
    pub embedding: Embedding,
    /// LSH signature for routing
    pub lsh_signature: Option<LshSignature>,
    /// Internal index key
    pub index_key: u64,
}

/// A search result from retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub relevance_score: f32,
    pub node_id: Option<NodeId>,
    /// Which retrieval methods matched (dense, sparse, bm25)
    pub matched_by: Vec<String>,
}

impl SearchResult {
    pub fn new(chunk: Chunk, relevance_score: f32) -> Self {
        Self {
            chunk,
            relevance_score,
            node_id: None,
            matched_by: Vec::new(),
        }
    }
}

/// Query request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub text: String,
    pub top_k: usize,
    pub filters: Option<QueryFilters>,
}

impl Query {
    pub fn new(text: impl Into<String>, top_k: usize) -> Self {
        Self {
            text: text.into(),
            top_k,
            filters: None,
        }
    }
}

/// Optional query filters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryFilters {
    pub source_url_prefix: Option<String>,
    pub min_timestamp: Option<DateTime<Utc>>,
    pub max_timestamp: Option<DateTime<Utc>>,
    pub document_ids: Option<Vec<DocumentId>>,
}

/// Node centroid for semantic routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCentroid {
    pub centroid_id: u32,
    /// Truncated embedding (e.g., 256 dimensions for Matryoshka)
    pub embedding: Vec<f32>,
    /// Number of chunks represented by this centroid
    pub chunk_count: usize,
}

/// Advertisement of a node's content for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAdvertisement {
    pub node_id: NodeId,
    pub centroids: Vec<NodeCentroid>,
    /// Bloom filter of LSH signatures
    pub lsh_bloom_filter: Option<Vec<u8>>,
    pub total_chunks: usize,
    pub last_updated: DateTime<Utc>,
}

/// Document to be indexed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: DocumentId,
    pub content: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            title: None,
            url: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }
}

/// Statistics about the local node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub index_size_bytes: u64,
    pub connected_peers: usize,
    pub uptime_seconds: u64,
}
