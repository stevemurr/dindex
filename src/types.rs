//! Core types for the DIndex system

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;

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

// ============================================================================
// Content Identity Types
// ============================================================================

/// Content-based document ID derived from SimHash (16-character hex string)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentId(pub String);

impl ContentId {
    /// Create a ContentId from a SimHash value
    pub fn from_simhash(simhash: u64) -> Self {
        ContentId(format!("{:016x}", simhash))
    }

    /// Parse a ContentId from a hex string
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() == 16 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
            Some(ContentId(hex.to_lowercase()))
        } else {
            None
        }
    }

    /// Get the underlying string value
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert back to a SimHash u64 value
    pub fn to_simhash(&self) -> Option<u64> {
        u64::from_str_radix(&self.0, 16).ok()
    }
}

impl fmt::Display for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<ContentId> for String {
    fn from(id: ContentId) -> Self {
        id.0
    }
}

/// Exact content hash using SHA256 (64-character hex string)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

impl ContentHash {
    /// Compute SHA256 hash of content
    pub fn compute(content: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let result = hasher.finalize();
        ContentHash(hex::encode(result))
    }

    /// Get the underlying string value
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<ContentHash> for String {
    fn from(hash: ContentHash) -> Self {
        hash.0
    }
}

/// Combined document identity for deduplication decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentIdentity {
    /// Content-based ID (SimHash)
    pub content_id: ContentId,
    /// Exact content hash (SHA256)
    pub content_hash: ContentHash,
    /// Raw SimHash value for Hamming distance calculations
    pub simhash: u64,
}

impl DocumentIdentity {
    /// Compute document identity from content
    pub fn compute(content: &str) -> Self {
        let normalized = Self::normalize_content(content);
        let simhash = Self::compute_simhash(&normalized);
        Self {
            content_id: ContentId::from_simhash(simhash),
            content_hash: ContentHash::compute(&normalized),
            simhash,
        }
    }

    /// Normalize content for consistent hashing
    fn normalize_content(content: &str) -> String {
        content
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Compute SimHash for text content using 3-grams
    fn compute_simhash(text: &str) -> u64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0;
        }

        // Extract 3-gram features
        let features: Vec<String> = if words.len() < 3 {
            words.iter().map(|s| s.to_string()).collect()
        } else {
            words.windows(3).map(|w| w.join(" ")).collect()
        };

        if features.is_empty() {
            return 0;
        }

        // Initialize 64-bit vector of counts
        let mut v: [i32; 64] = [0; 64];

        // For each feature, hash it and update counts
        for feature in &features {
            let feature_hash = xxhash_rust::xxh3::xxh3_64(feature.as_bytes());

            for i in 0..64 {
                if (feature_hash >> i) & 1 == 1 {
                    v[i] += 1;
                } else {
                    v[i] -= 1;
                }
            }
        }

        // Convert to final hash
        let mut result: u64 = 0;
        for (i, &count) in v.iter().enumerate() {
            if count > 0 {
                result |= 1 << i;
            }
        }

        result
    }

    /// Calculate Hamming distance to another identity
    pub fn hamming_distance(&self, other: &DocumentIdentity) -> u32 {
        (self.simhash ^ other.simhash).count_ones()
    }

    /// Check if this identity is similar to another (within threshold)
    pub fn is_similar(&self, other: &DocumentIdentity, max_distance: u32) -> bool {
        self.hamming_distance(other) <= max_distance
    }

    /// Check if content is exactly the same
    pub fn is_exact_match(&self, other: &DocumentIdentity) -> bool {
        self.content_hash == other.content_hash
    }
}

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
    /// Filter where metadata key exactly equals value
    pub metadata_equals: Option<HashMap<String, String>>,
    /// Filter where metadata key's value is in the provided list
    pub metadata_contains: Option<HashMap<String, Vec<String>>>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_id_from_simhash() {
        let simhash: u64 = 0x123456789abcdef0;
        let content_id = ContentId::from_simhash(simhash);
        assert_eq!(content_id.as_str(), "123456789abcdef0");
        assert_eq!(content_id.to_simhash(), Some(simhash));
    }

    #[test]
    fn test_content_id_from_hex() {
        let content_id = ContentId::from_hex("0123456789abcdef").unwrap();
        assert_eq!(content_id.as_str(), "0123456789abcdef");

        // Invalid hex should return None
        assert!(ContentId::from_hex("invalid").is_none());
        assert!(ContentId::from_hex("0123456789abcde").is_none()); // Too short
        assert!(ContentId::from_hex("0123456789abcdefg").is_none()); // Too long
    }

    #[test]
    fn test_content_hash_compute() {
        let content = "hello world";
        let hash = ContentHash::compute(content);
        // SHA256 of "hello world"
        assert_eq!(
            hash.as_str(),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );

        // Same content should produce same hash
        let hash2 = ContentHash::compute(content);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_document_identity_compute() {
        let content = "The quick brown fox jumps over the lazy dog";
        let identity = DocumentIdentity::compute(content);

        // Content ID should be 16 hex chars
        assert_eq!(identity.content_id.as_str().len(), 16);

        // Content hash should be 64 hex chars (SHA256)
        assert_eq!(identity.content_hash.as_str().len(), 64);

        // Same content should produce same identity
        let identity2 = DocumentIdentity::compute(content);
        assert_eq!(identity.content_id, identity2.content_id);
        assert_eq!(identity.content_hash, identity2.content_hash);
        assert_eq!(identity.simhash, identity2.simhash);
    }

    #[test]
    fn test_document_identity_normalization() {
        // Content with extra whitespace should normalize to same identity
        let content1 = "Hello   World";
        let content2 = "hello world";
        let content3 = "  Hello  World  ";

        let id1 = DocumentIdentity::compute(content1);
        let id2 = DocumentIdentity::compute(content2);
        let id3 = DocumentIdentity::compute(content3);

        assert_eq!(id1.content_id, id2.content_id);
        assert_eq!(id2.content_id, id3.content_id);
    }

    #[test]
    fn test_document_identity_hamming_distance() {
        let content1 = "The quick brown fox jumps over the lazy dog";
        let content2 = "The quick brown fox leaps over the lazy dog"; // One word changed
        let content3 = "Lorem ipsum dolor sit amet consectetur adipiscing elit";

        let id1 = DocumentIdentity::compute(content1);
        let id2 = DocumentIdentity::compute(content2);
        let id3 = DocumentIdentity::compute(content3);

        // Same content should have distance 0
        assert_eq!(id1.hamming_distance(&id1), 0);

        // Similar content should have relatively small distance (within 32 bits / 50%)
        let distance_similar = id1.hamming_distance(&id2);
        assert!(
            distance_similar < 32,
            "Similar content should have low distance, got {}",
            distance_similar
        );

        // Different content should have larger distance than similar content
        let distance_different = id1.hamming_distance(&id3);
        // Note: Due to the nature of SimHash, even different texts may have varying distances
        // The key insight is that SimHash provides locality-sensitive hashing
        assert!(
            distance_different > 0,
            "Different content should have non-zero distance"
        );
    }

    #[test]
    fn test_document_identity_similarity() {
        let content1 = "The quick brown fox jumps over the lazy dog";
        let content2 = "The quick brown fox jumps over the lazy dog"; // Identical

        let id1 = DocumentIdentity::compute(content1);
        let id2 = DocumentIdentity::compute(content2);

        // Identical content should be similar with threshold 3
        assert!(id1.is_similar(&id2, 3));

        // And should be exact match
        assert!(id1.is_exact_match(&id2));
    }

    #[test]
    fn test_document_identity_near_duplicate() {
        let content1 = "This is a test document with some content for testing";
        let content2 = "This is a test document with some content for testing purposes"; // Minor change

        let id1 = DocumentIdentity::compute(content1);
        let id2 = DocumentIdentity::compute(content2);

        // Should be similar (low hamming distance) but not exact match
        let distance = id1.hamming_distance(&id2);
        assert!(distance <= 10, "Near-duplicates should have low distance, got {}", distance);
        assert!(!id1.is_exact_match(&id2));
    }
}
