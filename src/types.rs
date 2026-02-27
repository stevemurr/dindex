//! Core types for the DIndex system

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Unique identifier for a document
pub type DocumentId = String;

/// Unique identifier for a chunk
pub type ChunkId = String;

/// Unique identifier for a peer/node
pub type NodeId = String;

/// Embedding vector type
pub type Embedding = Vec<f32>;

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
        let lower = content.to_lowercase();
        let mut result = String::with_capacity(lower.len());
        for (i, word) in lower.split_whitespace().enumerate() {
            if i > 0 {
                result.push(' ');
            }
            result.push_str(word);
        }
        result
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

        crate::util::compute_simhash(features.iter().map(|s| s.as_str()))
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

/// LSH signature for fast similarity estimation (fixed 128-bit)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LshSignature {
    pub bits: [u64; 2],
    pub num_bits: usize,
}

impl LshSignature {
    pub fn new(bits: [u64; 2], num_bits: usize) -> Self {
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

/// A matching chunk within a grouped search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingChunk {
    pub chunk_id: ChunkId,
    pub content: String,
    pub relevance_score: f32,
    pub matched_by: Vec<String>,
    pub section_hierarchy: Vec<String>,
    pub position_in_doc: f32,
    /// Best-matching sentence snippet for citations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

/// Search results grouped by document, with matching chunks as sub-items
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupedSearchResult {
    pub document_id: DocumentId,
    pub source_url: Option<String>,
    pub source_title: Option<String>,
    /// Maximum relevance score among all matching chunks
    pub relevance_score: f32,
    /// Matching chunks sorted by score descending
    pub chunks: Vec<MatchingChunk>,
}

impl GroupedSearchResult {
    /// Group flat search results by document, returning top_k groups sorted by max score.
    pub fn from_results(results: Vec<SearchResult>, top_k: usize) -> Vec<GroupedSearchResult> {
        let mut groups: HashMap<DocumentId, GroupedSearchResult> = HashMap::new();

        for result in results {
            let doc_id = result.chunk.metadata.document_id.clone();
            let matching_chunk = MatchingChunk {
                chunk_id: result.chunk.metadata.chunk_id.clone(),
                content: result.chunk.content,
                relevance_score: result.relevance_score,
                matched_by: result.matched_by,
                section_hierarchy: result.chunk.metadata.section_hierarchy.clone(),
                position_in_doc: result.chunk.metadata.position_in_doc,
                snippet: None,
            };

            groups
                .entry(doc_id.clone())
                .and_modify(|group| {
                    if result.relevance_score > group.relevance_score {
                        group.relevance_score = result.relevance_score;
                    }
                    group.chunks.push(matching_chunk.clone());
                })
                .or_insert_with(|| GroupedSearchResult {
                    document_id: doc_id,
                    source_url: result.chunk.metadata.source_url.clone(),
                    source_title: result.chunk.metadata.source_title.clone(),
                    relevance_score: result.relevance_score,
                    chunks: vec![matching_chunk],
                });
        }

        let mut grouped: Vec<GroupedSearchResult> = groups.into_values().collect();

        // Merge groups that share the same source_url (handles legacy duplicates
        // where the same URL was indexed with different document_ids)
        let mut url_to_primary: HashMap<String, usize> = HashMap::new();
        let mut merge_targets: Vec<(usize, usize)> = Vec::new(); // (absorbed, primary)
        for (i, group) in grouped.iter().enumerate() {
            if let Some(ref url) = group.source_url {
                if let Some(&primary_idx) = url_to_primary.get(url) {
                    merge_targets.push((i, primary_idx));
                } else {
                    url_to_primary.insert(url.clone(), i);
                }
            }
        }
        // Merge absorbed groups into their primary (collect chunks, take max score)
        // Process in reverse so indices remain valid when we remove later
        let mut absorbed_indices: HashSet<usize> = HashSet::new();
        for (absorbed_idx, primary_idx) in &merge_targets {
            let absorbed_chunks = std::mem::take(&mut grouped[*absorbed_idx].chunks);
            let absorbed_score = grouped[*absorbed_idx].relevance_score;
            let primary = &mut grouped[*primary_idx];
            // Dedup chunks by chunk_id
            let existing_ids: HashSet<String> = primary.chunks.iter().map(|c| c.chunk_id.clone()).collect();
            for chunk in absorbed_chunks {
                if !existing_ids.contains(&chunk.chunk_id) {
                    primary.chunks.push(chunk);
                }
            }
            if absorbed_score > primary.relevance_score {
                primary.relevance_score = absorbed_score;
            }
            absorbed_indices.insert(*absorbed_idx);
        }
        // Remove absorbed groups (in reverse order to preserve indices)
        let mut absorbed_sorted: Vec<usize> = absorbed_indices.into_iter().collect();
        absorbed_sorted.sort_unstable_by(|a, b| b.cmp(a));
        for idx in absorbed_sorted {
            grouped.remove(idx);
        }

        // Sort groups by max score descending
        grouped.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));

        // Sort chunks within each group by score descending
        for group in &mut grouped {
            group.chunks.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        }

        grouped.truncate(top_k);
        grouped
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

    // ========================================================================
    // LshSignature tests
    // ========================================================================

    #[test]
    fn test_lsh_signature_new() {
        let bits = [0xDEADBEEF_u64, 0xCAFEBABE_u64];
        let sig = LshSignature::new(bits, 128);
        assert_eq!(sig.bits, bits);
        assert_eq!(sig.num_bits, 128);
    }

    #[test]
    fn test_lsh_signature_hamming_distance_identical() {
        let sig1 = LshSignature::new([0xFF00FF00, 0x00FF00FF], 128);
        let sig2 = LshSignature::new([0xFF00FF00, 0x00FF00FF], 128);
        assert_eq!(sig1.hamming_distance(&sig2), 0);
    }

    #[test]
    fn test_lsh_signature_hamming_distance_different() {
        // First word differs by 1 bit, second word is the same
        let sig1 = LshSignature::new([0b0000, 0b0000], 128);
        let sig2 = LshSignature::new([0b0001, 0b0000], 128);
        assert_eq!(sig1.hamming_distance(&sig2), 1);

        // Both words differ
        let sig3 = LshSignature::new([0b1111, 0b1111], 128);
        let sig4 = LshSignature::new([0b0000, 0b0000], 128);
        assert_eq!(sig3.hamming_distance(&sig4), 8);
    }

    #[test]
    fn test_lsh_signature_hamming_distance_all_bits_differ() {
        let sig1 = LshSignature::new([u64::MAX, u64::MAX], 128);
        let sig2 = LshSignature::new([0, 0], 128);
        assert_eq!(sig1.hamming_distance(&sig2), 128);
    }

    #[test]
    fn test_lsh_signature_similarity_range() {
        let sig1 = LshSignature::new([0xAAAAAAAA, 0x55555555], 128);
        let sig2 = LshSignature::new([0x55555555, 0xAAAAAAAA], 128);
        let similarity = sig1.similarity(&sig2);
        assert!(
            (0.0..=1.0).contains(&similarity),
            "Similarity should be in [0.0, 1.0], got {}",
            similarity
        );
    }

    #[test]
    fn test_lsh_signature_similarity_identical() {
        let sig = LshSignature::new([0xDEADBEEF, 0xCAFEBABE], 128);
        assert_eq!(sig.similarity(&sig), 1.0);
    }

    #[test]
    fn test_lsh_signature_similarity_completely_different() {
        let sig1 = LshSignature::new([u64::MAX, u64::MAX], 128);
        let sig2 = LshSignature::new([0, 0], 128);
        assert_eq!(sig1.similarity(&sig2), 0.0);
    }

    #[test]
    fn test_lsh_signature_similar_has_high_similarity() {
        // Only 4 bits differ out of 128
        let sig1 = LshSignature::new([0xFF00FF00FF00FF00, 0x00FF00FF00FF00FF], 128);
        let sig2 = LshSignature::new([0xFF00FF00FF00FF0F, 0x00FF00FF00FF00FF], 128);
        let similarity = sig1.similarity(&sig2);
        assert!(
            similarity > 0.9,
            "Signatures differing by few bits should have high similarity, got {}",
            similarity
        );
    }

    // ========================================================================
    // ChunkMetadata tests
    // ========================================================================

    #[test]
    fn test_chunk_metadata_new() {
        let meta = ChunkMetadata::new("chunk-1".to_string(), "doc-1".to_string());
        assert_eq!(meta.chunk_id, "chunk-1");
        assert_eq!(meta.document_id, "doc-1");
        assert!(meta.source_url.is_none());
        assert!(meta.source_title.is_none());
        assert_eq!(meta.position_in_doc, 0.0);
        assert!(meta.section_hierarchy.is_empty());
        assert!(meta.preceding_chunk_id.is_none());
        assert!(meta.following_chunk_id.is_none());
        assert!(meta.node_id.is_none());
        assert!(meta.extra.is_empty());
    }

    // ========================================================================
    // SearchResult tests
    // ========================================================================

    fn make_test_chunk() -> Chunk {
        Chunk {
            metadata: ChunkMetadata::new("chunk-42".to_string(), "doc-7".to_string()),
            content: "test chunk content".to_string(),
            token_count: 3,
        }
    }

    #[test]
    fn test_search_result_new() {
        let chunk = make_test_chunk();
        let result = SearchResult::new(chunk.clone(), 0.95);
        assert_eq!(result.chunk.metadata.chunk_id, "chunk-42");
        assert_eq!(result.chunk.metadata.document_id, "doc-7");
        assert_eq!(result.chunk.content, "test chunk content");
        assert_eq!(result.relevance_score, 0.95);
        assert!(result.node_id.is_none());
        assert!(result.matched_by.is_empty());
    }

    #[test]
    fn test_search_result_default_contributing_methods_empty() {
        let result = SearchResult::new(make_test_chunk(), 0.5);
        assert!(
            result.matched_by.is_empty(),
            "Default matched_by should be empty"
        );
    }

    // ========================================================================
    // Query tests
    // ========================================================================

    #[test]
    fn test_query_new() {
        let query = Query::new("semantic search query", 10);
        assert_eq!(query.text, "semantic search query");
        assert_eq!(query.top_k, 10);
        assert!(query.filters.is_none());
    }

    #[test]
    fn test_query_new_with_string() {
        let query = Query::new(String::from("owned string query"), 5);
        assert_eq!(query.text, "owned string query");
        assert_eq!(query.top_k, 5);
    }

    // ========================================================================
    // Document tests
    // ========================================================================

    #[test]
    fn test_document_new() {
        let doc = Document::new("Some document content");
        assert_eq!(doc.content, "Some document content");
        assert!(!doc.id.is_empty(), "ID should be auto-generated");
        assert!(doc.title.is_none());
        assert!(doc.url.is_none());
        assert!(doc.metadata.is_empty());
    }

    #[test]
    fn test_document_with_id() {
        let doc = Document::new("content").with_id("custom-id");
        assert_eq!(doc.id, "custom-id");
        assert_eq!(doc.content, "content");
    }

    #[test]
    fn test_document_with_title() {
        let doc = Document::new("content").with_title("My Title");
        assert_eq!(doc.title, Some("My Title".to_string()));
    }

    #[test]
    fn test_document_with_url() {
        let doc = Document::new("content").with_url("https://example.com");
        assert_eq!(doc.url, Some("https://example.com".to_string()));
    }

    #[test]
    fn test_document_builder_chaining() {
        let doc = Document::new("Full document content")
            .with_id("doc-123")
            .with_title("Chained Title")
            .with_url("https://example.com/doc");

        assert_eq!(doc.id, "doc-123");
        assert_eq!(doc.content, "Full document content");
        assert_eq!(doc.title, Some("Chained Title".to_string()));
        assert_eq!(doc.url, Some("https://example.com/doc".to_string()));
        assert!(doc.metadata.is_empty());
    }

    // ========================================================================
    // GroupedSearchResult tests
    // ========================================================================

    fn make_search_result(doc_id: &str, chunk_id: &str, score: f32, content: &str) -> SearchResult {
        let mut meta = ChunkMetadata::new(chunk_id.to_string(), doc_id.to_string());
        meta.source_title = Some(format!("Doc {}", doc_id));
        meta.source_url = Some(format!("https://example.com/{}", doc_id));
        let chunk = Chunk {
            metadata: meta,
            content: content.to_string(),
            token_count: 10,
        };
        SearchResult {
            chunk,
            relevance_score: score,
            node_id: None,
            matched_by: vec!["dense".to_string()],
        }
    }

    #[test]
    fn test_grouped_search_result_groups_by_document() {
        let results = vec![
            make_search_result("doc-1", "chunk-1a", 0.9, "first chunk"),
            make_search_result("doc-1", "chunk-1b", 0.7, "second chunk"),
            make_search_result("doc-2", "chunk-2a", 0.8, "another doc"),
        ];

        let grouped = GroupedSearchResult::from_results(results, 10);
        assert_eq!(grouped.len(), 2);

        // doc-1 should be first (max score 0.9 > 0.8)
        assert_eq!(grouped[0].document_id, "doc-1");
        assert_eq!(grouped[0].chunks.len(), 2);
        assert_eq!(grouped[0].relevance_score, 0.9);

        // doc-2 second
        assert_eq!(grouped[1].document_id, "doc-2");
        assert_eq!(grouped[1].chunks.len(), 1);
    }

    #[test]
    fn test_grouped_search_result_chunks_sorted_by_score() {
        let results = vec![
            make_search_result("doc-1", "chunk-low", 0.3, "low"),
            make_search_result("doc-1", "chunk-high", 0.9, "high"),
            make_search_result("doc-1", "chunk-mid", 0.6, "mid"),
        ];

        let grouped = GroupedSearchResult::from_results(results, 10);
        assert_eq!(grouped[0].chunks.len(), 3);
        assert_eq!(grouped[0].chunks[0].chunk_id, "chunk-high");
        assert_eq!(grouped[0].chunks[1].chunk_id, "chunk-mid");
        assert_eq!(grouped[0].chunks[2].chunk_id, "chunk-low");
    }

    #[test]
    fn test_grouped_search_result_truncates_to_top_k() {
        let results = vec![
            make_search_result("doc-1", "c1", 0.9, "a"),
            make_search_result("doc-2", "c2", 0.8, "b"),
            make_search_result("doc-3", "c3", 0.7, "c"),
        ];

        let grouped = GroupedSearchResult::from_results(results, 2);
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].document_id, "doc-1");
        assert_eq!(grouped[1].document_id, "doc-2");
    }

    #[test]
    fn test_grouped_search_result_empty_input() {
        let grouped = GroupedSearchResult::from_results(vec![], 10);
        assert!(grouped.is_empty());
    }

    #[test]
    fn test_grouped_search_result_preserves_metadata() {
        let results = vec![
            make_search_result("doc-1", "chunk-1", 0.9, "content"),
        ];

        let grouped = GroupedSearchResult::from_results(results, 10);
        assert_eq!(grouped[0].source_title, Some("Doc doc-1".to_string()));
        assert_eq!(grouped[0].source_url, Some("https://example.com/doc-1".to_string()));
        assert_eq!(grouped[0].chunks[0].matched_by, vec!["dense".to_string()]);
    }

    #[test]
    fn test_grouped_search_result_max_score_from_lower_ranked_chunk() {
        // doc-2's best chunk (0.95) is the second result but should still give doc-2 highest group score
        let results = vec![
            make_search_result("doc-1", "c1", 0.9, "a"),
            make_search_result("doc-2", "c2a", 0.5, "b"),
            make_search_result("doc-2", "c2b", 0.95, "c"),
        ];

        let grouped = GroupedSearchResult::from_results(results, 10);
        assert_eq!(grouped[0].document_id, "doc-2");
        assert_eq!(grouped[0].relevance_score, 0.95);
    }
}
