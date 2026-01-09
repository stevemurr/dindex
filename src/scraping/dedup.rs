//! Deduplication for URL and content
//!
//! Implements two-level deduplication:
//! - Level 1: URL deduplication using Bloom filter (local, fast)
//! - Level 2: Content deduplication using SimHash (network-wide via DHT)

use blake3::hash;
use lru::LruCache;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use url::Url;

/// Document identifier for deduplication
pub type DocumentId = String;

/// URL deduplicator using a scalable Bloom filter implementation
/// For simplicity, we use a HashSet with capacity tracking.
/// In production, this should be replaced with a proper scalable Bloom filter.
pub struct UrlDeduplicator {
    /// Set of seen URL hashes
    seen: HashSet<u64>,
    /// Target false positive rate
    _target_fp_rate: f64,
    /// Maximum capacity before reset
    max_capacity: usize,
}

impl UrlDeduplicator {
    /// Create a new URL deduplicator
    pub fn new(max_capacity: usize, target_fp_rate: f64) -> Self {
        Self {
            seen: HashSet::with_capacity(max_capacity.min(100_000)),
            _target_fp_rate: target_fp_rate,
            max_capacity,
        }
    }

    /// Check if a URL is new (not seen before)
    pub fn is_new_url(&mut self, url: &Url) -> bool {
        let normalized = Self::normalize_url(url);
        let hash = Self::hash_url(&normalized);

        if self.seen.contains(&hash) {
            false
        } else {
            // Check capacity
            if self.seen.len() >= self.max_capacity {
                // In a real implementation, this would use a scalable bloom filter
                // that adds new filters instead of resetting
                tracing::warn!("URL deduplicator at capacity, some duplicates may be recrawled");
            }
            self.seen.insert(hash);
            true
        }
    }

    /// Check if URL was seen (without marking it)
    pub fn was_seen(&self, url: &Url) -> bool {
        let normalized = Self::normalize_url(url);
        let hash = Self::hash_url(&normalized);
        self.seen.contains(&hash)
    }

    /// Mark a URL as seen
    pub fn mark_seen(&mut self, url: &Url) {
        let normalized = Self::normalize_url(url);
        let hash = Self::hash_url(&normalized);
        self.seen.insert(hash);
    }

    /// Get the number of URLs seen
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Normalize a URL for consistent hashing
    fn normalize_url(url: &Url) -> String {
        let mut normalized = url.clone();

        // Remove fragment
        normalized.set_fragment(None);

        // Sort query parameters
        if let Some(query) = normalized.query() {
            let mut params: Vec<_> = query.split('&').collect();
            params.sort();
            normalized.set_query(Some(&params.join("&")));
        }

        // Lowercase and return
        normalized.as_str().to_lowercase()
    }

    /// Hash a normalized URL
    fn hash_url(normalized: &str) -> u64 {
        let h = hash(normalized.as_bytes());
        // blake3 always produces 32 bytes, so [..8] is always valid
        let bytes: [u8; 8] = h.as_bytes()[..8]
            .try_into()
            .expect("blake3 hash is always 32 bytes");
        u64::from_be_bytes(bytes)
    }
}

impl Default for UrlDeduplicator {
    fn default() -> Self {
        Self::new(10_000_000, 0.01) // 10M URLs, 1% FP rate
    }
}

/// SimHash for content similarity detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimHash(pub u64);

impl SimHash {
    /// Compute SimHash for text content using 3-grams
    pub fn compute(text: &str) -> Self {
        let features = Self::extract_features(text);
        let hash = Self::compute_from_features(&features);
        SimHash(hash)
    }

    /// Extract features (3-grams) from text
    fn extract_features(text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 3 {
            return words.iter().map(|s| s.to_string()).collect();
        }

        words
            .windows(3)
            .map(|w| w.join(" "))
            .collect()
    }

    /// Compute SimHash from feature hashes
    fn compute_from_features(features: &[String]) -> u64 {
        if features.is_empty() {
            return 0;
        }

        // Initialize 64-bit vector of counts
        let mut v: [i32; 64] = [0; 64];

        // For each feature, hash it and update counts
        for feature in features {
            let h = hash(feature.as_bytes());
            // blake3 always produces 32 bytes, so [..8] is always valid
            let bytes: [u8; 8] = h.as_bytes()[..8]
                .try_into()
                .expect("blake3 hash is always 32 bytes");
            let feature_hash = u64::from_be_bytes(bytes);

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

    /// Calculate Hamming distance between two SimHashes
    pub fn hamming_distance(&self, other: &SimHash) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Check if two hashes are similar (within threshold)
    pub fn is_similar(&self, other: &SimHash, max_distance: u32) -> bool {
        self.hamming_distance(other) <= max_distance
    }

    /// Get the raw hash value
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Content deduplicator using SimHash
pub struct ContentDeduplicator {
    /// Local cache of recent SimHashes
    local_cache: LruCache<u64, DocumentId>,
    /// Maximum Hamming distance for near-duplicate detection
    max_distance: u32,
    /// Local document ID (for new content registration)
    local_node_id: String,
}

impl ContentDeduplicator {
    /// Create a new content deduplicator
    pub fn new(cache_size: usize, max_distance: u32, local_node_id: String) -> Self {
        // Use max(1) to ensure cache_size is at least 1
        let cache_capacity = NonZeroUsize::new(cache_size.max(1))
            .expect("max(1) guarantees non-zero");
        Self {
            local_cache: LruCache::new(cache_capacity),
            max_distance,
            local_node_id,
        }
    }

    /// Compute SimHash for text
    pub fn compute_simhash(&self, text: &str) -> SimHash {
        SimHash::compute(text)
    }

    /// Check if content is a duplicate (local check only)
    /// Returns Some(document_id) if duplicate found, None otherwise
    pub fn is_duplicate_local(&mut self, text: &str) -> Option<DocumentId> {
        let simhash = SimHash::compute(text);

        // Check exact match first
        if let Some(doc_id) = self.local_cache.get(&simhash.0) {
            return Some(doc_id.clone());
        }

        // Check near-duplicates (expensive for large caches)
        // In production, use a more efficient structure
        let cache_snapshot: Vec<(u64, DocumentId)> = self
            .local_cache
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();

        for (cached_hash, doc_id) in cache_snapshot {
            let cached_simhash = SimHash(cached_hash);
            if simhash.is_similar(&cached_simhash, self.max_distance) {
                return Some(doc_id);
            }
        }

        None
    }

    /// Register new content in the local cache
    pub fn register(&mut self, text: &str, document_id: DocumentId) -> SimHash {
        let simhash = SimHash::compute(text);
        self.local_cache.put(simhash.0, document_id);
        simhash
    }

    /// Register a pre-computed SimHash
    pub fn register_hash(&mut self, simhash: SimHash, document_id: DocumentId) {
        self.local_cache.put(simhash.0, document_id);
    }

    /// Get cache statistics
    pub fn stats(&self) -> DedupStats {
        DedupStats {
            cached_hashes: self.local_cache.len(),
            max_distance: self.max_distance,
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.local_cache.clear();
    }
}

/// Statistics from the deduplicator
#[derive(Debug, Clone)]
pub struct DedupStats {
    pub cached_hashes: usize,
    pub max_distance: u32,
}

/// SimHash query for DHT lookup
#[derive(Debug, Clone)]
pub struct SimHashQuery {
    /// The SimHash to query
    pub simhash: SimHash,
    /// Maximum Hamming distance
    pub max_distance: u32,
    /// Requesting node
    pub from_node: String,
}

/// SimHash registration for DHT storage
#[derive(Debug, Clone)]
pub struct SimHashRegistration {
    /// The SimHash value
    pub simhash: SimHash,
    /// Document ID
    pub document_id: DocumentId,
    /// Source node
    pub source_node: String,
    /// Registration timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_normalization() {
        let url1 = Url::parse("https://example.com/page#section").unwrap();
        let url2 = Url::parse("https://example.com/page").unwrap();

        let norm1 = UrlDeduplicator::normalize_url(&url1);
        let norm2 = UrlDeduplicator::normalize_url(&url2);

        assert_eq!(norm1, norm2);
    }

    #[test]
    fn test_url_dedup() {
        let mut dedup = UrlDeduplicator::new(1000, 0.01);

        let url1 = Url::parse("https://example.com/page1").unwrap();
        let url2 = Url::parse("https://example.com/page2").unwrap();

        assert!(dedup.is_new_url(&url1));
        assert!(!dedup.is_new_url(&url1)); // Second time should be duplicate
        assert!(dedup.is_new_url(&url2));
    }

    #[test]
    fn test_simhash_identical() {
        let text = "The quick brown fox jumps over the lazy dog";
        let hash1 = SimHash::compute(text);
        let hash2 = SimHash::compute(text);

        assert_eq!(hash1, hash2);
        assert_eq!(hash1.hamming_distance(&hash2), 0);
    }

    #[test]
    fn test_simhash_similar() {
        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "The quick brown fox leaps over the lazy dog"; // Changed one word

        let hash1 = SimHash::compute(text1);
        let hash2 = SimHash::compute(text2);

        // Should be similar (low Hamming distance)
        let distance = hash1.hamming_distance(&hash2);
        assert!(distance < 20, "Expected similar texts to have low distance, got {}", distance);
    }

    #[test]
    fn test_simhash_different() {
        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "Lorem ipsum dolor sit amet consectetur adipiscing elit";

        let hash1 = SimHash::compute(text1);
        let hash2 = SimHash::compute(text2);

        // Should be different (high Hamming distance)
        let distance = hash1.hamming_distance(&hash2);
        assert!(distance > 10, "Expected different texts to have high distance, got {}", distance);
    }

    #[test]
    fn test_content_dedup_local() {
        let mut dedup = ContentDeduplicator::new(100, 3, "node1".to_string());

        let text1 = "This is the first document with some content for testing purposes.";
        let text2 = "This is the first document with some content for testing purposes."; // Identical
        let text3 = "This is a completely different document with other content entirely.";

        // Register first document
        dedup.register(text1, "doc1".to_string());

        // Check for duplicate
        let dup_result = dedup
            .is_duplicate_local(text2)
            .expect("similar text should be detected as duplicate");
        assert_eq!(dup_result, "doc1");

        // Check different document
        let diff_result = dedup.is_duplicate_local(text3);
        assert!(diff_result.is_none());
    }

    #[test]
    fn test_simhash_features() {
        let text = "one two three four five";
        let features = SimHash::extract_features(text);

        assert_eq!(features.len(), 3); // 5 words = 3 trigrams
        assert_eq!(features[0], "one two three");
        assert_eq!(features[1], "two three four");
        assert_eq!(features[2], "three four five");
    }
}
