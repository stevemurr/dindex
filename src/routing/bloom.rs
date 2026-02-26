//! Bloom filter for efficient negative filtering
//!
//! Used to quickly eliminate irrelevant nodes before centroid comparison

use xxhash_rust::xxh3::xxh3_64_with_seed;

/// Bloom filter implementation
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Bit array
    bits: Vec<u8>,
    /// Number of bits
    num_bits: usize,
    /// Number of hash functions
    num_hashes: usize,
}

impl BloomFilter {
    /// Create a new bloom filter
    ///
    /// # Arguments
    /// * `num_items` - Expected number of items
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    pub fn new(num_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal parameters
        // m = -n * ln(p) / (ln(2)^2)
        let m = (-(num_items as f64) * false_positive_rate.ln() / (2.0_f64.ln().powi(2))).ceil() as usize;
        let num_bits = m.max(8);
        let num_bytes = (num_bits + 7) / 8;

        // k = m/n * ln(2)
        let k = ((num_bits as f64 / num_items as f64) * 2.0_f64.ln()).round() as usize;
        let num_hashes = k.max(1).min(16);

        Self {
            bits: vec![0u8; num_bytes],
            num_bits,
            num_hashes,
        }
    }

    /// Create a bloom filter with specific parameters
    pub fn with_params(num_bits: usize, num_hashes: usize) -> Self {
        let num_bytes = (num_bits + 7) / 8;
        Self {
            bits: vec![0u8; num_bytes],
            num_bits,
            num_hashes,
        }
    }

    /// Insert an item into the bloom filter
    pub fn insert(&mut self, item: &[u8]) {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let bit_idx = hash % self.num_bits;
            let byte_idx = bit_idx / 8;
            let bit_offset = bit_idx % 8;
            self.bits[byte_idx] |= 1 << bit_offset;
        }
    }

    /// Check if an item might be in the set
    /// Returns false if definitely not in set, true if possibly in set
    pub fn contains(&self, item: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let hash = self.hash(item, i);
            let bit_idx = hash % self.num_bits;
            let byte_idx = bit_idx / 8;
            let bit_offset = bit_idx % 8;

            if (self.bits[byte_idx] & (1 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Get hash for item with given seed
    fn hash(&self, item: &[u8], seed: usize) -> usize {
        xxh3_64_with_seed(item, seed as u64) as usize
    }

    /// Export the bloom filter as bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(8 + 8 + self.bits.len());
        result.extend_from_slice(&(self.num_bits as u64).to_le_bytes());
        result.extend_from_slice(&(self.num_hashes as u64).to_le_bytes());
        result.extend_from_slice(&self.bits);
        result
    }

    /// Import a bloom filter from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 16 {
            return None;
        }

        let num_bits = u64::from_le_bytes(data[0..8].try_into().ok()?) as usize;
        let num_hashes = u64::from_le_bytes(data[8..16].try_into().ok()?) as usize;
        let bits = data[16..].to_vec();

        if bits.len() < (num_bits + 7) / 8 {
            return None;
        }

        Some(Self {
            bits,
            num_bits,
            num_hashes,
        })
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.bits.len()
    }

    /// Get fill ratio (fraction of bits set)
    pub fn fill_ratio(&self) -> f64 {
        let set_bits: usize = self.bits.iter().map(|b| b.count_ones() as usize).sum();
        set_bits as f64 / self.num_bits as f64
    }
}

/// LSH Banded Bloom Filter for semantic routing
///
/// Uses the LSH banding technique: divides signatures into bands and checks
/// if ANY band matches. This correctly handles the fact that similar vectors
/// have similar (not identical) LSH signatures.
#[derive(Debug, Clone)]
pub struct BandedBloomFilter {
    /// Underlying bloom filter
    bloom: BloomFilter,
    /// Number of bands to divide the LSH signature into
    num_bands: usize,
    /// Bits per band
    bits_per_band: usize,
}

impl BandedBloomFilter {
    /// Create a new banded bloom filter
    ///
    /// # Arguments
    /// * `num_items` - Expected number of LSH signatures to store
    /// * `lsh_bits` - Total bits in each LSH signature
    /// * `num_bands` - Number of bands to divide signature into
    /// * `false_positive_rate` - Desired false positive rate per band
    pub fn new(num_items: usize, lsh_bits: usize, num_bands: usize, false_positive_rate: f64) -> Self {
        // Each item contributes `num_bands` entries to the bloom filter
        let total_entries = num_items * num_bands;
        let bloom = BloomFilter::new(total_entries, false_positive_rate);
        let bits_per_band = lsh_bits / num_bands;

        Self {
            bloom,
            num_bands,
            bits_per_band,
        }
    }

    /// Create with specific parameters
    pub fn with_params(num_bits: usize, num_hashes: usize, lsh_bits: usize, num_bands: usize) -> Self {
        Self {
            bloom: BloomFilter::with_params(num_bits, num_hashes),
            num_bands,
            bits_per_band: lsh_bits / num_bands,
        }
    }

    /// Insert an LSH signature (adds all bands)
    pub fn insert(&mut self, signature: &[u64], num_bits: usize) {
        for band_idx in 0..self.num_bands {
            let band_hash = self.extract_band_hash(signature, num_bits, band_idx);
            self.bloom.insert(&band_hash);
        }
    }

    /// Check if ANY band of the signature matches
    /// Returns true if the signature might have a similar item in the filter
    pub fn might_contain_similar(&self, signature: &[u64], num_bits: usize) -> bool {
        for band_idx in 0..self.num_bands {
            let band_hash = self.extract_band_hash(signature, num_bits, band_idx);
            if self.bloom.contains(&band_hash) {
                return true;
            }
        }
        false
    }

    /// Count how many bands match (for scoring)
    pub fn count_matching_bands(&self, signature: &[u64], num_bits: usize) -> usize {
        let mut count = 0;
        for band_idx in 0..self.num_bands {
            let band_hash = self.extract_band_hash(signature, num_bits, band_idx);
            if self.bloom.contains(&band_hash) {
                count += 1;
            }
        }
        count
    }

    /// Extract a band from the signature and create a hashable key
    fn extract_band_hash(&self, signature: &[u64], num_bits: usize, band_idx: usize) -> [u8; 9] {
        let start_bit = band_idx * self.bits_per_band;
        let end_bit = (start_bit + self.bits_per_band).min(num_bits);

        // Extract the band bits
        let mut band_bits: u64 = 0;
        for bit in start_bit..end_bit {
            let word_idx = bit / 64;
            let bit_offset = bit % 64;
            if word_idx < signature.len() && (signature[word_idx] & (1 << bit_offset)) != 0 {
                band_bits |= 1 << (bit - start_bit);
            }
        }

        // Create key: [band_idx as u8] + [band_bits as le_bytes]
        let mut key = [0u8; 9];
        key[0] = band_idx as u8;
        key[1..9].copy_from_slice(&band_bits.to_le_bytes());
        key
    }

    /// Export to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let bloom_bytes = self.bloom.to_bytes();
        let mut result = Vec::with_capacity(8 + bloom_bytes.len());
        // Store num_bands and bits_per_band
        result.extend_from_slice(&(self.num_bands as u32).to_le_bytes());
        result.extend_from_slice(&(self.bits_per_band as u32).to_le_bytes());
        result.extend_from_slice(&bloom_bytes);
        result
    }

    /// Import from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }

        let num_bands = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let bits_per_band = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let bloom = BloomFilter::from_bytes(&data[8..])?;

        Some(Self {
            bloom,
            num_bands,
            bits_per_band,
        })
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        8 + self.bloom.size_bytes()
    }

    /// Get fill ratio
    pub fn fill_ratio(&self) -> f64 {
        self.bloom.fill_ratio()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter() {
        let mut bf = BloomFilter::new(100, 0.01);

        bf.insert(b"hello");
        bf.insert(b"world");

        assert!(bf.contains(b"hello"));
        assert!(bf.contains(b"world"));
        assert!(!bf.contains(b"foo")); // Might have false positive, but unlikely
    }

    #[test]
    fn test_bloom_filter_serialization() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"test");

        let bytes = bf.to_bytes();
        let bf2 = BloomFilter::from_bytes(&bytes).unwrap();

        assert!(bf2.contains(b"test"));
        assert_eq!(bf.num_bits, bf2.num_bits);
    }

    #[test]
    fn test_banded_bloom_filter_exact_match() {
        // 128-bit LSH signatures with 8 bands of 16 bits each
        let mut bbf = BandedBloomFilter::new(100, 128, 8, 0.01);

        // Create a test signature (2 x u64 = 128 bits)
        let sig1 = vec![0x1234567890ABCDEFu64, 0xFEDCBA0987654321u64];

        bbf.insert(&sig1, 128);

        // Exact match should definitely be found
        assert!(bbf.might_contain_similar(&sig1, 128));
        assert_eq!(bbf.count_matching_bands(&sig1, 128), 8);
    }

    #[test]
    fn test_banded_bloom_filter_similar_signature() {
        let mut bbf = BandedBloomFilter::new(100, 128, 8, 0.01);

        // Create original signature
        let sig1 = vec![0x1234567890ABCDEFu64, 0xFEDCBA0987654321u64];
        bbf.insert(&sig1, 128);

        // Create similar signature (flip a few bits - simulates similar vectors)
        // Flip bits 0, 64, 65 (affects 2 bands)
        let sig2 = vec![0x1234567890ABCDEEu64, 0xFEDCBA0987654320u64];

        // Should still match (most bands unchanged)
        assert!(bbf.might_contain_similar(&sig2, 128));
        // At least 6 bands should match (we flipped bits in at most 2 bands)
        assert!(bbf.count_matching_bands(&sig2, 128) >= 6);
    }

    #[test]
    fn test_banded_bloom_filter_different_signature() {
        let mut bbf = BandedBloomFilter::new(10, 128, 8, 0.01);

        // Only insert one signature
        let sig1 = vec![0x1234567890ABCDEFu64, 0xFEDCBA0987654321u64];
        bbf.insert(&sig1, 128);

        // Completely different signature
        let sig2 = vec![0x0000000000000000u64, 0x0000000000000000u64];

        // Should have very few matching bands (only false positives)
        let matches = bbf.count_matching_bands(&sig2, 128);
        // With only 10 items and 8 bands, false positive rate should be low
        assert!(matches <= 2, "Expected few false positives, got {}", matches);
    }

    #[test]
    fn test_banded_bloom_filter_serialization() {
        let mut bbf = BandedBloomFilter::new(100, 128, 8, 0.01);
        let sig = vec![0x1234567890ABCDEFu64, 0xFEDCBA0987654321u64];
        bbf.insert(&sig, 128);

        let bytes = bbf.to_bytes();
        let bbf2 = BandedBloomFilter::from_bytes(&bytes).unwrap();

        assert!(bbf2.might_contain_similar(&sig, 128));
        assert_eq!(bbf2.num_bands, 8);
    }
}
