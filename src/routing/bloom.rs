//! Bloom filter for efficient negative filtering
//!
//! Used to quickly eliminate irrelevant nodes before centroid comparison

use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

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

    /// Estimate the number of items in the filter
    pub fn estimate_count(&self) -> usize {
        let set_bits: usize = self.bits.iter().map(|b| b.count_ones() as usize).sum();
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;
        let x = set_bits as f64;

        // n ≈ -m/k * ln(1 - x/m)
        ((-m / k) * (1.0 - x / m).ln()).round() as usize
    }

    /// Merge another bloom filter into this one (union)
    pub fn merge(&mut self, other: &BloomFilter) {
        assert_eq!(self.num_bits, other.num_bits);
        assert_eq!(self.num_hashes, other.num_hashes);

        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
    }

    /// Get fill ratio (fraction of bits set)
    pub fn fill_ratio(&self) -> f64 {
        let set_bits: usize = self.bits.iter().map(|b| b.count_ones() as usize).sum();
        set_bits as f64 / self.num_bits as f64
    }
}

/// Counting bloom filter (allows deletions)
#[derive(Debug, Clone)]
pub struct CountingBloomFilter {
    /// Counters (4 bits each, packed into bytes)
    counters: Vec<u8>,
    /// Number of counters
    num_counters: usize,
    /// Number of hash functions
    num_hashes: usize,
}

impl CountingBloomFilter {
    /// Create a new counting bloom filter
    pub fn new(num_items: usize, false_positive_rate: f64) -> Self {
        let m = (-(num_items as f64) * false_positive_rate.ln() / (2.0_f64.ln().powi(2))).ceil() as usize;
        let num_counters = m.max(8);
        let num_bytes = (num_counters + 1) / 2; // 4 bits per counter

        let k = ((num_counters as f64 / num_items as f64) * 2.0_f64.ln()).round() as usize;
        let num_hashes = k.max(1).min(16);

        Self {
            counters: vec![0u8; num_bytes],
            num_counters,
            num_hashes,
        }
    }

    /// Insert an item
    pub fn insert(&mut self, item: &[u8]) {
        for i in 0..self.num_hashes {
            let hash = xxh3_64_with_seed(item, i as u64) as usize;
            let idx = hash % self.num_counters;
            self.increment_counter(idx);
        }
    }

    /// Remove an item
    pub fn remove(&mut self, item: &[u8]) {
        for i in 0..self.num_hashes {
            let hash = xxh3_64_with_seed(item, i as u64) as usize;
            let idx = hash % self.num_counters;
            self.decrement_counter(idx);
        }
    }

    /// Check if item might be present
    pub fn contains(&self, item: &[u8]) -> bool {
        for i in 0..self.num_hashes {
            let hash = xxh3_64_with_seed(item, i as u64) as usize;
            let idx = hash % self.num_counters;
            if self.get_counter(idx) == 0 {
                return false;
            }
        }
        true
    }

    fn get_counter(&self, idx: usize) -> u8 {
        let byte_idx = idx / 2;
        if idx % 2 == 0 {
            self.counters[byte_idx] & 0x0F
        } else {
            (self.counters[byte_idx] >> 4) & 0x0F
        }
    }

    fn increment_counter(&mut self, idx: usize) {
        let byte_idx = idx / 2;
        let current = self.get_counter(idx);
        if current < 15 {
            // Don't overflow
            if idx % 2 == 0 {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0xF0) | (current + 1);
            } else {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0x0F) | ((current + 1) << 4);
            }
        }
    }

    fn decrement_counter(&mut self, idx: usize) {
        let byte_idx = idx / 2;
        let current = self.get_counter(idx);
        if current > 0 {
            if idx % 2 == 0 {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0xF0) | (current - 1);
            } else {
                self.counters[byte_idx] = (self.counters[byte_idx] & 0x0F) | ((current - 1) << 4);
            }
        }
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
    fn test_counting_bloom_filter() {
        let mut cbf = CountingBloomFilter::new(100, 0.01);

        cbf.insert(b"hello");
        assert!(cbf.contains(b"hello"));

        cbf.remove(b"hello");
        assert!(!cbf.contains(b"hello"));
    }
}
