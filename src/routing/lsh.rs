//! Locality-Sensitive Hashing (LSH) for fast similarity estimation
//!
//! Uses random hyperplane LSH for cosine similarity

use crate::types::{Embedding, LshSignature};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// LSH index using random hyperplanes
pub struct LshIndex {
    /// Random hyperplanes for hashing
    hyperplanes: Vec<Vec<f32>>,
    /// Number of hash bits
    num_bits: usize,
    /// Embedding dimensions
    dimensions: usize,
}

impl LshIndex {
    /// Create a new LSH index with random hyperplanes
    pub fn new(dimensions: usize, num_bits: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Generate random hyperplanes (each with dimension = embedding_dim)
        let hyperplanes: Vec<Vec<f32>> = (0..num_bits)
            .map(|_| {
                (0..dimensions)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        Self {
            hyperplanes,
            num_bits,
            dimensions,
        }
    }

    /// Compute LSH signature for an embedding
    pub fn hash(&self, embedding: &Embedding) -> LshSignature {
        if embedding.len() != self.dimensions {
            tracing::warn!(
                "LSH dimension mismatch: expected {}, got {} — returning zero signature",
                self.dimensions,
                embedding.len()
            );
            return LshSignature::new([0u64; 2], self.num_bits);
        }
        assert!(self.num_bits <= 128, "LSH signature supports at most 128 bits");

        let mut bits = [0u64; 2];

        for (i, hyperplane) in self.hyperplanes.iter().enumerate() {
            // Compute dot product with hyperplane
            let dot: f32 = embedding
                .iter()
                .zip(hyperplane.iter())
                .map(|(e, h)| e * h)
                .sum();

            // Set bit if dot product is positive
            if dot > 0.0 {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                bits[word_idx] |= 1 << bit_idx;
            }
        }

        LshSignature::new(bits, self.num_bits)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_similar_vectors() {
        let lsh = LshIndex::new(128, 64, 42);

        // Create two similar vectors
        let v1: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0).sin()).collect();
        let mut v2 = v1.clone();
        // Add small noise
        for val in v2.iter_mut() {
            *val += 0.01;
        }

        let sig1 = lsh.hash(&v1);
        let sig2 = lsh.hash(&v2);

        // Similar vectors should have similar signatures (low hamming distance)
        let distance = sig1.hamming_distance(&sig2);
        assert!(distance < 10, "Similar vectors should have low hamming distance");

        let estimated_sim = sig1.similarity(&sig2);
        assert!(estimated_sim > 0.9, "Estimated similarity should be high");
    }

    #[test]
    fn test_lsh_deterministic() {
        let lsh1 = LshIndex::new(64, 64, 99);
        let lsh2 = LshIndex::new(64, 64, 99);
        let v: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0).cos()).collect();

        let sig1 = lsh1.hash(&v);
        let sig2 = lsh2.hash(&v);

        assert_eq!(sig1, sig2, "Same seed should produce identical signatures");
    }

    #[test]
    fn test_lsh_dimension_mismatch_returns_zero() {
        let lsh = LshIndex::new(128, 64, 42);
        // Embedding with wrong dimensions (64 instead of 128)
        let v: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();

        let sig = lsh.hash(&v);
        assert_eq!(sig.bits, [0u64; 2], "Dimension mismatch should return zero signature");
    }

    #[test]
    fn test_lsh_different_vectors() {
        let lsh = LshIndex::new(128, 64, 42);

        // Create two orthogonal vectors
        let v1: Vec<f32> = (0..128).map(|i| if i < 64 { 1.0 } else { 0.0 }).collect();
        let v2: Vec<f32> = (0..128).map(|i| if i >= 64 { 1.0 } else { 0.0 }).collect();

        let sig1 = lsh.hash(&v1);
        let sig2 = lsh.hash(&v2);

        // Orthogonal vectors should have signatures with ~50% hamming distance
        let distance = sig1.hamming_distance(&sig2);
        assert!(
            distance > 20 && distance < 44,
            "Orthogonal vectors should have ~32 hamming distance, got {}",
            distance
        );
    }
}
