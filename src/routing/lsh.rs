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
        assert_eq!(embedding.len(), self.dimensions);

        let num_u64s = (self.num_bits + 63) / 64;
        let mut bits = vec![0u64; num_u64s];

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

    /// Hash multiple embeddings
    pub fn hash_batch(&self, embeddings: &[Embedding]) -> Vec<LshSignature> {
        embeddings.iter().map(|e| self.hash(e)).collect()
    }

    /// Estimate cosine similarity from LSH signatures
    /// Uses the relationship: cos(θ) ≈ 1 - 2 * hamming_distance / num_bits
    pub fn estimate_similarity(&self, a: &LshSignature, b: &LshSignature) -> f32 {
        let hamming = a.hamming_distance(b);
        // Cosine similarity estimate based on angular distance
        let theta = std::f32::consts::PI * (hamming as f32 / self.num_bits as f32);
        theta.cos()
    }

    /// Find candidates above a similarity threshold
    pub fn find_candidates(
        &self,
        query: &LshSignature,
        signatures: &[(String, LshSignature)],
        min_similarity: f32,
    ) -> Vec<(String, f32)> {
        let max_hamming = ((1.0 - min_similarity.acos() / std::f32::consts::PI) * self.num_bits as f32) as usize;

        signatures
            .iter()
            .filter_map(|(id, sig)| {
                let hamming = query.hamming_distance(sig);
                if hamming <= max_hamming {
                    Some((id.clone(), self.estimate_similarity(query, sig)))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Multi-probe LSH for better recall
pub struct MultiProbeLsh {
    base_lsh: LshIndex,
    num_probes: usize,
}

impl MultiProbeLsh {
    pub fn new(dimensions: usize, num_bits: usize, num_probes: usize, seed: u64) -> Self {
        Self {
            base_lsh: LshIndex::new(dimensions, num_bits, seed),
            num_probes,
        }
    }

    /// Hash with multi-probe (returns base signature + nearby signatures)
    pub fn hash_with_probes(&self, embedding: &Embedding) -> Vec<LshSignature> {
        let base = self.base_lsh.hash(embedding);
        let mut signatures = vec![base.clone()];

        // Generate probe signatures by flipping individual bits
        // This is a simplified version - full multi-probe uses more sophisticated probing
        for i in 0..self.num_probes.min(self.base_lsh.num_bits) {
            let mut probe_bits = base.bits.clone();
            let word_idx = i / 64;
            let bit_idx = i % 64;
            probe_bits[word_idx] ^= 1 << bit_idx;
            signatures.push(LshSignature::new(probe_bits, base.num_bits));
        }

        signatures
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

        let estimated_sim = lsh.estimate_similarity(&sig1, &sig2);
        assert!(estimated_sim > 0.9, "Estimated similarity should be high");
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
