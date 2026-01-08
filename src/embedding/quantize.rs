//! Embedding quantization utilities
//!
//! Supports:
//! - INT8 scalar quantization (4x compression, 99% performance retention)
//! - Binary quantization (32x compression for high-dimensional models)

use crate::types::{Embedding, QuantizedEmbedding};

/// Quantization parameters for INT8
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i8,
    pub min_val: f32,
    pub max_val: f32,
}

impl QuantizationParams {
    /// Calculate quantization parameters from embedding statistics
    pub fn from_embeddings(embeddings: &[Embedding]) -> Self {
        if embeddings.is_empty() {
            return Self::default();
        }

        // Find global min/max across all embeddings
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for emb in embeddings {
            for &val in emb {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Symmetric quantization around zero
        let abs_max = max_val.abs().max(min_val.abs());
        let scale = abs_max / 127.0;

        Self {
            scale,
            zero_point: 0,
            min_val: -abs_max,
            max_val: abs_max,
        }
    }
}

impl Default for QuantizationParams {
    fn default() -> Self {
        Self {
            scale: 1.0 / 127.0,
            zero_point: 0,
            min_val: -1.0,
            max_val: 1.0,
        }
    }
}

/// Quantize a single embedding to INT8
pub fn quantize_int8(embedding: &Embedding, params: &QuantizationParams) -> QuantizedEmbedding {
    embedding
        .iter()
        .map(|&val| {
            let quantized = (val / params.scale).round() as i32;
            quantized.clamp(-128, 127) as i8
        })
        .collect()
}

/// Dequantize INT8 embedding back to f32
pub fn dequantize_int8(quantized: &QuantizedEmbedding, params: &QuantizationParams) -> Embedding {
    quantized
        .iter()
        .map(|&val| (val as f32) * params.scale)
        .collect()
}

/// Batch quantize embeddings
pub fn quantize_batch(
    embeddings: &[Embedding],
    params: &QuantizationParams,
) -> Vec<QuantizedEmbedding> {
    embeddings
        .iter()
        .map(|emb| quantize_int8(emb, params))
        .collect()
}

/// Binary quantization (1 bit per dimension)
/// Returns a packed bit vector
pub fn quantize_binary(embedding: &Embedding) -> Vec<u8> {
    let num_bytes = (embedding.len() + 7) / 8;
    let mut result = vec![0u8; num_bytes];

    for (i, &val) in embedding.iter().enumerate() {
        if val > 0.0 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            result[byte_idx] |= 1 << bit_idx;
        }
    }

    result
}

/// Compute Hamming distance between binary quantized embeddings
pub fn binary_hamming_distance(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as usize)
        .sum()
}

/// Compute approximate cosine similarity using INT8 quantized embeddings
pub fn quantized_cosine_similarity(
    a: &QuantizedEmbedding,
    b: &QuantizedEmbedding,
    params: &QuantizationParams,
) -> f32 {
    assert_eq!(a.len(), b.len());

    // Compute dot product in i32 to avoid overflow
    let dot: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i64) * (y as i64))
        .sum();

    // Compute magnitudes
    let mag_a: i64 = a.iter().map(|&x| (x as i64) * (x as i64)).sum();
    let mag_b: i64 = b.iter().map(|&x| (x as i64) * (x as i64)).sum();

    if mag_a > 0 && mag_b > 0 {
        (dot as f64 / ((mag_a as f64).sqrt() * (mag_b as f64).sqrt())) as f32
    } else {
        0.0
    }
}

/// Product quantization (PQ) codebook
/// Divides the embedding into M subspaces, each quantized with K centroids
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    /// Number of subspaces
    pub num_subspaces: usize,
    /// Number of centroids per subspace
    pub num_centroids: usize,
    /// Codebook: [num_subspaces][num_centroids][subspace_dim]
    pub codebook: Vec<Vec<Vec<f32>>>,
    /// Dimension per subspace
    pub subspace_dim: usize,
}

impl ProductQuantizer {
    /// Create a new product quantizer (codebook must be trained separately)
    pub fn new(embedding_dim: usize, num_subspaces: usize, num_centroids: usize) -> Self {
        assert!(embedding_dim % num_subspaces == 0);
        let subspace_dim = embedding_dim / num_subspaces;

        // Initialize with random codebook (should be trained with k-means)
        let codebook = (0..num_subspaces)
            .map(|_| {
                (0..num_centroids)
                    .map(|_| vec![0.0f32; subspace_dim])
                    .collect()
            })
            .collect();

        Self {
            num_subspaces,
            num_centroids,
            codebook,
            subspace_dim,
        }
    }

    /// Encode an embedding using product quantization
    pub fn encode(&self, embedding: &Embedding) -> Vec<u8> {
        assert!(self.num_centroids <= 256, "PQ requires num_centroids <= 256 for u8 codes");

        let mut codes = Vec::with_capacity(self.num_subspaces);

        for m in 0..self.num_subspaces {
            let start = m * self.subspace_dim;
            let end = start + self.subspace_dim;
            let subvector = &embedding[start..end];

            // Find nearest centroid
            let mut best_centroid = 0u8;
            let mut best_dist = f32::MAX;

            for (k, centroid) in self.codebook[m].iter().enumerate() {
                let dist: f32 = subvector
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                if dist < best_dist {
                    best_dist = dist;
                    best_centroid = k as u8;
                }
            }

            codes.push(best_centroid);
        }

        codes
    }

    /// Decode PQ codes back to approximate embedding
    pub fn decode(&self, codes: &[u8]) -> Embedding {
        assert_eq!(codes.len(), self.num_subspaces);

        let mut embedding = Vec::with_capacity(self.num_subspaces * self.subspace_dim);

        for (m, &code) in codes.iter().enumerate() {
            embedding.extend_from_slice(&self.codebook[m][code as usize]);
        }

        embedding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_quantization() {
        let embedding = vec![0.5, -0.3, 0.8, -0.1, 0.0];
        let params = QuantizationParams::default();

        let quantized = quantize_int8(&embedding, &params);
        let dequantized = dequantize_int8(&quantized, &params);

        // Check that dequantized is close to original
        for (orig, deq) in embedding.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.02);
        }
    }

    #[test]
    fn test_binary_quantization() {
        let embedding = vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.1, -0.5, 0.2];
        let binary = quantize_binary(&embedding);

        // Positive values: indices 0, 2, 5, 7
        // Binary: 10100101 = 0b10100101
        assert_eq!(binary.len(), 1);
        assert_eq!(binary[0], 0b10100101);
    }

    #[test]
    fn test_quantized_cosine() {
        let a: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
        let b: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];

        let params = QuantizationParams::default();
        let qa = quantize_int8(&a, &params);
        let qb = quantize_int8(&b, &params);

        let sim = quantized_cosine_similarity(&qa, &qb, &params);
        assert!((sim - 1.0).abs() < 0.01);
    }
}
