//! Content centroid computation for semantic routing
//!
//! Generates cluster centroids from node content for efficient routing

use crate::types::{Embedding, NodeCentroid};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tracing::{debug, info};

/// Centroid generator using k-means clustering
pub struct CentroidGenerator {
    num_centroids: usize,
    max_iterations: usize,
    convergence_threshold: f32,
}

impl CentroidGenerator {
    pub fn new(num_centroids: usize) -> Self {
        Self {
            num_centroids,
            max_iterations: 100,
            convergence_threshold: 1e-6,
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Generate centroids from embeddings using k-means++
    pub fn generate(&self, embeddings: &[Embedding]) -> Vec<NodeCentroid> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let k = self.num_centroids.min(embeddings.len());
        if k == 0 {
            return Vec::new();
        }

        info!("Generating {} centroids from {} embeddings", k, embeddings.len());

        // Initialize centroids using k-means++
        let mut centroids = self.kmeans_plus_plus_init(embeddings, k);

        // Run k-means iterations
        let mut assignments = vec![0usize; embeddings.len()];
        let mut prev_inertia = f32::MAX;

        for iteration in 0..self.max_iterations {
            // Assign points to nearest centroid
            for (i, embedding) in embeddings.iter().enumerate() {
                let mut best_centroid = 0;
                let mut best_distance = f32::MAX;

                for (j, centroid) in centroids.iter().enumerate() {
                    let distance = cosine_distance(embedding, centroid);
                    if distance < best_distance {
                        best_distance = distance;
                        best_centroid = j;
                    }
                }
                assignments[i] = best_centroid;
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0f32; embeddings[0].len()]; k];
            let mut counts = vec![0usize; k];

            for (i, embedding) in embeddings.iter().enumerate() {
                let cluster = assignments[i];
                counts[cluster] += 1;
                for (j, &val) in embedding.iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
            }

            // Compute mean and normalize
            for (centroid, &count) in new_centroids.iter_mut().zip(counts.iter()) {
                if count > 0 {
                    for val in centroid.iter_mut() {
                        *val /= count as f32;
                    }
                    normalize_in_place(centroid);
                }
            }

            // Check convergence
            let inertia = self.compute_inertia(embeddings, &new_centroids, &assignments);
            let delta = (prev_inertia - inertia).abs();

            debug!(
                "K-means iteration {}: inertia = {:.6}, delta = {:.6}",
                iteration, inertia, delta
            );

            if delta < self.convergence_threshold {
                info!("K-means converged after {} iterations", iteration + 1);
                break;
            }

            centroids = new_centroids;
            prev_inertia = inertia;
        }

        // Count points per centroid
        let mut centroid_counts = vec![0usize; k];
        for &assignment in &assignments {
            centroid_counts[assignment] += 1;
        }

        // Build NodeCentroid results
        centroids
            .into_iter()
            .enumerate()
            .filter(|(i, _)| centroid_counts[*i] > 0)
            .map(|(i, centroid)| NodeCentroid {
                centroid_id: i as u32,
                embedding: centroid,
                chunk_count: centroid_counts[i],
            })
            .collect()
    }

    /// Initialize centroids using k-means++ algorithm
    fn kmeans_plus_plus_init(&self, embeddings: &[Embedding], k: usize) -> Vec<Embedding> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut centroids = Vec::with_capacity(k);

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..embeddings.len());
        centroids.push(embeddings[first_idx].clone());

        // Choose remaining centroids with probability proportional to distance squared
        while centroids.len() < k {
            let mut distances: Vec<f32> = embeddings
                .iter()
                .map(|emb| {
                    centroids
                        .iter()
                        .map(|c| cosine_distance(emb, c))
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            // Square distances for probability
            for d in distances.iter_mut() {
                *d = d.powi(2);
            }

            // Normalize to probability distribution
            let sum: f32 = distances.iter().sum();
            if sum > 0.0 {
                for d in distances.iter_mut() {
                    *d /= sum;
                }
            }

            // Sample from distribution
            let r: f32 = rng.gen();
            let mut cumsum = 0.0;
            let mut chosen_idx = 0;

            for (i, &prob) in distances.iter().enumerate() {
                cumsum += prob;
                if cumsum >= r {
                    chosen_idx = i;
                    break;
                }
            }

            centroids.push(embeddings[chosen_idx].clone());
        }

        centroids
    }

    /// Compute total inertia (sum of squared distances to assigned centroids)
    fn compute_inertia(
        &self,
        embeddings: &[Embedding],
        centroids: &[Embedding],
        assignments: &[usize],
    ) -> f32 {
        embeddings
            .iter()
            .zip(assignments.iter())
            .map(|(emb, &cluster)| cosine_distance(emb, &centroids[cluster]).powi(2))
            .sum()
    }
}

/// Truncate centroids for Matryoshka routing (reduce dimensions)
pub fn truncate_centroids(centroids: &[NodeCentroid], target_dims: usize) -> Vec<NodeCentroid> {
    centroids
        .iter()
        .map(|c| {
            let truncated: Vec<f32> = c.embedding.iter().take(target_dims).copied().collect();
            let mut normalized = truncated;
            normalize_in_place(&mut normalized);

            NodeCentroid {
                centroid_id: c.centroid_id,
                embedding: normalized,
                chunk_count: c.chunk_count,
            }
        })
        .collect()
}

/// Find nearest centroids for a query embedding
pub fn find_nearest_centroids(
    query: &Embedding,
    centroids: &[NodeCentroid],
    top_k: usize,
) -> Vec<(u32, f32)> {
    let mut scores: Vec<(u32, f32)> = centroids
        .iter()
        .map(|c| {
            let sim = cosine_similarity(query, &c.embedding);
            (c.centroid_id, sim)
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);
    scores
}

/// Cosine distance (1 - cosine similarity)
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Normalize vector in place
fn normalize_in_place(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in v.iter_mut() {
            *val /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centroid_generation() {
        let generator = CentroidGenerator::new(3);

        // Create clustered embeddings
        let mut embeddings = Vec::new();

        // Cluster 1: around [1, 0, 0]
        for _ in 0..10 {
            embeddings.push(vec![1.0 + rand::random::<f32>() * 0.1, rand::random::<f32>() * 0.1, rand::random::<f32>() * 0.1]);
        }

        // Cluster 2: around [0, 1, 0]
        for _ in 0..10 {
            embeddings.push(vec![rand::random::<f32>() * 0.1, 1.0 + rand::random::<f32>() * 0.1, rand::random::<f32>() * 0.1]);
        }

        // Cluster 3: around [0, 0, 1]
        for _ in 0..10 {
            embeddings.push(vec![rand::random::<f32>() * 0.1, rand::random::<f32>() * 0.1, 1.0 + rand::random::<f32>() * 0.1]);
        }

        let centroids = generator.generate(&embeddings);
        assert_eq!(centroids.len(), 3);

        // Each centroid should have approximately 10 chunks
        for c in &centroids {
            assert!(c.chunk_count >= 5);
        }
    }

    #[test]
    fn test_truncate_centroids() {
        let centroids = vec![
            NodeCentroid {
                centroid_id: 0,
                embedding: vec![1.0, 0.5, 0.3, 0.1],
                chunk_count: 10,
            },
        ];

        let truncated = truncate_centroids(&centroids, 2);
        assert_eq!(truncated[0].embedding.len(), 2);
    }
}
