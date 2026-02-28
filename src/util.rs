//! Shared utility functions

/// Truncate a string to a maximum length, appending "..." if truncated.
/// Handles multi-byte characters by finding a valid char boundary.
pub fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    let suffix = "...";
    let target = max_len.saturating_sub(suffix.len());
    // Find a valid char boundary at or before target
    let mut end = target;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{}", &s[..end], suffix)
}

/// Truncate a string for display, collapsing newlines to spaces.
/// Handles multi-byte characters by finding a valid char boundary.
pub fn truncate_for_display(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max_len {
        return s;
    }
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// Compute SimHash from an iterator of string features using xxh3.
///
/// SimHash is a locality-sensitive hash that produces similar hashes for
/// similar inputs. Each feature string is hashed, and the resulting bits
/// are accumulated in a 64-element vote array. The final hash is formed
/// by setting each bit position to 1 if more features had that bit set
/// than not.
pub fn compute_simhash<'a>(features: impl Iterator<Item = &'a str>) -> u64 {
    let mut v = [0i32; 64];
    let mut has_features = false;

    for feature in features {
        has_features = true;
        let hash = xxhash_rust::xxh3::xxh3_64(feature.as_bytes());
        for (i, vote) in v.iter_mut().enumerate() {
            if (hash >> i) & 1 == 1 {
                *vote += 1;
            } else {
                *vote -= 1;
            }
        }
    }

    if !has_features {
        return 0;
    }

    let mut simhash: u64 = 0;
    for (i, vote) in v.iter().enumerate() {
        if *vote > 0 {
            simhash |= 1u64 << i;
        }
    }
    simhash
}

/// Normalize an embedding vector to unit length, returning a new vector.
pub fn normalize_embedding(embedding: &[f32]) -> Vec<f32> {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter().map(|x| x / norm).collect()
    } else {
        embedding.to_vec()
    }
}

/// Normalize an embedding vector to unit length in place.
pub fn normalize_in_place(v: &mut [f32]) {
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

    // ========================================================================
    // compute_simhash
    // ========================================================================

    #[test]
    fn simhash_deterministic() {
        let tokens = vec!["hello", "world", "foo", "bar"];
        let h1 = compute_simhash(tokens.iter().copied());
        let h2 = compute_simhash(tokens.iter().copied());
        assert_eq!(h1, h2, "same input must produce same hash");
    }

    #[test]
    fn simhash_empty_input_returns_zero() {
        let h = compute_simhash(std::iter::empty());
        assert_eq!(h, 0, "empty iterator should produce hash 0");
    }

    #[test]
    fn simhash_single_token() {
        let h = compute_simhash(std::iter::once("hello"));
        // A single token should produce a non-zero hash (extremely unlikely to be 0)
        assert_ne!(h, 0, "single token should produce a non-zero hash");
    }

    #[test]
    fn simhash_different_inputs_differ() {
        let h1 = compute_simhash(["apple", "banana", "cherry"].iter().copied());
        let h2 = compute_simhash(["dog", "cat", "fish"].iter().copied());
        assert_ne!(
            h1, h2,
            "completely different inputs should produce different hashes"
        );
    }

    #[test]
    fn simhash_similar_inputs_have_low_hamming_distance() {
        // Two very similar token lists that share most tokens
        let shared: Vec<&str> = (0..50).map(|_| "common_token").collect();

        let mut a = shared.clone();
        a.push("unique_a");

        let mut b = shared;
        b.push("unique_b");

        let h1 = compute_simhash(a.iter().copied());
        let h2 = compute_simhash(b.iter().copied());

        let hamming = (h1 ^ h2).count_ones();
        assert!(
            hamming <= 10,
            "similar inputs should have low hamming distance, got {}",
            hamming
        );
    }

    #[test]
    fn simhash_very_different_inputs_have_high_hamming_distance() {
        // Use many distinct tokens to push the hashes far apart
        let a: Vec<&str> = vec![
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
            "golf", "hotel", "india", "juliet",
        ];
        let b: Vec<&str> = vec![
            "kilo", "lima", "mike", "november", "oscar", "papa",
            "quebec", "romeo", "sierra", "tango",
        ];

        let h1 = compute_simhash(a.iter().copied());
        let h2 = compute_simhash(b.iter().copied());

        let hamming = (h1 ^ h2).count_ones();
        // Completely unrelated token sets should differ in many bit positions
        assert!(
            hamming > 5,
            "very different inputs should have higher hamming distance, got {}",
            hamming
        );
    }

    // ========================================================================
    // truncate_str
    // ========================================================================

    #[test]
    fn truncate_str_short_string_unchanged() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_str_exact_length_unchanged() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn truncate_str_long_string_truncated() {
        let result = truncate_str("hello world", 8);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 8);
        assert_eq!(result, "hello...");
    }

    #[test]
    fn truncate_str_unicode_boundary_safety() {
        // Multi-byte character: each char is 4 bytes
        let s = "\u{1F600}\u{1F601}\u{1F602}"; // three emoji, 12 bytes total
        let result = truncate_str(s, 8);
        // Should not panic and should produce valid UTF-8
        assert!(result.ends_with("..."));
        // The function should find a valid char boundary
        for ch in result.chars() {
            // Just iterating ensures it's valid UTF-8
            let _ = ch;
        }
    }

    #[test]
    fn truncate_str_very_short_max() {
        // max_len smaller than "..." (3 bytes)
        let result = truncate_str("hello world", 2);
        // target = 2 - 3 = 0 (saturating), so we get just "..."
        // Actually with saturating_sub(3) on 2, target = 0, end = 0, result = "..."
        assert_eq!(result, "...");
    }

    #[test]
    fn truncate_str_empty_string() {
        assert_eq!(truncate_str("", 10), "");
    }

    // ========================================================================
    // truncate_for_display
    // ========================================================================

    #[test]
    fn truncate_for_display_short_string_unchanged() {
        assert_eq!(truncate_for_display("hello", 10), "hello");
    }

    #[test]
    fn truncate_for_display_long_string_truncated() {
        let result = truncate_for_display("hello world, this is a long string", 11);
        assert_eq!(result.len(), 11);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn truncate_for_display_collapses_newlines() {
        let result = truncate_for_display("hello\nworld", 100);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn truncate_for_display_collapses_newlines_then_truncates() {
        let result = truncate_for_display("hello\nworld\nfoo\nbar", 11);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn truncate_for_display_unicode_boundary() {
        // Two emoji (8 bytes) + text
        let s = "\u{1F600}\u{1F601}abc";
        let result = truncate_for_display(s, 6);
        // Should not panic; finds valid char boundary at or before byte 6
        for ch in result.chars() {
            let _ = ch;
        }
    }

    #[test]
    fn truncate_for_display_empty_string() {
        assert_eq!(truncate_for_display("", 10), "");
    }

    // ========================================================================
    // normalize_embedding
    // ========================================================================

    #[test]
    fn normalize_embedding_unit_vector_unchanged() {
        let unit = vec![1.0_f32, 0.0, 0.0];
        let result = normalize_embedding(&unit);
        for (a, b) in result.iter().zip(unit.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn normalize_embedding_produces_unit_length() {
        let v = vec![3.0_f32, 4.0];
        let result = normalize_embedding(&v);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "expected unit length, got {}",
            norm
        );
        // 3/5 = 0.6, 4/5 = 0.8
        assert!((result[0] - 0.6).abs() < 1e-6);
        assert!((result[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_embedding_zero_vector_unchanged() {
        let zero = vec![0.0_f32, 0.0, 0.0];
        let result = normalize_embedding(&zero);
        assert_eq!(result, zero);
    }

    #[test]
    fn normalize_embedding_negative_values() {
        let v = vec![-3.0_f32, 4.0];
        let result = normalize_embedding(&v);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_embedding_already_normalized() {
        let v = vec![0.6_f32, 0.8];
        let result = normalize_embedding(&v);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    // ========================================================================
    // normalize_in_place
    // ========================================================================

    #[test]
    fn normalize_in_place_produces_unit_length() {
        let mut v = vec![3.0_f32, 4.0];
        normalize_in_place(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_in_place_zero_vector_unchanged() {
        let mut v = vec![0.0_f32, 0.0, 0.0];
        normalize_in_place(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_in_place_unit_vector_unchanged() {
        let mut v = vec![0.0_f32, 1.0, 0.0];
        normalize_in_place(&mut v);
        assert!((v[0]).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
        assert!((v[2]).abs() < 1e-6);
    }

    #[test]
    fn normalize_in_place_matches_normalize_embedding() {
        let original = vec![1.5_f32, -2.7, 0.3, 4.1];
        let from_fn = normalize_embedding(&original);
        let mut in_place = original;
        normalize_in_place(&mut in_place);
        for (a, b) in from_fn.iter().zip(in_place.iter()) {
            assert!((a - b).abs() < 1e-6, "results should match");
        }
    }
}
