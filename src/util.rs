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
        for i in 0..64 {
            if (hash >> i) & 1 == 1 {
                v[i] += 1;
            } else {
                v[i] -= 1;
            }
        }
    }

    if !has_features {
        return 0;
    }

    let mut simhash: u64 = 0;
    for i in 0..64 {
        if v[i] > 0 {
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
