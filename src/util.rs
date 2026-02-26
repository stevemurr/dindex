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
