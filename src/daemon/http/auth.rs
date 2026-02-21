//! HTTP API Authentication Middleware
//!
//! Provides API key authentication for the HTTP API.

use axum::{
    body::Body,
    extract::State,
    http::{header, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;

use super::types::ErrorResponse;

/// Shared state for authentication
#[derive(Clone)]
pub struct AuthState {
    /// Valid API keys (empty means no auth required)
    pub api_keys: Arc<Vec<String>>,
}

impl AuthState {
    pub fn new(api_keys: Vec<String>) -> Self {
        Self {
            api_keys: Arc::new(api_keys),
        }
    }

    /// Check if authentication is required
    pub fn auth_required(&self) -> bool {
        !self.api_keys.is_empty()
    }

    /// Validate an API key
    pub fn validate_key(&self, key: &str) -> bool {
        if self.api_keys.is_empty() {
            return true;
        }
        self.api_keys.iter().any(|k| k == key)
    }
}

/// Authentication middleware
pub async fn auth_middleware(
    State(auth): State<AuthState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Skip auth if no API keys configured
    if !auth.auth_required() {
        return next.run(request).await;
    }

    // Check for API key in Authorization header
    // Supports: "Bearer <key>" or just "<key>"
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    let api_key = auth_header.map(|h| {
        if let Some(key) = h.strip_prefix("Bearer ") {
            key.trim()
        } else {
            h.trim()
        }
    });

    match api_key {
        Some(key) if auth.validate_key(key) => next.run(request).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse::unauthorized()),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_state_no_keys() {
        let auth = AuthState::new(vec![]);
        assert!(!auth.auth_required());
        assert!(auth.validate_key("anything"));
    }

    #[test]
    fn test_auth_state_with_keys() {
        let auth = AuthState::new(vec!["secret123".to_string(), "key456".to_string()]);
        assert!(auth.auth_required());
        assert!(auth.validate_key("secret123"));
        assert!(auth.validate_key("key456"));
        assert!(!auth.validate_key("wrong"));
    }
}
