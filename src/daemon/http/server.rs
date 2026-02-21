//! HTTP API Server
//!
//! Axum-based HTTP server for the dindex REST API.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::http::Method;
use tokio::net::TcpListener;
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::config::HttpConfig;
use crate::daemon::handler::RequestHandler;

use super::auth::AuthState;
use super::handlers::AppState;
use super::routes::create_router;

/// HTTP API server
pub struct HttpServer {
    config: HttpConfig,
    handler: Arc<RequestHandler>,
}

impl HttpServer {
    /// Create a new HTTP server
    pub fn new(config: HttpConfig, handler: Arc<RequestHandler>) -> Self {
        Self { config, handler }
    }

    /// Run the HTTP server
    pub async fn run(&self, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
        let addr: SocketAddr = self
            .config
            .listen_addr
            .parse()
            .context("Invalid HTTP listen address")?;

        // Create application state
        let app_state = AppState {
            handler: self.handler.clone(),
        };

        // Create auth state
        let auth_state = AuthState::new(self.config.api_keys.clone());

        // Create router
        let mut app = create_router(app_state, auth_state);

        // Add CORS if enabled
        if self.config.cors_enabled {
            let cors = CorsLayer::new()
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(Any)
                .allow_origin(Any);
            app = app.layer(cors);
        }

        // Add tracing
        app = app.layer(TraceLayer::new_for_http());

        // Bind to address
        let listener = TcpListener::bind(&addr)
            .await
            .context("Failed to bind HTTP server")?;

        info!("HTTP API server listening on http://{}", addr);

        // Run server with graceful shutdown
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown.recv().await;
                info!("HTTP server shutting down");
            })
            .await
            .context("HTTP server error")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_listen_addr() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        assert_eq!(addr.port(), 8080);

        let addr: SocketAddr = "0.0.0.0:9000".parse().unwrap();
        assert_eq!(addr.port(), 9000);
    }
}
