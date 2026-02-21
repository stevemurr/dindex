//! HTTP API Server Module
//!
//! Provides a REST API for dindex, complementing the Unix socket IPC.
//! This enables remote access from iOS/visionOS clients and cross-platform tools.

pub mod auth;
pub mod handlers;
pub mod routes;
pub mod server;
pub mod types;

pub use server::HttpServer;
