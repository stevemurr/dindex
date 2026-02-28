//! Daemon, HTTP API, and metrics configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Daemon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Automatically start daemon if not running
    pub auto_start: bool,
    /// Socket path override (defaults to XDG_RUNTIME_DIR/dindex/dindex.sock)
    pub socket_path: Option<PathBuf>,
    /// Write pipeline batch size
    pub batch_size: usize,
    /// Commit interval in seconds
    pub commit_interval_secs: u64,
    /// Maximum pending writes before forcing commit
    pub max_pending_writes: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            auto_start: false,
            socket_path: None,
            batch_size: 100,
            commit_interval_secs: 30,
            max_pending_writes: 10000,
        }
    }
}

/// HTTP API server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpConfig {
    /// Enable HTTP API server
    pub enabled: bool,
    /// Listen address for HTTP server (e.g., "0.0.0.0:8080")
    pub listen_addr: String,
    /// API keys for authentication (empty = no auth required)
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// Enable CORS (useful for browser-based clients)
    pub cors_enabled: bool,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            listen_addr: "127.0.0.1:8080".to_string(),
            api_keys: Vec::new(),
            cors_enabled: false,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable Prometheus /metrics endpoint
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Interval for updating system metrics (memory, etc.) in seconds
    #[serde(default = "default_system_metrics_interval")]
    pub system_metrics_interval_secs: u64,
}

fn default_true() -> bool {
    true
}

fn default_system_metrics_interval() -> u64 {
    15
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            system_metrics_interval_secs: 15,
        }
    }
}
