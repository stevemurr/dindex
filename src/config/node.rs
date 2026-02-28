//! Node networking configuration

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Node networking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Listen address for P2P connections
    pub listen_addr: String,
    /// Bootstrap peers to connect to
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,
    /// Data directory for persistence
    pub data_dir: PathBuf,
    /// Enable mDNS for local peer discovery
    pub enable_mdns: bool,
    /// DHT replication factor (k)
    pub replication_factor: usize,
    /// Query timeout in seconds
    pub query_timeout_secs: u64,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            listen_addr: "/ip4/0.0.0.0/udp/0/quic-v1".to_string(),
            bootstrap_peers: Vec::new(),
            data_dir: directories::ProjectDirs::from("", "", "dindex")
                .map(|d| d.data_dir().to_path_buf())
                .unwrap_or_else(|| PathBuf::from(".dindex")),
            enable_mdns: true,
            replication_factor: 3,
            query_timeout_secs: 10,
        }
    }
}
