//! P2P networking layer using libp2p
//!
//! Features:
//! - Kademlia DHT for peer discovery and content routing
//! - GossipSub for pub/sub messaging
//! - QUIC transport for low-latency connections
//! - AutoNAT for NAT traversal

mod behaviour;
mod messages;
mod node;

pub use behaviour::*;
pub use messages::*;
pub use node::*;
