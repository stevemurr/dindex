//! libp2p behaviour composition

use libp2p::{
    autonat, gossipsub, identify, identity::Keypair, kad, mdns, ping, relay,
    request_response,
    swarm::NetworkBehaviour,
    PeerId, StreamProtocol,
};
use std::time::Duration;

use super::protocol::DIndexCodec;

/// Combined network behaviour for DIndex
#[derive(NetworkBehaviour)]
pub struct DIndexBehaviour {
    /// Kademlia DHT for peer discovery and content routing
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    /// GossipSub for pub/sub messaging
    pub gossipsub: gossipsub::Behaviour,
    /// Identify protocol for peer identification
    pub identify: identify::Behaviour,
    /// mDNS for local peer discovery
    pub mdns: mdns::tokio::Behaviour,
    /// Ping for connection keepalive
    pub ping: ping::Behaviour,
    /// AutoNAT for NAT detection
    pub autonat: autonat::Behaviour,
    /// Relay for NAT traversal
    pub relay: relay::client::Behaviour,
    /// Request-response for direct peer queries (avoids GossipSub broadcast)
    pub request_response: request_response::Behaviour<DIndexCodec>,
}

impl DIndexBehaviour {
    /// Create a new DIndex behaviour
    pub fn new(
        local_peer_id: PeerId,
        local_public_key: libp2p::identity::PublicKey,
        keypair: Keypair,
        relay_behaviour: relay::client::Behaviour,
    ) -> anyhow::Result<Self> {
        // Kademlia configuration
        let protocol = StreamProtocol::new(super::QUERY_PROTOCOL);
        let mut kad_config = kad::Config::new(protocol);
        // SAFETY: 20 is always non-zero
        let replication_factor = std::num::NonZeroUsize::new(20)
            .expect("20 is non-zero");
        kad_config
            .set_replication_factor(replication_factor)
            .set_query_timeout(Duration::from_secs(60))
            .set_record_ttl(Some(Duration::from_secs(48 * 60 * 60))) // 48 hours
            .set_provider_record_ttl(Some(Duration::from_secs(48 * 60 * 60)))
            .set_publication_interval(Some(Duration::from_secs(22 * 60 * 60))); // 22 hours

        let kademlia = kad::Behaviour::with_config(
            local_peer_id,
            kad::store::MemoryStore::new(local_peer_id),
            kad_config,
        );

        // GossipSub configuration
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(10))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .message_id_fn(|msg| {
                // Include source in hash so identical payloads from different peers
                // produce distinct message IDs (prevents dropping legitimate responses)
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                if let Some(source) = &msg.source {
                    hasher.update(source.to_bytes());
                }
                hasher.update(&msg.data);
                gossipsub::MessageId::from(hex::encode(hasher.finalize()))
            })
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build gossipsub config: {}", e))?;

        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(keypair),
            gossipsub_config,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create gossipsub: {}", e))?;

        // Identify configuration
        let identify = identify::Behaviour::new(identify::Config::new(
            super::PROTOCOL_VERSION.to_string(),
            local_public_key,
        ));

        // mDNS for local discovery
        let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), local_peer_id)?;

        // Ping for keepalive
        let ping = ping::Behaviour::new(ping::Config::new().with_interval(Duration::from_secs(30)));

        // AutoNAT
        let autonat = autonat::Behaviour::new(local_peer_id, autonat::Config::default());

        // Request-response for direct queries
        let rr_config = request_response::Config::default()
            .with_request_timeout(Duration::from_secs(30));
        let rr = request_response::Behaviour::new(
            [(
                StreamProtocol::new(super::protocol::DIRECT_QUERY_PROTOCOL),
                request_response::ProtocolSupport::Full,
            )],
            rr_config,
        );

        Ok(Self {
            kademlia,
            gossipsub,
            identify,
            mdns,
            ping,
            autonat,
            relay: relay_behaviour,
            request_response: rr,
        })
    }
}

