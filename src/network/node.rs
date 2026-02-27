//! P2P network node implementation

use super::{
    behaviour::DIndexBehaviour,
    messages::{topics, NetworkMessage, QueryRequest, QueryResponse},
};
use crate::config::NodeConfig;
use crate::types::{NodeAdvertisement, Query};

use anyhow::Result;
use futures::StreamExt;
use libp2p::{
    gossipsub::{self, IdentTopic},
    identity::Keypair,
    identify, kad,
    ping,
    request_response::{self, ResponseChannel},
    swarm::SwarmEvent,
    Multiaddr, PeerId, Swarm,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

/// Maximum number of connected peers tracked
const MAX_CONNECTED_PEERS: usize = 500;
/// Maximum number of cached node advertisements
const MAX_NODE_ADVERTISEMENTS: usize = 1000;
/// Maximum number of pending response channels before cleanup
const MAX_PENDING_RESPONSE_CHANNELS: usize = 500;
/// How long stale response channels are kept before cleanup (seconds)
const STALE_RESPONSE_CHANNEL_SECS: u64 = 60;
/// How long node advertisements are kept before expiration (seconds)
const ADVERTISEMENT_EXPIRY_SECS: i64 = 3600; // 1 hour

/// Pending query state
struct PendingQuery {
    response_tx: oneshot::Sender<Vec<QueryResponse>>,
    responses: Vec<QueryResponse>,
    expected_peers: Vec<PeerId>,
    deadline: Instant,
}

/// Network node for P2P communication
pub struct NetworkNode {
    /// Local peer ID
    pub local_peer_id: PeerId,
    /// Keypair for signing (reserved for future signing operations)
    #[allow(dead_code)]
    keypair: Keypair,
    /// Swarm instance
    swarm: Swarm<DIndexBehaviour>,
    /// Connected peers
    connected_peers: Arc<RwLock<HashMap<PeerId, PeerInfo>>>,
    /// Cached node advertisements
    node_advertisements: Arc<RwLock<HashMap<PeerId, NodeAdvertisement>>>,
    /// Pending distributed queries awaiting responses
    pending_queries: HashMap<String, PendingQuery>,
    /// Response channels for incoming request-response queries (request_id -> (channel, created_at))
    pending_response_channels: HashMap<String, (ResponseChannel<QueryResponse>, Instant)>,
    /// Mapping from outbound request-response IDs to our query request_ids
    outbound_request_map: HashMap<request_response::OutboundRequestId, String>,
    /// Command receiver
    command_rx: mpsc::Receiver<NetworkCommand>,
    /// Event sender
    event_tx: mpsc::Sender<NetworkEvent>,
}

/// Information about a connected peer
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub peer_id: PeerId,
    pub addresses: Vec<Multiaddr>,
    pub connected_at: Instant,
    pub last_seen: Instant,
}

/// Commands to send to the network node
#[derive(Debug)]
pub enum NetworkCommand {
    /// Connect to a peer
    Connect(Multiaddr),
    /// Disconnect from a peer
    Disconnect(PeerId),
    /// Send a query to specific peers
    SendQuery {
        peers: Vec<PeerId>,
        request: QueryRequest,
        response_tx: oneshot::Sender<Vec<QueryResponse>>,
        timeout: Duration,
    },
    /// Send a response to a query
    SendQueryResponse(QueryResponse),
    /// Broadcast advertisement
    BroadcastAdvertisement(NodeAdvertisement),
    /// Get connected peers
    GetPeers(oneshot::Sender<Vec<PeerInfo>>),
    /// Shutdown
    Shutdown,
}

/// Events from the network node
#[derive(Debug, Clone)]
pub enum NetworkEvent {
    /// Query received from peer - application should handle and respond
    QueryReceived {
        peer_id: PeerId,
        request: QueryRequest,
    },
    /// Advertisement received
    AdvertisementReceived(NodeAdvertisement),
    /// Peer connected
    PeerConnected(PeerId),
    /// Peer disconnected
    PeerDisconnected(PeerId),
}

/// Handle for interacting with the network node
#[derive(Clone)]
pub struct NetworkHandle {
    command_tx: mpsc::Sender<NetworkCommand>,
    pub local_peer_id: PeerId,
}

impl NetworkHandle {
    /// Connect to a peer
    pub async fn connect(&self, addr: Multiaddr) -> Result<()> {
        self.command_tx
            .send(NetworkCommand::Connect(addr))
            .await
            .map_err(|_| anyhow::anyhow!("Network node shut down"))?;
        Ok(())
    }

    /// Send a query to peers
    pub async fn query(
        &self,
        peers: Vec<PeerId>,
        query: Query,
        embedding: Option<Vec<f32>>,
        timeout: Duration,
    ) -> Result<Vec<QueryResponse>> {
        let mut request = QueryRequest::new(query)
            .with_origin(self.local_peer_id.to_string())
            .with_target_peers(peers.iter().map(|p| p.to_string()).collect());
        if let Some(emb) = embedding {
            request = request.with_embedding(emb);
        }

        let (tx, rx) = oneshot::channel();
        self.command_tx
            .send(NetworkCommand::SendQuery {
                peers,
                request,
                response_tx: tx,
                timeout,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Network node shut down"))?;

        // Wait for responses - the network node handles the timeout internally
        rx.await.map_err(|_| anyhow::anyhow!("Response channel closed"))
    }

    /// Send a response to a query
    pub async fn send_response(&self, response: QueryResponse) -> Result<()> {
        self.command_tx
            .send(NetworkCommand::SendQueryResponse(response))
            .await
            .map_err(|_| anyhow::anyhow!("Network node shut down"))?;
        Ok(())
    }

    /// Broadcast node advertisement
    pub async fn broadcast_advertisement(&self, advertisement: NodeAdvertisement) -> Result<()> {
        self.command_tx
            .send(NetworkCommand::BroadcastAdvertisement(advertisement))
            .await
            .map_err(|_| anyhow::anyhow!("Network node shut down"))?;
        Ok(())
    }

    /// Get connected peers
    pub async fn get_peers(&self) -> Result<Vec<PeerInfo>> {
        let (tx, rx) = oneshot::channel();
        self.command_tx
            .send(NetworkCommand::GetPeers(tx))
            .await
            .map_err(|_| anyhow::anyhow!("Network node shut down"))?;
        rx.await.map_err(|_| anyhow::anyhow!("Response channel closed"))
    }

    /// Shutdown the network node
    pub async fn shutdown(&self) -> Result<()> {
        self.command_tx
            .send(NetworkCommand::Shutdown)
            .await
            .map_err(|_| anyhow::anyhow!("Network node already shut down"))?;
        Ok(())
    }
}

/// Maximum allowed size for incoming network messages (16 MB).
/// Messages exceeding this limit are dropped to prevent memory exhaustion attacks.
const MAX_NETWORK_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Load an existing node keypair from disk, or generate and persist a new one.
///
/// The keypair is stored at `{data_dir}/node_key` using libp2p's protobuf encoding.
/// This ensures the node retains a stable PeerId across restarts.
fn load_or_generate_keypair(data_dir: &Path) -> Result<Keypair> {
    let key_path = data_dir.join("node_key");

    if key_path.exists() {
        let bytes = std::fs::read(&key_path)
            .map_err(|e| anyhow::anyhow!("Failed to read keypair from {}: {}", key_path.display(), e))?;
        let keypair = Keypair::from_protobuf_encoding(&bytes)
            .map_err(|e| anyhow::anyhow!("Failed to decode keypair from {}: {}", key_path.display(), e))?;
        info!("Loaded existing node keypair from {}", key_path.display());
        Ok(keypair)
    } else {
        let keypair = Keypair::generate_ed25519();
        // Ensure the data directory exists
        std::fs::create_dir_all(data_dir)
            .map_err(|e| anyhow::anyhow!("Failed to create data directory {}: {}", data_dir.display(), e))?;
        let bytes = keypair.to_protobuf_encoding()
            .map_err(|e| anyhow::anyhow!("Failed to encode keypair: {}", e))?;
        std::fs::write(&key_path, &bytes)
            .map_err(|e| anyhow::anyhow!("Failed to write keypair to {}: {}", key_path.display(), e))?;
        info!("Generated new node keypair, saved to {}", key_path.display());
        Ok(keypair)
    }
}

impl NetworkNode {
    /// Create a new network node
    pub async fn new(
        _config: &NodeConfig,
    ) -> Result<(Self, NetworkHandle, mpsc::Receiver<NetworkEvent>)> {
        // Load persisted keypair or generate a new one
        let keypair = load_or_generate_keypair(&_config.data_dir)?;
        let local_peer_id = PeerId::from(keypair.public());

        info!("Local peer ID: {}", local_peer_id);

        // Build the swarm
        let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair.clone())
            .with_tokio()
            .with_quic()
            .with_relay_client(noise::Config::new, yamux::Config::default)?
            .with_behaviour(|key, relay_behaviour| {
                DIndexBehaviour::new(
                    PeerId::from(key.public()),
                    key.public(),
                    key.clone(),
                    relay_behaviour,
                )
                .expect("Failed to create behaviour")
            })?
            .with_swarm_config(|c| {
                c.with_idle_connection_timeout(Duration::from_secs(60))
            })
            .build();

        // Create channels
        let (command_tx, command_rx) = mpsc::channel(256);
        let (event_tx, event_rx) = mpsc::channel(256);

        let handle = NetworkHandle {
            command_tx,
            local_peer_id,
        };

        let node = Self {
            local_peer_id,
            keypair,
            swarm,
            connected_peers: Arc::new(RwLock::new(HashMap::new())),
            node_advertisements: Arc::new(RwLock::new(HashMap::new())),
            pending_queries: HashMap::new(),
            pending_response_channels: HashMap::new(),
            outbound_request_map: HashMap::new(),
            command_rx,
            event_tx,
        };

        Ok((node, handle, event_rx))
    }

    /// Start listening and processing events
    pub async fn run(mut self, config: &NodeConfig) -> Result<()> {
        // Start listening
        let listen_addr: Multiaddr = config.listen_addr.parse()?;
        self.swarm.listen_on(listen_addr)?;

        // Subscribe to gossipsub topics
        let advert_topic = IdentTopic::new(topics::ADVERTISEMENTS);
        let query_topic = IdentTopic::new(topics::QUERIES);
        let response_topic = IdentTopic::new(topics::QUERY_RESPONSES);

        self.swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&advert_topic)?;
        self.swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&query_topic)?;
        self.swarm
            .behaviour_mut()
            .gossipsub
            .subscribe(&response_topic)?;

        // Connect to bootstrap peers
        for peer_addr in &config.bootstrap_peers {
            if let Ok(addr) = peer_addr.parse::<Multiaddr>() {
                info!("Connecting to bootstrap peer: {}", addr);
                if let Err(e) = self.swarm.dial(addr.clone()) {
                    warn!("Failed to dial bootstrap peer {}: {}", addr, e);
                }
            }
        }

        // Main event loop
        let mut query_timeout_check = tokio::time::interval(Duration::from_millis(100));

        loop {
            tokio::select! {
                // Handle swarm events
                event = self.swarm.select_next_some() => {
                    self.handle_swarm_event(event).await;
                }
                // Handle commands
                Some(cmd) = self.command_rx.recv() => {
                    if matches!(cmd, NetworkCommand::Shutdown) {
                        info!("Network node shutting down");
                        break;
                    }
                    self.handle_command(cmd).await;
                }
                // Check for query timeouts
                _ = query_timeout_check.tick() => {
                    self.check_query_timeouts();
                }
            }
        }

        Ok(())
    }

    async fn handle_swarm_event(&mut self, event: SwarmEvent<super::behaviour::DIndexBehaviourEvent>) {
        match event {
            SwarmEvent::Behaviour(behaviour_event) => {
                self.handle_behaviour_event(behaviour_event).await;
            }
            SwarmEvent::ConnectionEstablished {
                peer_id, endpoint, ..
            } => {
                info!("Connected to peer: {}", peer_id);
                let addr = endpoint.get_remote_address().clone();

                {
                    let mut peers = self.connected_peers.write();
                    peers.insert(
                        peer_id,
                        PeerInfo {
                            peer_id,
                            addresses: vec![addr],
                            connected_at: Instant::now(),
                            last_seen: Instant::now(),
                        },
                    );

                    // Enforce max connected peers by removing oldest
                    if peers.len() > MAX_CONNECTED_PEERS {
                        let oldest = peers
                            .iter()
                            .min_by_key(|(_, info)| info.last_seen)
                            .map(|(pid, _)| *pid);
                        if let Some(oldest_pid) = oldest {
                            peers.remove(&oldest_pid);
                            warn!("Evicted oldest peer {} to stay within max peer limit", oldest_pid);
                        }
                    }
                }

                // Add to Kademlia
                self.swarm
                    .behaviour_mut()
                    .kademlia
                    .add_address(&peer_id, endpoint.get_remote_address().clone());

                let _ = self.event_tx.send(NetworkEvent::PeerConnected(peer_id)).await;
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                info!("Disconnected from peer: {}", peer_id);
                self.connected_peers.write().remove(&peer_id);
                let _ = self
                    .event_tx
                    .send(NetworkEvent::PeerDisconnected(peer_id))
                    .await;
            }
            SwarmEvent::NewListenAddr { address, .. } => {
                info!("Listening on: {}/p2p/{}", address, self.local_peer_id);
            }
            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                if let Some(peer_id) = peer_id {
                    warn!("Failed to connect to {}: {}", peer_id, error);
                }
            }
            _ => {}
        }
    }

    async fn handle_behaviour_event(&mut self, event: super::behaviour::DIndexBehaviourEvent) {
        use super::behaviour::DIndexBehaviourEvent;

        match event {
            DIndexBehaviourEvent::Mdns(mdns::Event::Discovered(peers)) => {
                for (peer_id, addr) in peers {
                    debug!("Discovered peer via mDNS: {} at {}", peer_id, addr);
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());

                    if let Err(e) = self.swarm.dial(addr) {
                        debug!("Failed to dial discovered peer: {}", e);
                    }
                }
            }
            DIndexBehaviourEvent::Mdns(mdns::Event::Expired(peers)) => {
                for (peer_id, _) in peers {
                    debug!("mDNS peer expired: {}", peer_id);
                }
            }
            DIndexBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message_id,
                message,
            }) => {
                debug!(
                    "GossipSub message from {}: {:?}",
                    propagation_source, message_id
                );

                if message.data.len() > MAX_NETWORK_MESSAGE_SIZE {
                    warn!(
                        "Dropping oversized network message from {}: {} bytes exceeds {} byte limit",
                        propagation_source,
                        message.data.len(),
                        MAX_NETWORK_MESSAGE_SIZE,
                    );
                    return;
                }

                if let Ok(msg) = bincode::deserialize::<NetworkMessage>(&message.data) {
                    match msg {
                        NetworkMessage::Advertisement(advert) => {
                            let _node_id = advert.node_id.clone();
                            self.node_advertisements
                                .write()
                                .insert(propagation_source, advert.clone());
                            let _ = self
                                .event_tx
                                .send(NetworkEvent::AdvertisementReceived(advert))
                                .await;
                        }
                        NetworkMessage::QueryRequest(request) => {
                            // Filter: if target_peers is non-empty, only execute if we're targeted
                            if !request.target_peers.is_empty() {
                                let local_id = self.local_peer_id.to_string();
                                if !request.target_peers.contains(&local_id) {
                                    debug!(
                                        "Dropping query {} — not targeted (targets: {:?})",
                                        request.request_id, request.target_peers
                                    );
                                    return;
                                }
                            }
                            debug!("Received query request: {}", request.request_id);
                            let _ = self
                                .event_tx
                                .send(NetworkEvent::QueryReceived {
                                    peer_id: propagation_source,
                                    request,
                                })
                                .await;
                        }
                        NetworkMessage::QueryResponse(response) => {
                            debug!("Received GossipSub query response: {}", response.request_id);
                            self.handle_query_response(response);
                        }
                    }
                }
            }
            DIndexBehaviourEvent::Kademlia(kad::Event::OutboundQueryProgressed {
                id,
                result,
                ..
            }) => {
                debug!("Kademlia query {:?} progressed: {:?}", id, result);
            }
            DIndexBehaviourEvent::Identify(identify::Event::Received { peer_id, info, .. }) => {
                debug!(
                    "Identified peer {}: {} with {:?}",
                    peer_id, info.protocol_version, info.protocols
                );

                // Add addresses to Kademlia
                for addr in info.listen_addrs {
                    self.swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr);
                }
            }
            DIndexBehaviourEvent::Ping(ping::Event { peer, result, .. }) => {
                if let Ok(rtt) = result {
                    debug!("Ping to {}: {:?}", peer, rtt);
                    if let Some(info) = self.connected_peers.write().get_mut(&peer) {
                        info.last_seen = Instant::now();
                    }
                }
            }
            DIndexBehaviourEvent::RequestResponse(
                request_response::Event::Message { peer, message },
            ) => {
                match message {
                    request_response::Message::Request {
                        request,
                        channel,
                        ..
                    } => {
                        debug!(
                            "Direct query from {}: {}",
                            peer, request.request_id
                        );
                        // Store the response channel so we can reply directly
                        self.pending_response_channels
                            .insert(request.request_id.clone(), (channel, Instant::now()));
                        let _ = self
                            .event_tx
                            .send(NetworkEvent::QueryReceived {
                                peer_id: peer,
                                request,
                            })
                            .await;
                    }
                    request_response::Message::Response {
                        response,
                        request_id: outbound_id,
                    } => {
                        debug!(
                            "Direct response from {}: {}",
                            peer, response.request_id
                        );
                        // Clean up the outbound mapping
                        self.outbound_request_map.remove(&outbound_id);
                        // Route to pending_queries (same path as GossipSub responses)
                        self.handle_query_response(response);
                    }
                }
            }
            DIndexBehaviourEvent::RequestResponse(
                request_response::Event::OutboundFailure {
                    peer,
                    request_id: outbound_id,
                    error,
                },
            ) => {
                warn!("Direct query to {} failed: {}", peer, error);
                // Clean up and check if we should complete the pending query
                if let Some(req_id) = self.outbound_request_map.remove(&outbound_id) {
                    // Check if all expected peers have responded or failed
                    if let Some(pending) = self.pending_queries.get(&req_id) {
                        if pending.responses.len() + 1 >= pending.expected_peers.len()
                            || Instant::now() >= pending.deadline
                        {
                            if let Some(pending) = self.pending_queries.remove(&req_id) {
                                let _ = pending.response_tx.send(pending.responses);
                            }
                        }
                    }
                }
            }
            DIndexBehaviourEvent::RequestResponse(
                request_response::Event::InboundFailure {
                    peer, error, ..
                },
            ) => {
                warn!("Inbound request from {} failed: {}", peer, error);
            }
            DIndexBehaviourEvent::RequestResponse(
                request_response::Event::ResponseSent { peer, .. },
            ) => {
                debug!("Direct response sent to {}", peer);
            }
            _ => {}
        }
    }

    /// Publish a query response via GossipSub broadcast
    fn publish_gossipsub_response(&mut self, response: &QueryResponse) {
        let msg = NetworkMessage::QueryResponse(response.clone());
        if let Ok(data) = bincode::serialize(&msg) {
            debug!("Sending GossipSub query response: {}", response.request_id);
            let topic = IdentTopic::new(topics::QUERY_RESPONSES);
            if let Err(e) = self
                .swarm
                .behaviour_mut()
                .gossipsub
                .publish(topic, data)
            {
                warn!("Failed to publish query response: {}", e);
            }
        }
    }

    /// Handle a query response (shared between GossipSub and request-response paths)
    fn handle_query_response(&mut self, response: QueryResponse) {
        if let Some(pending) = self.pending_queries.get_mut(&response.request_id) {
            // Validate responder: if responder_peer is set, check it's expected
            if let Some(ref responder) = response.responder_peer {
                let is_expected = pending
                    .expected_peers
                    .iter()
                    .any(|p| p.to_string() == *responder);
                if !is_expected {
                    debug!(
                        "Dropping response from unexpected peer {} for query {}",
                        responder, response.request_id
                    );
                    return;
                }

                // Check for duplicate responses from the same peer
                let already_responded = pending
                    .responses
                    .iter()
                    .any(|r| r.responder_peer.as_deref() == Some(responder));
                if already_responded {
                    debug!(
                        "Dropping duplicate response from {} for query {}",
                        responder, response.request_id
                    );
                    return;
                }
            }

            pending.responses.push(response.clone());

            // Check if we've received all expected responses
            if pending.responses.len() >= pending.expected_peers.len()
                || Instant::now() >= pending.deadline
            {
                if let Some(pending) = self.pending_queries.remove(&response.request_id) {
                    let _ = pending.response_tx.send(pending.responses);
                }
            }
        }
    }

    async fn handle_command(&mut self, cmd: NetworkCommand) {
        match cmd {
            NetworkCommand::Connect(addr) => {
                if let Err(e) = self.swarm.dial(addr.clone()) {
                    warn!("Failed to dial {}: {}", addr, e);
                }
            }
            NetworkCommand::Disconnect(peer_id) => {
                let _ = self.swarm.disconnect_peer_id(peer_id);
            }
            NetworkCommand::SendQuery {
                peers,
                request,
                response_tx,
                timeout,
            } => {
                debug!("Sending query {} to {} peers", request.request_id, peers.len());

                // Store pending query
                self.pending_queries.insert(
                    request.request_id.clone(),
                    PendingQuery {
                        response_tx,
                        responses: Vec::new(),
                        expected_peers: peers.clone(),
                        deadline: Instant::now() + timeout,
                    },
                );

                if !request.target_peers.is_empty() {
                    // Targeted query: use request-response for direct delivery
                    debug!(
                        "Using request-response for targeted query {} to {} peers",
                        request.request_id, peers.len()
                    );
                    for peer_id in &peers {
                        let outbound_id = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_request(peer_id, request.clone());
                        self.outbound_request_map
                            .insert(outbound_id, request.request_id.clone());
                    }
                } else {
                    // Broadcast query: use GossipSub
                    let msg = NetworkMessage::QueryRequest(request.clone());
                    if let Ok(data) = bincode::serialize(&msg) {
                        let topic = IdentTopic::new(topics::QUERIES);
                        if let Err(e) = self
                            .swarm
                            .behaviour_mut()
                            .gossipsub
                            .publish(topic, data)
                        {
                            warn!("Failed to publish query: {}", e);
                            if let Some(pending) = self.pending_queries.remove(&request.request_id) {
                                let _ = pending.response_tx.send(Vec::new());
                            }
                        }
                    }
                }
            }
            NetworkCommand::SendQueryResponse(response) => {
                // Try to respond via request-response channel first (direct delivery)
                if let Some((channel, _created)) = self
                    .pending_response_channels
                    .remove(&response.request_id)
                {
                    debug!(
                        "Sending direct response for query {}",
                        response.request_id
                    );
                    if self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, response.clone())
                        .is_err()
                    {
                        warn!(
                            "Failed to send direct response for query {} (channel closed), falling back to GossipSub",
                            response.request_id
                        );
                        // Fall back to GossipSub
                        self.publish_gossipsub_response(&response);
                    }
                } else {
                    // No direct channel — query came via GossipSub, respond via GossipSub
                    self.publish_gossipsub_response(&response);
                }
            }
            NetworkCommand::BroadcastAdvertisement(advert) => {
                let msg = NetworkMessage::Advertisement(advert);
                if let Ok(data) = bincode::serialize(&msg) {
                    let topic = IdentTopic::new(topics::ADVERTISEMENTS);
                    if let Err(e) = self
                        .swarm
                        .behaviour_mut()
                        .gossipsub
                        .publish(topic, data)
                    {
                        warn!("Failed to publish advertisement: {}", e);
                    }
                }
            }
            NetworkCommand::GetPeers(response_tx) => {
                let peers: Vec<PeerInfo> = self.connected_peers.read().values().cloned().collect();
                let _ = response_tx.send(peers);
            }
            NetworkCommand::Shutdown => {
                // Handled in main loop
            }
        }
    }

    /// Check for timed-out queries, stale response channels, and expired advertisements
    fn check_query_timeouts(&mut self) {
        let now = Instant::now();

        // Complete timed-out queries with whatever responses we have
        let timed_out: Vec<String> = self
            .pending_queries
            .iter()
            .filter(|(_, q)| now >= q.deadline)
            .map(|(id, _)| id.clone())
            .collect();

        for request_id in timed_out {
            if let Some(pending) = self.pending_queries.remove(&request_id) {
                debug!(
                    "Query {} timed out with {} responses",
                    request_id,
                    pending.responses.len()
                );
                let _ = pending.response_tx.send(pending.responses);
            }
        }

        // Clean up stale response channels (application never sent a response)
        let stale_threshold = Duration::from_secs(STALE_RESPONSE_CHANNEL_SECS);
        let stale_channels: Vec<String> = self
            .pending_response_channels
            .iter()
            .filter(|(_, (_, created))| now.duration_since(*created) > stale_threshold)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &stale_channels {
            self.pending_response_channels.remove(id);
        }
        if !stale_channels.is_empty() {
            debug!("Cleaned up {} stale response channels", stale_channels.len());
        }

        // Enforce max pending response channels by removing oldest
        if self.pending_response_channels.len() > MAX_PENDING_RESPONSE_CHANNELS {
            let mut entries: Vec<(String, Instant)> = self
                .pending_response_channels
                .iter()
                .map(|(id, (_, created))| (id.clone(), *created))
                .collect();
            entries.sort_by_key(|(_, created)| *created);
            let to_remove = self.pending_response_channels.len() - MAX_PENDING_RESPONSE_CHANNELS;
            for (id, _) in entries.into_iter().take(to_remove) {
                self.pending_response_channels.remove(&id);
            }
            warn!("Evicted {} response channels exceeding max capacity", to_remove);
        }

        // Clean up orphaned outbound request mappings (no matching pending query)
        let orphaned: Vec<request_response::OutboundRequestId> = self
            .outbound_request_map
            .iter()
            .filter(|(_, req_id)| !self.pending_queries.contains_key(*req_id))
            .map(|(outbound_id, _)| *outbound_id)
            .collect();
        for id in &orphaned {
            self.outbound_request_map.remove(id);
        }
        if !orphaned.is_empty() {
            debug!("Cleaned up {} orphaned outbound request mappings", orphaned.len());
        }

        // Expire stale node advertisements
        let expiry = chrono::Duration::seconds(ADVERTISEMENT_EXPIRY_SECS);
        let cutoff = chrono::Utc::now() - expiry;
        let mut expired_ads = Vec::new();
        {
            let ads = self.node_advertisements.read();
            for (peer_id, ad) in ads.iter() {
                if ad.last_updated < cutoff {
                    expired_ads.push(*peer_id);
                }
            }
        }
        if !expired_ads.is_empty() {
            let mut ads = self.node_advertisements.write();
            for peer_id in &expired_ads {
                ads.remove(peer_id);
            }
            debug!("Expired {} stale node advertisements", expired_ads.len());
        }

        // Enforce max advertisement count by removing oldest
        {
            let ads = self.node_advertisements.read();
            if ads.len() > MAX_NODE_ADVERTISEMENTS {
                drop(ads);
                let mut ads = self.node_advertisements.write();
                let mut entries: Vec<(PeerId, chrono::DateTime<chrono::Utc>)> = ads
                    .iter()
                    .map(|(pid, ad)| (*pid, ad.last_updated))
                    .collect();
                entries.sort_by_key(|(_, ts)| *ts);
                let to_remove = ads.len() - MAX_NODE_ADVERTISEMENTS;
                for (pid, _) in entries.into_iter().take(to_remove) {
                    ads.remove(&pid);
                }
                warn!("Evicted {} advertisements exceeding max capacity", to_remove);
            }
        }
    }
}

// Re-export necessary libp2p types
use libp2p::{noise, yamux};
use libp2p::mdns;
