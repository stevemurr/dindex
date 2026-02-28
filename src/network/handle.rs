//! Network handle and public types for interacting with the network node

use super::messages::{QueryRequest, QueryResponse};
use crate::types::{NodeAdvertisement, Query};

use anyhow::Result;
use libp2p::{Multiaddr, PeerId};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

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
    pub(super) command_tx: mpsc::Sender<NetworkCommand>,
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
