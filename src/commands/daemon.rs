use anyhow::Result;
use dindex::{
    client::{self, ClientError},
    config::Config,
    daemon::{self, Daemon},
    network::{NetworkEvent, NetworkNode, QueryResponse},
    query::{QueryCoordinator, QueryExecutor},
    routing::{AdvertisementBuilder, QueryRouter},
    util::truncate_str,
};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

use crate::DaemonAction;

/// Handle daemon management commands
pub async fn handle_daemon(config: Config, action: DaemonAction) -> Result<()> {
    match action {
        DaemonAction::Start { foreground: _ } => {
            // Daemonization already handled in main() before tokio started
            if daemon::is_daemon_running(&config.node.data_dir) {
                println!("Daemon is already running");
                return Ok(());
            }

            // Start the daemon (we're either in foreground or already daemonized)
            let daemon = Daemon::start(config).await?;
            daemon.run().await?;
            Ok(())
        }
        DaemonAction::Stop => {
            if !daemon::is_daemon_running(&config.node.data_dir) {
                println!("Daemon is not running");
                return Ok(());
            }

            println!("Stopping daemon...");
            match client::shutdown().await {
                Ok(()) => println!("Daemon stopped"),
                Err(ClientError::DaemonNotRunning) => println!("Daemon is not running"),
                Err(e) => anyhow::bail!("Failed to stop daemon: {}", e),
            }
            Ok(())
        }
        DaemonAction::Status => {
            match client::status().await {
                Ok(status) => {
                    println!("Daemon Status:");
                    println!("  Running: {}", status.running);
                    println!("  Uptime: {}s", status.uptime_seconds);
                    println!("  Memory: {} MB", status.memory_mb);
                    println!("  Active Jobs: {}", status.active_jobs);
                    println!("  Pending Writes: {}", status.pending_writes);
                }
                Err(ClientError::DaemonNotRunning) => {
                    println!("Daemon is not running");
                    if let Some(pid) = daemon::get_daemon_pid(&config.node.data_dir) {
                        println!("  Stale PID file found (PID {})", pid);
                    }
                }
                Err(e) => anyhow::bail!("Failed to get status: {}", e),
            }
            Ok(())
        }
        DaemonAction::Restart => {
            // Stop first if running
            if daemon::is_daemon_running(&config.node.data_dir) {
                println!("Stopping daemon...");
                let _ = client::shutdown().await;
                tokio::time::sleep(Duration::from_secs(1)).await;
            }

            println!("Starting daemon...");
            let daemon = Daemon::start(config).await?;
            daemon.run().await?;
            Ok(())
        }
    }
}

pub async fn start_node_inner(
    mut config: Config,
    listen: Option<String>,
    bootstrap: Vec<String>,
) -> Result<()> {
    if let Some(addr) = listen {
        config.node.listen_addr = addr;
    }
    config.node.bootstrap_peers.extend(bootstrap);

    info!("Starting DIndex node (daemon + P2P)...");
    info!("Data directory: {}", config.node.data_dir.display());

    // Start daemon first
    let daemon = Daemon::start(config.clone()).await?;
    info!("Daemon started");

    // Create QueryExecutor for handling incoming peer queries
    let query_executor = Arc::new(QueryExecutor::new(
        daemon.index_manager().retriever(),
        daemon.index_manager().embedding_engine(),
    ));

    // Initialize P2P components
    let (node, handle, mut event_rx) = NetworkNode::new(&config.node).await?;

    info!("Node started with peer ID: {}", handle.local_peer_id);

    // Create QueryRouter for semantic routing
    let query_router = Arc::new(QueryRouter::new(
        config.embedding.dimensions,
        &config.routing,
    ));

    // Load persisted routing state (node advertisements from previous sessions)
    let routing_state_path = config.node.data_dir.join("routing_state.json");
    match query_router.load(&routing_state_path) {
        Ok(count) if count > 0 => info!("Restored {} node advertisements from previous session", count),
        Ok(_) => {}
        Err(e) => warn!("Failed to load routing state: {} (starting fresh)", e),
    }

    let query_router_for_events = query_router.clone();
    let query_router_for_advert = query_router.clone();
    let query_router_for_shutdown = query_router.clone();
    let routing_state_path_shutdown = routing_state_path.clone();

    // Create QueryCoordinator for distributed search
    let query_coordinator = Arc::new(QueryCoordinator::new(
        Some(daemon.index_manager().retriever()),
        daemon.index_manager().embedding_engine(),
        query_router,
        Some(handle.clone()),
        config.clone(),
    ));

    // Set the coordinator on the request handler for distributed search
    daemon.request_handler().set_query_coordinator(query_coordinator);
    info!("Distributed search enabled via QueryCoordinator");

    // Run the P2P node in background
    let node_config = config.node.clone();
    tokio::spawn(async move {
        if let Err(e) = node.run(&node_config).await {
            tracing::error!("Node error: {}", e);
        }
    });

    // Build and broadcast our node advertisement
    let embeddings = daemon.index_manager().all_embeddings();
    if !embeddings.is_empty() {
        // Generate LSH signatures for bloom filter optimization
        let lsh_sigs: Vec<dindex::types::LshSignature> = embeddings.iter()
            .map(|emb| query_router_for_advert.hash_query(emb))
            .collect();

        let advertisement = AdvertisementBuilder::new(handle.local_peer_id.to_string())
            .with_centroids(&embeddings, config.routing.num_centroids, Some(config.embedding.truncated_dimensions))
            .with_lsh(lsh_sigs)
            .build(
                config.routing.lsh_bits,
                config.routing.lsh_num_bands,
                config.routing.bloom_false_positive_rate,
            );
        info!(
            "Broadcasting advertisement: {} centroids, {} chunks",
            advertisement.centroids.len(),
            advertisement.total_chunks
        );
        // Delay briefly to allow connections to establish
        let advert_handle = handle.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            if let Err(e) = advert_handle.broadcast_advertisement(advertisement).await {
                warn!("Failed to broadcast advertisement: {}", e);
            }
        });
    } else {
        info!("No indexed content, skipping advertisement broadcast");
    }

    // Spawn network event handler for incoming queries
    let network_handle = handle.clone();
    let executor = query_executor.clone();
    let local_peer_id = handle.local_peer_id.to_string();
    let p2p_metrics = daemon.request_handler().metrics().clone();
    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            match event {
                NetworkEvent::QueryReceived { peer_id, request } => {
                    info!(
                        "Received query from {}: '{}'",
                        peer_id,
                        truncate_str(&request.query.text, 50)
                    );
                    p2p_metrics.p2p_queries_received.inc();

                    // Execute the query locally
                    match executor.execute_request(&request) {
                        Ok(result) => {
                            info!(
                                "Query executed: {} results in {}ms",
                                result.results.len(),
                                result.processing_time_ms
                            );

                            // Send response back
                            let response = QueryResponse::new(
                                request.request_id.clone(),
                                result.results,
                            )
                            .with_timing(result.processing_time_ms)
                            .with_responder(local_peer_id.clone());

                            if let Err(e) = network_handle.send_response(response).await {
                                tracing::error!("Failed to send query response: {}", e);
                            }
                        }
                        Err(e) => {
                            tracing::error!("Query execution failed: {}", e);
                            // Send empty response on error
                            let response = QueryResponse::new(request.request_id.clone(), vec![])
                                .with_responder(local_peer_id.clone());
                            let _ = network_handle.send_response(response).await;
                        }
                    }
                }
                NetworkEvent::PeerConnected(peer_id) => {
                    info!("Peer connected: {}", peer_id);
                    p2p_metrics.p2p_connected_peers.inc();
                }
                NetworkEvent::PeerDisconnected(peer_id) => {
                    info!("Peer disconnected: {}", peer_id);
                    p2p_metrics.p2p_connected_peers.dec();
                }
                NetworkEvent::AdvertisementReceived(advert) => {
                    info!(
                        "Received advertisement from node {} with {} centroids, {} chunks",
                        advert.node_id,
                        advert.centroids.len(),
                        advert.total_chunks
                    );
                    query_router_for_events.register_node(advert);
                }
            }
        }
    });

    // Run daemon (this blocks until shutdown)
    daemon.run().await?;

    // Save routing state for next startup
    if query_router_for_shutdown.node_count() > 0 {
        if let Err(e) = query_router_for_shutdown.save(&routing_state_path_shutdown) {
            warn!("Failed to save routing state: {}", e);
        }
    }

    Ok(())
}
