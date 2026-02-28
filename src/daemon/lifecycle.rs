//! Daemon Lifecycle Management
//!
//! Handles daemon startup, shutdown, and single-instance guarantees.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::signal;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

use crate::config::{Config, EmbeddingConfig};
use crate::embedding::EmbeddingEngine;

use super::handler::RequestHandler;
use super::http::HttpServer;
use super::index_manager::IndexManager;
use super::metrics::DaemonMetrics;
use super::recovery::{RecoveryManager, RecoveryResult};
use super::server::IpcServer;
use super::write_pipeline::WritePipeline;

/// PID file for single-instance guarantee
const PID_FILE_NAME: &str = "dindex.pid";

/// Daemon instance managing all components
pub struct Daemon {
    config: Config,
    index_manager: Arc<IndexManager>,
    handler: Arc<RequestHandler>,
    server: IpcServer,
    shutdown_tx: broadcast::Sender<()>,
    pid_file_path: PathBuf,
}

impl Daemon {
    /// Start the daemon
    pub async fn start(config: Config) -> Result<Self> {
        info!("Starting DIndex daemon");

        // Acquire single-instance lock
        let pid_file_path = config.node.data_dir.join(PID_FILE_NAME);
        Self::acquire_lock(&pid_file_path)?;

        // Initialize embedding engine in a blocking task with timeout
        // This prevents blocking the async runtime during model loading
        let embedding_config = config.embedding.clone();
        let embedding_engine = match Self::init_embedding_engine(embedding_config).await {
            Ok(engine) => {
                info!(
                    "Embedding engine initialized: {} ({}D, GPU: {})",
                    config.embedding.model_name,
                    config.embedding.dimensions,
                    config.embedding.use_gpu
                );
                Some(Arc::new(engine))
            }
            Err(e) => {
                warn!("Failed to initialize embedding engine: {}. Embeddings will use fallback.", e);
                warn!("Check your [embedding] config in dindex.toml (backend, endpoint, model).");
                None
            }
        };

        // Run crash recovery before loading indices
        let data_dir = &config.node.data_dir;
        let mut recovery = RecoveryManager::new(data_dir.to_path_buf())?;
        match recovery.recover()? {
            RecoveryResult::NoRecoveryNeeded => info!("Clean shutdown detected, no recovery needed"),
            RecoveryResult::RolledBackWrite { stream_id, chunks } => {
                warn!("Recovered from interrupted write: rolled back stream {} ({} chunks)", stream_id, chunks);
            }
            RecoveryResult::CompletedCommit => info!("Completed interrupted commit during recovery"),
        }

        // Initialize index manager
        let index_manager = Arc::new(
            IndexManager::load(&config).context("Failed to load index manager")?,
        );

        // Set embedding engine on index manager for search queries
        if let Some(ref engine) = embedding_engine {
            index_manager.set_embedding_engine(engine.clone());
        }

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = broadcast::channel(16);

        // Create daemon metrics
        let metrics = DaemonMetrics::shared();

        // Start write pipeline with metrics
        let write_pipeline = Arc::new(WritePipeline::start_with_metrics(
            index_manager.clone(),
            embedding_engine.clone(),
            config.bulk_import.batch_size,
            Duration::from_secs(30), // Commit every 30 seconds
            shutdown_rx,
            Some(metrics.clone()),
        ));

        // Create request handler
        let handler = Arc::new(RequestHandler::new(
            index_manager.clone(),
            write_pipeline.clone(),
            config.clone(),
            shutdown_tx.clone(),
            metrics,
        ));

        // Create IPC server
        let socket_path = IpcServer::default_socket_path();
        let server = IpcServer::new(socket_path, handler.clone());

        info!("Daemon initialized");
        info!("Data directory: {}", config.node.data_dir.display());
        info!("Socket path: {}", server.socket_path().display());

        Ok(Self {
            config,
            index_manager,
            handler,
            server,
            shutdown_tx,
            pid_file_path,
        })
    }

    /// Run the daemon (blocking)
    pub async fn run(&self) -> Result<()> {
        info!("Daemon running");

        // Subscribe to shutdown signal
        let shutdown_rx = self.shutdown_tx.subscribe();

        // Spawn periodic memory usage update
        let metrics_for_memory = self.handler.metrics().clone();
        let mut shutdown_rx_memory = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(15));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        metrics_for_memory.update_memory_usage();
                    }
                    _ = shutdown_rx_memory.recv() => break,
                }
            }
        });

        // Start IPC server - run directly, not spawned, to catch errors
        let shutdown_rx_server = self.shutdown_tx.subscribe();
        let socket_path = self.server.socket_path().to_path_buf();
        let handler = self.handler.clone();

        info!("Starting IPC server on: {}", socket_path.display());

        // Ensure socket directory exists
        if let Some(parent) = socket_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                error!("Failed to create socket directory: {}", e);
            }
        }

        let server = IpcServer::new(socket_path.clone(), handler.clone());
        let server_handle = tokio::spawn(async move {
            info!("IPC server task starting...");
            match server.run(shutdown_rx_server).await {
                Ok(()) => info!("IPC server shut down cleanly"),
                Err(e) => error!("IPC server failed: {}", e),
            }
        });

        // Start HTTP server if enabled
        let http_handle = if self.config.http.enabled {
            let http_config = self.config.http.clone();
            let http_handler = self.handler.clone();
            let shutdown_rx_http = self.shutdown_tx.subscribe();

            info!("Starting HTTP API server on: {}", http_config.listen_addr);

            let http_server = HttpServer::new(http_config, http_handler);
            Some(tokio::spawn(async move {
                info!("HTTP server task starting...");
                match http_server.run(shutdown_rx_http).await {
                    Ok(()) => info!("HTTP server shut down cleanly"),
                    Err(e) => error!("HTTP server failed: {}", e),
                }
            }))
        } else {
            None
        };

        // Wait for shutdown signal (Ctrl+C or SIGTERM)
        tokio::select! {
            _ = signal::ctrl_c() => {
                info!("Received Ctrl+C, shutting down");
            }
            _ = Self::wait_for_sigterm() => {
                info!("Received SIGTERM, shutting down");
            }
            _ = Self::wait_for_shutdown(shutdown_rx) => {
                info!("Shutdown requested via IPC");
            }
        }

        // Trigger shutdown
        let _ = self.shutdown_tx.send(());

        // Wait for servers to stop, aborting if they don't shut down in time
        let server_abort = server_handle.abort_handle();
        if tokio::time::timeout(Duration::from_secs(5), server_handle).await.is_err() {
            warn!("IPC server did not shut down within 5s, aborting");
            server_abort.abort();
        }
        if let Some(http_handle) = http_handle {
            let http_abort = http_handle.abort_handle();
            if tokio::time::timeout(Duration::from_secs(5), http_handle).await.is_err() {
                warn!("HTTP server did not shut down within 5s, aborting");
                http_abort.abort();
            }
        }

        // Final cleanup
        self.shutdown().await?;

        Ok(())
    }

    /// Shutdown the daemon gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down daemon");

        // Commit any pending changes
        if let Err(e) = self.index_manager.commit() {
            warn!("Failed to commit during shutdown: {}", e);
        }

        // Release lock
        Self::release_lock(&self.pid_file_path)?;

        info!("Daemon shutdown complete");
        Ok(())
    }

    /// Get the index manager
    pub fn index_manager(&self) -> Arc<IndexManager> {
        self.index_manager.clone()
    }

    /// Get the request handler
    pub fn request_handler(&self) -> Arc<RequestHandler> {
        self.handler.clone()
    }

    /// Initialize the embedding engine in a blocking task
    ///
    /// This runs model loading in spawn_blocking to avoid blocking the async runtime.
    /// Model loading can take a long time (downloading from HuggingFace, loading weights).
    async fn init_embedding_engine(config: EmbeddingConfig) -> Result<EmbeddingEngine> {
        info!("Loading embedding model: {} (this may take a moment...)", config.model_name);

        // Use spawn_blocking since EmbeddingEngine::new() does blocking I/O
        // (model download from HuggingFace, file I/O for loading weights)
        let result = tokio::task::spawn_blocking(move || {
            EmbeddingEngine::new(&config)
        })
        .await
        .context("Embedding engine task panicked")?;

        result
    }

    /// Acquire single-instance lock via PID file
    ///
    /// Uses `create_new(true)` for atomic creation to avoid TOCTOU races
    /// where two processes could interleave exists/read/remove/create.
    fn acquire_lock(pid_file_path: &Path) -> Result<()> {
        use std::fs::OpenOptions;

        // Try atomic create — fails if file already exists
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(pid_file_path)
        {
            Ok(mut file) => {
                writeln!(file, "{}", std::process::id())?;
                return Ok(());
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // File exists — check if it's stale below
            }
            Err(e) => {
                return Err(e).context("Failed to create PID file");
            }
        }

        // PID file exists — check if the process is still running
        let mut file = File::open(pid_file_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        if let Ok(pid) = contents.trim().parse::<u32>() {
            if Self::process_exists(pid) {
                anyhow::bail!(
                    "Daemon is already running (PID {}). Stop it first or remove {}",
                    pid,
                    pid_file_path.display()
                );
            }
        }

        // Stale PID file — remove and retry once
        std::fs::remove_file(pid_file_path)?;

        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(pid_file_path)
        {
            Ok(mut file) => {
                writeln!(file, "{}", std::process::id())?;
                Ok(())
            }
            Err(e) => Err(e).context("Failed to create PID file after removing stale lock"),
        }
    }

    /// Release single-instance lock
    fn release_lock(pid_file_path: &Path) -> Result<()> {
        if pid_file_path.exists() {
            std::fs::remove_file(pid_file_path)?;
        }
        Ok(())
    }

    /// Check if a process with the given PID exists
    fn process_exists(pid: u32) -> bool {
        #[cfg(unix)]
        {
            // On Unix, use kill(pid, 0) to check if process exists
            // Signal 0 doesn't actually send a signal, just checks if the process exists
            unsafe {
                libc::kill(pid as i32, 0) == 0
            }
        }

        #[cfg(not(unix))]
        {
            // On other platforms, assume process exists if we can't check
            let _ = pid;
            true
        }
    }

    /// Wait for SIGTERM signal
    #[cfg(unix)]
    async fn wait_for_sigterm() {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut sigterm) => {
                sigterm.recv().await;
            }
            Err(e) => {
                tracing::warn!("Failed to register SIGTERM handler: {}. Falling back to pending future.", e);
                std::future::pending::<()>().await;
            }
        }
    }

    #[cfg(not(unix))]
    async fn wait_for_sigterm() {
        // On non-Unix, just wait forever (Ctrl+C will still work)
        std::future::pending::<()>().await
    }

    /// Wait for shutdown signal from broadcast channel
    async fn wait_for_shutdown(mut rx: broadcast::Receiver<()>) {
        let _ = rx.recv().await;
    }
}

/// Check if the daemon is running
pub fn is_daemon_running(data_dir: &Path) -> bool {
    let pid_file_path = data_dir.join(PID_FILE_NAME);

    if !pid_file_path.exists() {
        return false;
    }

    if let Ok(mut file) = File::open(&pid_file_path) {
        let mut contents = String::new();
        if file.read_to_string(&mut contents).is_ok() {
            if let Ok(pid) = contents.trim().parse::<u32>() {
                return Daemon::process_exists(pid);
            }
        }
    }

    false
}

/// Get the PID of the running daemon, if any
pub fn get_daemon_pid(data_dir: &Path) -> Option<u32> {
    let pid_file_path = data_dir.join(PID_FILE_NAME);

    if !pid_file_path.exists() {
        return None;
    }

    File::open(&pid_file_path)
        .ok()
        .and_then(|mut file| {
            let mut contents = String::new();
            file.read_to_string(&mut contents).ok()?;
            contents.trim().parse().ok()
        })
        .filter(|&pid| Daemon::process_exists(pid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_pid_lock_acquire_release() {
        let temp_dir = TempDir::new().unwrap();
        let pid_path = temp_dir.path().join("test.pid");

        // Acquire lock
        Daemon::acquire_lock(&pid_path).unwrap();
        assert!(pid_path.exists());

        // Release lock
        Daemon::release_lock(&pid_path).unwrap();
        assert!(!pid_path.exists());
    }

    #[test]
    fn test_is_daemon_running() {
        let temp_dir = TempDir::new().unwrap();
        assert!(!is_daemon_running(temp_dir.path()));
    }
}
