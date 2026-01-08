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

use crate::config::Config;

use super::handler::RequestHandler;
use super::index_manager::IndexManager;
use super::server::IpcServer;
use super::write_pipeline::WritePipeline;

/// PID file for single-instance guarantee
const PID_FILE_NAME: &str = "dindex.pid";

/// Daemon instance managing all components
pub struct Daemon {
    config: Config,
    index_manager: Arc<IndexManager>,
    write_pipeline: Arc<WritePipeline>,
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

        // Initialize index manager
        let index_manager = Arc::new(
            IndexManager::load(&config).context("Failed to load index manager")?,
        );

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = broadcast::channel(16);

        // Start write pipeline
        let write_pipeline = Arc::new(WritePipeline::start(
            index_manager.clone(),
            config.bulk_import.batch_size,
            Duration::from_secs(30), // Commit every 30 seconds
            shutdown_rx,
        ));

        // Create request handler
        let handler = Arc::new(RequestHandler::new(
            index_manager.clone(),
            write_pipeline.clone(),
            config.clone(),
            shutdown_tx.clone(),
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
            write_pipeline,
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

        // Start IPC server
        let server_handle = {
            let shutdown_rx = self.shutdown_tx.subscribe();
            let socket_path = self.server.socket_path().to_path_buf();
            let handler = self.handler.clone();

            let server = IpcServer::new(socket_path, handler);
            tokio::spawn(async move {
                if let Err(e) = server.run(shutdown_rx).await {
                    error!("IPC server error: {}", e);
                }
            })
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

        // Wait for server to stop
        let _ = tokio::time::timeout(Duration::from_secs(5), server_handle).await;

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

    /// Acquire single-instance lock via PID file
    fn acquire_lock(pid_file_path: &Path) -> Result<()> {
        // Check if daemon is already running
        if pid_file_path.exists() {
            let mut file = File::open(pid_file_path)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;

            if let Ok(pid) = contents.trim().parse::<u32>() {
                // Check if process is still running
                if Self::process_exists(pid) {
                    anyhow::bail!(
                        "Daemon is already running (PID {}). Stop it first or remove {}",
                        pid,
                        pid_file_path.display()
                    );
                }
            }

            // Stale PID file, remove it
            std::fs::remove_file(pid_file_path)?;
        }

        // Create PID file
        let mut file = File::create(pid_file_path)?;
        writeln!(file, "{}", std::process::id())?;

        Ok(())
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
            // On Unix, we can use kill(pid, 0) to check if process exists
            use std::os::unix::process::CommandExt;
            let result = std::process::Command::new("kill")
                .args(["-0", &pid.to_string()])
                .status();

            matches!(result, Ok(status) if status.success())
        }

        #[cfg(not(unix))]
        {
            // On other platforms, assume process exists if we can't check
            true
        }
    }

    /// Wait for SIGTERM signal
    #[cfg(unix)]
    async fn wait_for_sigterm() {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate()).expect("Failed to register SIGTERM");
        sigterm.recv().await;
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

    fn test_config(data_dir: &std::path::Path) -> Config {
        let mut config = Config::default();
        config.node.data_dir = data_dir.to_path_buf();
        config
    }

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
