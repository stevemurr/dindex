//! IPC Server
//!
//! Unix domain socket server for client-daemon communication.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use super::handler::RequestHandler;
use super::protocol::{decode_message, encode_message, Request, Response, MAX_MESSAGE_SIZE};

/// IPC server listening on a Unix domain socket
pub struct IpcServer {
    socket_path: PathBuf,
    handler: Arc<RequestHandler>,
}

impl IpcServer {
    /// Create a new IPC server
    pub fn new(socket_path: PathBuf, handler: Arc<RequestHandler>) -> Self {
        Self {
            socket_path,
            handler,
        }
    }

    /// Get the default socket path
    pub fn default_socket_path() -> PathBuf {
        // Try XDG_RUNTIME_DIR first (Linux standard)
        if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
            let socket_dir = PathBuf::from(runtime_dir).join("dindex");
            return socket_dir.join("dindex.sock");
        }

        // Fall back to /tmp
        PathBuf::from("/tmp/dindex.sock")
    }

    /// Run the IPC server
    pub async fn run(&self, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.socket_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Remove existing socket file if it exists
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path)
                .context("Failed to remove existing socket file")?;
        }

        // Bind to socket
        let listener = UnixListener::bind(&self.socket_path)
            .context("Failed to bind to Unix socket")?;

        info!("IPC server listening on: {}", self.socket_path.display());

        // Set permissions (readable/writable by owner and group)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.socket_path, std::fs::Permissions::from_mode(0o660))?;
        }

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, _addr)) => {
                            debug!("New client connection");
                            let handler = self.handler.clone();
                            tokio::spawn(async move {
                                if let Err(e) = handle_connection(stream, handler).await {
                                    warn!("Connection error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            error!("Accept error: {}", e);
                        }
                    }
                }
                _ = shutdown.recv() => {
                    info!("IPC server shutting down");
                    break;
                }
            }
        }

        // Clean up socket file
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }

        Ok(())
    }

    /// Get the socket path
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

/// Handle a single client connection
async fn handle_connection(mut stream: UnixStream, handler: Arc<RequestHandler>) -> Result<()> {
    loop {
        // Read message length (4 bytes, little-endian)
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // Client disconnected
                debug!("Client disconnected");
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        }

        let msg_len = u32::from_le_bytes(len_buf) as usize;
        if msg_len > MAX_MESSAGE_SIZE {
            error!("Message too large: {} bytes", msg_len);
            let response = Response::error(
                super::protocol::ErrorCode::InvalidRequest,
                format!("Message too large: {} bytes", msg_len),
            );
            send_response(&mut stream, &response).await?;
            continue;
        }

        // Read message payload
        let mut payload = vec![0u8; msg_len];
        stream.read_exact(&mut payload).await?;

        // Decode request
        let request: Request = match decode_message(&payload) {
            Ok(req) => req,
            Err(e) => {
                error!("Failed to decode request: {}", e);
                let response = Response::error(
                    super::protocol::ErrorCode::InvalidRequest,
                    format!("Failed to decode request: {}", e),
                );
                send_response(&mut stream, &response).await?;
                continue;
            }
        };

        // Handle request
        let response = handler.handle(request).await;

        // Send response
        send_response(&mut stream, &response).await?;

        // Check if this was a shutdown request
        if matches!(response, Response::Ok) {
            // Give a moment for the response to be sent
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }
}

/// Send a response to the client
async fn send_response(stream: &mut UnixStream, response: &Response) -> Result<()> {
    let encoded = encode_message(response)?;
    stream.write_all(&encoded).await?;
    stream.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_socket_path() {
        let path = IpcServer::default_socket_path();
        assert!(path.to_string_lossy().contains("dindex"));
    }
}
