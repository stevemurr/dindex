//! Client Connection
//!
//! Handles Unix socket connection to the daemon.

use std::path::PathBuf;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tracing::debug;

use crate::daemon::protocol::{
    decode_message, encode_message, Request, Response, MAX_MESSAGE_SIZE,
};
use crate::daemon::server::IpcServer;

use super::ClientError;

/// Client connection to the daemon
pub struct DaemonClient {
    stream: UnixStream,
}

impl DaemonClient {
    /// Connect to the daemon
    pub async fn connect() -> Result<Self, ClientError> {
        Self::connect_to(IpcServer::default_socket_path()).await
    }

    /// Connect to the daemon at a specific socket path
    pub async fn connect_to(socket_path: PathBuf) -> Result<Self, ClientError> {
        debug!("Connecting to daemon at: {}", socket_path.display());

        let stream = UnixStream::connect(&socket_path)
            .await
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound
                    || e.kind() == std::io::ErrorKind::ConnectionRefused
                {
                    ClientError::DaemonNotRunning
                } else {
                    ClientError::ConnectionFailed(e.to_string())
                }
            })?;

        debug!("Connected to daemon");
        Ok(Self { stream })
    }

    /// Send a request and receive a response
    pub async fn send(&mut self, request: Request) -> Result<Response, ClientError> {
        // Encode and send request
        let encoded = encode_message(&request)
            .map_err(|e| ClientError::RequestFailed(format!("Failed to encode request: {}", e)))?;

        self.stream
            .write_all(&encoded)
            .await
            .map_err(|e| ClientError::RequestFailed(format!("Failed to send request: {}", e)))?;

        self.stream
            .flush()
            .await
            .map_err(|e| ClientError::RequestFailed(format!("Failed to flush: {}", e)))?;

        // Read response length
        let mut len_buf = [0u8; 4];
        self.stream
            .read_exact(&mut len_buf)
            .await
            .map_err(|e| ClientError::RequestFailed(format!("Failed to read response length: {}", e)))?;

        let msg_len = u32::from_le_bytes(len_buf) as usize;
        if msg_len > MAX_MESSAGE_SIZE {
            return Err(ClientError::RequestFailed(format!(
                "Response too large: {} bytes",
                msg_len
            )));
        }

        // Read response payload
        let mut payload = vec![0u8; msg_len];
        self.stream
            .read_exact(&mut payload)
            .await
            .map_err(|e| ClientError::RequestFailed(format!("Failed to read response: {}", e)))?;

        // Decode response
        let response: Response = decode_message(&payload)
            .map_err(|e| ClientError::RequestFailed(format!("Failed to decode response: {}", e)))?;

        Ok(response)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connect_when_not_running() {
        // Should fail gracefully when daemon is not running
        let result = DaemonClient::connect().await;
        assert!(matches!(result, Err(ClientError::DaemonNotRunning)));
    }
}
