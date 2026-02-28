//! Request-response codec for direct peer-to-peer queries
//!
//! Uses length-prefixed bincode encoding for efficient serialization.
//! This is used instead of GossipSub for targeted queries, avoiding
//! broadcast overhead when we know exactly which peers to query.

use super::messages::{QueryRequest, QueryResponse};
use async_trait::async_trait;
use futures::prelude::*;
use libp2p::StreamProtocol;
use std::io;

/// Maximum message size (16 MB) to prevent memory exhaustion
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Protocol name for direct query requests
pub const DIRECT_QUERY_PROTOCOL: &str = "/dindex/query-direct/1.0.0";

/// Codec for DIndex request-response protocol.
///
/// Uses a simple length-prefixed bincode encoding:
/// - 4 bytes (big-endian u32): message length
/// - N bytes: bincode-encoded message
#[derive(Debug, Clone, Default)]
pub struct DIndexCodec;

#[async_trait]
impl libp2p::request_response::Codec for DIndexCodec {
    type Protocol = StreamProtocol;
    type Request = QueryRequest;
    type Response = QueryResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let data = read_length_prefixed(io).await?;
        bincode::deserialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let data = read_length_prefixed(io).await?;
        bincode::deserialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let data = bincode::serialize(&req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        write_length_prefixed(io, &data).await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let data = bincode::serialize(&res)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        write_length_prefixed(io, &data).await
    }
}

/// Read a length-prefixed message from the stream
async fn read_length_prefixed<T: AsyncRead + Unpin>(io: &mut T) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    if len > MAX_MESSAGE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message too large: {} bytes (max {})", len, MAX_MESSAGE_SIZE),
        ));
    }

    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Write a length-prefixed message to the stream
async fn write_length_prefixed<T: AsyncWrite + Unpin>(
    io: &mut T,
    data: &[u8],
) -> io::Result<()> {
    if data.len() > MAX_MESSAGE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "message too large to send: {} bytes (max {})",
                data.len(),
                MAX_MESSAGE_SIZE
            ),
        ));
    }
    let len = data.len() as u32;
    io.write_all(&len.to_be_bytes()).await?;
    io.write_all(data).await?;
    io.close().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Query;
    use futures::io::Cursor as AsyncCursor;
    use libp2p::request_response::Codec as _;
    use libp2p::StreamProtocol;

    #[test]
    fn test_protocol_name() {
        assert_eq!(DIRECT_QUERY_PROTOCOL, "/dindex/query-direct/1.0.0");
    }

    #[tokio::test]
    async fn test_codec_request_roundtrip() {
        let mut codec = DIndexCodec;
        let protocol = StreamProtocol::new(DIRECT_QUERY_PROTOCOL);

        let request = QueryRequest::new(Query::new("test query", 5))
            .with_embedding(vec![0.1, 0.2, 0.3]);

        // Write request to buffer
        let mut buf = Vec::new();
        codec
            .write_request(&protocol, &mut buf, request.clone())
            .await
            .unwrap();

        // Read request back
        let mut cursor = AsyncCursor::new(buf);
        let decoded = codec.read_request(&protocol, &mut cursor).await.unwrap();

        assert_eq!(decoded.request_id, request.request_id);
        assert_eq!(decoded.query.text, "test query");
        assert_eq!(decoded.query.top_k, 5);
        assert_eq!(decoded.query_embedding, Some(vec![0.1, 0.2, 0.3]));
    }

    #[tokio::test]
    async fn test_codec_response_roundtrip() {
        let mut codec = DIndexCodec;
        let protocol = StreamProtocol::new(DIRECT_QUERY_PROTOCOL);

        let response = QueryResponse::new("req-1".to_string(), vec![])
            .with_timing(42)
            .with_responder("peer-A".to_string());

        // Write response to buffer
        let mut buf = Vec::new();
        codec
            .write_response(&protocol, &mut buf, response.clone())
            .await
            .unwrap();

        // Read response back
        let mut cursor = AsyncCursor::new(buf);
        let decoded = codec.read_response(&protocol, &mut cursor).await.unwrap();

        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.processing_time_ms, 42);
        assert_eq!(decoded.responder_peer, Some("peer-A".to_string()));
    }

    #[tokio::test]
    async fn test_read_truncated_length_prefix() {
        // Only 2 bytes instead of 4
        let mut cursor = AsyncCursor::new(vec![0x00, 0x01]);
        let result = read_length_prefixed(&mut cursor).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn test_read_truncated_payload() {
        // Length says 100 but only 10 bytes follow
        let mut buf = Vec::new();
        buf.extend_from_slice(&100u32.to_be_bytes());
        buf.extend_from_slice(&[0u8; 10]);
        let mut cursor = AsyncCursor::new(buf);
        let result = read_length_prefixed(&mut cursor).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn test_read_zero_length_message() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_be_bytes());
        let mut cursor = AsyncCursor::new(buf);
        let result = read_length_prefixed(&mut cursor).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_read_empty_stream() {
        let mut cursor = AsyncCursor::new(Vec::new());
        let result = read_length_prefixed(&mut cursor).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn test_read_invalid_bincode() {
        // Valid length prefix with garbage bincode data
        let garbage = vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB];
        let mut buf = Vec::new();
        buf.extend_from_slice(&(garbage.len() as u32).to_be_bytes());
        buf.extend_from_slice(&garbage);
        let mut cursor = AsyncCursor::new(buf);

        let mut codec = DIndexCodec;
        let protocol = StreamProtocol::new(DIRECT_QUERY_PROTOCOL);
        let result: io::Result<QueryRequest> = codec.read_request(&protocol, &mut cursor).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn test_write_rejects_oversized_message() {
        let mut buf = Vec::new();
        // Create data exceeding MAX_MESSAGE_SIZE
        let big_data = vec![0u8; MAX_MESSAGE_SIZE + 1];
        let result = write_length_prefixed(&mut buf, &big_data).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too large to send"));
    }

    #[tokio::test]
    async fn test_codec_rejects_oversized_message() {
        // Create a fake length-prefixed message claiming to be larger than MAX_MESSAGE_SIZE
        let fake_len = (MAX_MESSAGE_SIZE + 1) as u32;
        let buf = fake_len.to_be_bytes().to_vec();
        let mut cursor = AsyncCursor::new(buf);

        let mut codec = DIndexCodec;
        let protocol = StreamProtocol::new(DIRECT_QUERY_PROTOCOL);
        let result: io::Result<QueryRequest> = codec.read_request(&protocol, &mut cursor).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("message too large"));
    }
}
