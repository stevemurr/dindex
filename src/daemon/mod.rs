//! Daemon Module
//!
//! Single daemon architecture for DIndex. The daemon owns all indexes and
//! provides IPC access for CLI commands.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      dindex daemon                           │
//! │                                                              │
//! │  ┌──────────────────────────────────────────────────────┐   │
//! │  │                   Index Manager                       │   │
//! │  │  ┌─────────────┐ ┌─────────────┐ ┌────────────────┐  │   │
//! │  │  │ USearch     │ │ Tantivy     │ │ Chunk Storage  │  │   │
//! │  │  │ (Vector)    │ │ (BM25)      │ │ (Registry)     │  │   │
//! │  │  └─────────────┘ └─────────────┘ └────────────────┘  │   │
//! │  └──────────────────────────┬───────────────────────────┘   │
//! │                             │                                │
//! │  ┌──────────────────────────┴───────────────────────────┐   │
//! │  │                   Write Pipeline                      │   │
//! │  │    ┌─────────┐    ┌─────────┐    ┌──────────────┐    │   │
//! │  │    │ Ingest  │───▶│ Batch   │───▶│ Commit       │    │   │
//! │  │    │ Queue   │    │ Writer  │    │ Controller   │    │   │
//! │  │    └─────────┘    └─────────┘    └──────────────┘    │   │
//! │  └──────────────────────────────────────────────────────┘   │
//! │                                                              │
//! │  ┌────────────────┐                                         │
//! │  │ Unix Socket    │ ◀─── IPC from CLI                       │
//! │  │ (Local IPC)    │                                         │
//! │  └────────────────┘                                         │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! Start the daemon:
//! ```bash
//! dindex daemon start
//! ```
//!
//! Check status:
//! ```bash
//! dindex daemon status
//! ```
//!
//! Stop the daemon:
//! ```bash
//! dindex daemon stop
//! ```

pub mod handler;
pub mod http;
pub mod index_manager;
pub mod jobs;
pub mod lifecycle;
pub mod metrics;
pub mod protocol;
pub mod recovery;
pub mod server;
pub mod write_pipeline;

pub use handler::RequestHandler;
pub use index_manager::IndexManager;
pub use jobs::JobManager;
pub use lifecycle::{get_daemon_pid, is_daemon_running, Daemon};
pub use metrics::{DaemonMetrics, MetricsSnapshot, Timer};
pub use protocol::{
    decode_message, encode_message, ChunkPayload, DaemonStatus, ErrorCode, ImportOptions,
    ImportSource, IndexStats, JobStats, OutputFormat, Progress, Request, Response, ScrapeOptions,
};
pub use recovery::{RecoveryManager, RecoveryResult, RecoveryState};
pub use server::IpcServer;
pub use write_pipeline::WritePipeline;
pub use http::HttpServer;
