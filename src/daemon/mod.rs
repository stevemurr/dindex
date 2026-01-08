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
pub mod index_manager;
pub mod lifecycle;
pub mod protocol;
pub mod server;
pub mod write_pipeline;

pub use handler::RequestHandler;
pub use index_manager::IndexManager;
pub use lifecycle::{get_daemon_pid, is_daemon_running, Daemon};
pub use protocol::{
    decode_message, encode_message, DaemonStatus, ErrorCode, IndexStats, JobStats, OutputFormat,
    Progress, Request, Response,
};
pub use server::IpcServer;
pub use write_pipeline::WritePipeline;
