# Single Daemon Architecture Implementation Plan

## Overview

This document outlines the implementation plan for transitioning DIndex from a multi-process architecture (where each CLI command directly accesses indexes) to a single daemon architecture where one long-running process owns all indexes and CLI commands communicate via IPC.

## Goals

1. **Eliminate locking conflicts** - No more Tantivy writer lock blocking concurrent operations
2. **Enable concurrent read/write** - Search while importing, indexing, or scraping
3. **Reduce memory footprint** - Single embedding model instance (~500MB) shared across operations
4. **Unify with P2P** - Daemon becomes the P2P node, CLI commands are thin clients
5. **Improve consistency** - Atomic commits, consistent snapshots for readers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      dindex daemon                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Index Manager                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌────────────────┐  │   │
│  │  │ USearch     │ │ Tantivy     │ │ Chunk Storage  │  │   │
│  │  │ (Vector)    │ │ (BM25)      │ │ (Registry)     │  │   │
│  │  └─────────────┘ └─────────────┘ └────────────────┘  │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│  ┌──────────────────────────┴───────────────────────────┐   │
│  │                   Query Engine                        │   │
│  │         (Hybrid retrieval, RRF fusion)               │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                                │
│  ┌──────────────────────────┴───────────────────────────┐   │
│  │                   Write Pipeline                      │   │
│  │    ┌─────────┐    ┌─────────┐    ┌──────────────┐    │   │
│  │    │ Ingest  │───▶│ Batch   │───▶│ Commit       │    │   │
│  │    │ Queue   │    │ Writer  │    │ Controller   │    │   │
│  │    └─────────┘    └─────────┘    └──────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐   │
│  │ Unix Socket    │ │ libp2p         │ │ HTTP API       │   │
│  │ (Local IPC)    │ │ (P2P Network)  │ │ (Optional)     │   │
│  └───────┬────────┘ └────────────────┘ └────────────────┘   │
└──────────┼──────────────────────────────────────────────────┘
           │
     ┌─────┴─────┐
     │  dindex   │  CLI (thin client)
     │  search   │
     │  index    │
     │  import   │
     │  scrape   │
     └───────────┘
```

## IPC Protocol

### Transport

- **Primary**: Unix Domain Socket at `$XDG_RUNTIME_DIR/dindex/dindex.sock` or `/tmp/dindex.sock`
- **Future**: Optional TCP socket for remote access

### Message Format

Use a length-prefixed binary protocol with `bincode` serialization:

```
┌──────────────┬──────────────────────────┐
│ Length (u32) │ Payload (bincode bytes)  │
└──────────────┴──────────────────────────┘
```

### Request/Response Types

```rust
// requests.rs
#[derive(Serialize, Deserialize)]
pub enum Request {
    // Queries
    Search {
        query: String,
        top_k: usize,
        format: OutputFormat,
    },
    
    // Write operations
    IndexDocuments {
        stream_id: Uuid,  // For streaming chunks
    },
    IndexChunk {
        stream_id: Uuid,
        chunk: ChunkPayload,
    },
    IndexComplete {
        stream_id: Uuid,
    },
    
    // Import
    ImportStart {
        source: ImportSource,
        options: ImportOptions,
    },
    ImportCancel {
        job_id: Uuid,
    },
    
    // Scrape
    ScrapeStart {
        urls: Vec<String>,
        options: ScrapeOptions,
    },
    
    // Management
    Status,
    Stats,
    Shutdown,
    ForceCommit,
}

#[derive(Serialize, Deserialize)]
pub enum Response {
    // Query results
    SearchResults {
        results: Vec<SearchResult>,
        query_time_ms: u64,
    },
    
    // Streaming acknowledgments
    StreamReady { stream_id: Uuid },
    ChunkAck { stream_id: Uuid, count: usize },
    
    // Job status
    JobStarted { job_id: Uuid },
    JobProgress { job_id: Uuid, progress: Progress },
    JobComplete { job_id: Uuid, stats: JobStats },
    JobFailed { job_id: Uuid, error: String },
    
    // Status
    Status(DaemonStatus),
    Stats(IndexStats),
    
    // Errors
    Error { code: ErrorCode, message: String },
    Ok,
}
```

## Implementation Phases

---

### Phase 1: Core Daemon Infrastructure

**Goal**: Establish daemon lifecycle and basic IPC without changing existing functionality.

#### 1.1 Daemon Module Structure

Create new module structure:

```
src/
├── daemon/
│   ├── mod.rs           # Daemon entry point
│   ├── server.rs        # Unix socket server
│   ├── protocol.rs      # Request/Response types
│   ├── handler.rs       # Request dispatch
│   └── lifecycle.rs     # Start, stop, health
├── client/
│   ├── mod.rs           # Client connection
│   └── commands.rs      # Command-specific clients
```

#### 1.2 Daemon Lifecycle

```rust
// daemon/lifecycle.rs

pub struct Daemon {
    config: Config,
    index_manager: Arc<IndexManager>,
    socket_path: PathBuf,
    shutdown: broadcast::Sender<()>,
}

impl Daemon {
    pub async fn start(config: Config) -> Result<Self>;
    pub async fn run(&self) -> Result<()>;  // Main loop
    pub async fn shutdown(&self) -> Result<()>;
    
    // PID file for single-instance guarantee
    fn acquire_lock(&self) -> Result<()>;
    fn release_lock(&self) -> Result<()>;
}
```

#### 1.3 Socket Server

```rust
// daemon/server.rs

pub struct IpcServer {
    socket_path: PathBuf,
    handler: Arc<RequestHandler>,
}

impl IpcServer {
    pub async fn run(&self, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
        let listener = UnixListener::bind(&self.socket_path)?;
        
        loop {
            tokio::select! {
                accept = listener.accept() => {
                    let (stream, _) = accept?;
                    let handler = self.handler.clone();
                    tokio::spawn(async move {
                        handle_connection(stream, handler).await
                    });
                }
                _ = shutdown.recv() => break,
            }
        }
        Ok(())
    }
}
```

#### 1.4 Client Connection

```rust
// client/mod.rs

pub struct DaemonClient {
    stream: UnixStream,
}

impl DaemonClient {
    pub async fn connect() -> Result<Self> {
        let socket_path = get_socket_path();
        let stream = UnixStream::connect(&socket_path).await
            .map_err(|_| Error::DaemonNotRunning)?;
        Ok(Self { stream })
    }
    
    pub async fn send(&mut self, req: Request) -> Result<Response>;
    
    // Streaming support for large operations
    pub async fn send_stream<T: Serialize>(
        &mut self, 
        items: impl Stream<Item = T>
    ) -> Result<Response>;
}
```

#### 1.5 CLI Changes for Phase 1

Update `main.rs` to add daemon commands:

```rust
#[derive(Subcommand)]
enum Commands {
    // New daemon commands
    #[command(name = "daemon")]
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
    
    // Existing commands unchanged for now
    Search { ... },
    Index { ... },
    // ...
}

#[derive(Subcommand)]
enum DaemonAction {
    Start {
        #[arg(short, long)]
        foreground: bool,
    },
    Stop,
    Status,
    Restart,
}
```

#### 1.6 Deliverables

- [ ] Daemon starts and listens on Unix socket
- [ ] PID file prevents multiple instances
- [ ] `dindex daemon start/stop/status` commands work
- [ ] Basic health check via socket
- [ ] Graceful shutdown with signal handling (SIGTERM, SIGINT)

---

### Phase 2: Migrate Search to Client Mode

**Goal**: `dindex search` uses daemon when available, falls back to direct access.

#### 2.1 Index Manager Service

Extract index management from CLI into a shared service:

```rust
// daemon/index_manager.rs

pub struct IndexManager {
    config: Config,
    vector_index: Arc<VectorIndex>,
    bm25_index: Arc<Bm25Index>,
    chunk_storage: Arc<ChunkStorage>,
    document_registry: Arc<DocumentRegistry>,
    embedder: Arc<Embedder>,
}

impl IndexManager {
    pub async fn load(config: &Config) -> Result<Self>;
    
    // Query interface
    pub async fn search(
        &self, 
        query: &str, 
        top_k: usize
    ) -> Result<Vec<SearchResult>>;
    
    // Will be used in Phase 3
    pub async fn index_chunks(&self, chunks: Vec<Chunk>) -> Result<()>;
    pub async fn commit(&self) -> Result<()>;
}
```

#### 2.2 Search Handler

```rust
// daemon/handler.rs

impl RequestHandler {
    async fn handle_search(
        &self,
        query: String,
        top_k: usize,
    ) -> Response {
        let start = Instant::now();
        
        match self.index_manager.search(&query, top_k).await {
            Ok(results) => Response::SearchResults {
                results,
                query_time_ms: start.elapsed().as_millis() as u64,
            },
            Err(e) => Response::Error {
                code: ErrorCode::SearchFailed,
                message: e.to_string(),
            },
        }
    }
}
```

#### 2.3 Search Client

```rust
// client/commands.rs

pub async fn search(
    query: &str,
    top_k: usize,
    format: OutputFormat,
) -> Result<Vec<SearchResult>> {
    let mut client = DaemonClient::connect().await?;
    
    let response = client.send(Request::Search {
        query: query.to_string(),
        top_k,
        format,
    }).await?;
    
    match response {
        Response::SearchResults { results, .. } => Ok(results),
        Response::Error { message, .. } => Err(Error::SearchFailed(message)),
        _ => Err(Error::UnexpectedResponse),
    }
}
```

#### 2.4 CLI Search Update

```rust
// main.rs - search command

Commands::Search { query, top_k, format, .. } => {
    // Try daemon first
    match client::commands::search(&query, top_k, format).await {
        Ok(results) => {
            output_results(results, format);
        }
        Err(Error::DaemonNotRunning) => {
            // Fallback to direct access (existing code)
            eprintln!("Note: Daemon not running, using direct access");
            search_direct(&query, top_k, format).await?;
        }
        Err(e) => return Err(e),
    }
}
```

#### 2.5 Deliverables

- [ ] `IndexManager` extracted and working in daemon
- [ ] Search requests handled via IPC
- [ ] `dindex search` uses daemon when available
- [ ] Fallback to direct access when daemon not running
- [ ] Search performance comparable to direct access (<10% overhead)

---

### Phase 3: Migrate Write Operations

**Goal**: `index`, `import`, `scrape` stream data to daemon for indexing.

#### 3.1 Write Pipeline

```rust
// daemon/write_pipeline.rs

pub struct WritePipeline {
    ingest_queue: mpsc::Sender<IngestItem>,
    batch_size: usize,
    commit_interval: Duration,
}

enum IngestItem {
    Chunk {
        stream_id: Uuid,
        chunk: Chunk,
    },
    Commit {
        stream_id: Uuid,
        respond_to: oneshot::Sender<Result<()>>,
    },
}

impl WritePipeline {
    pub fn new(
        index_manager: Arc<IndexManager>,
        batch_size: usize,
        commit_interval: Duration,
    ) -> Self;
    
    // Background worker
    pub async fn run(&self, mut shutdown: broadcast::Receiver<()>) {
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut interval = tokio::time::interval(self.commit_interval);
        
        loop {
            tokio::select! {
                Some(item) = self.ingest_queue.recv() => {
                    match item {
                        IngestItem::Chunk { chunk, .. } => {
                            batch.push(chunk);
                            if batch.len() >= self.batch_size {
                                self.flush_batch(&mut batch).await;
                            }
                        }
                        IngestItem::Commit { respond_to, .. } => {
                            self.flush_batch(&mut batch).await;
                            let result = self.index_manager.commit().await;
                            let _ = respond_to.send(result);
                        }
                    }
                }
                _ = interval.tick() => {
                    if !batch.is_empty() {
                        self.flush_batch(&mut batch).await;
                    }
                }
                _ = shutdown.recv() => {
                    // Final flush before shutdown
                    self.flush_batch(&mut batch).await;
                    self.index_manager.commit().await.ok();
                    break;
                }
            }
        }
    }
}
```

#### 3.2 Streaming Protocol for Index Command

```rust
// Protocol for streaming chunks from CLI to daemon

// Client side (index command)
pub async fn index_documents(paths: Vec<PathBuf>) -> Result<IndexStats> {
    let mut client = DaemonClient::connect().await?;
    
    // Start stream
    let stream_id = match client.send(Request::IndexDocuments { 
        stream_id: Uuid::new_v4() 
    }).await? {
        Response::StreamReady { stream_id } => stream_id,
        _ => return Err(Error::UnexpectedResponse),
    };
    
    // Stream chunks
    for path in paths {
        let chunks = chunk_document(&path)?;
        for chunk in chunks {
            client.send(Request::IndexChunk {
                stream_id,
                chunk: chunk.into(),
            }).await?;
        }
    }
    
    // Complete and wait for commit
    match client.send(Request::IndexComplete { stream_id }).await? {
        Response::JobComplete { stats, .. } => Ok(stats),
        Response::Error { message, .. } => Err(Error::IndexFailed(message)),
        _ => Err(Error::UnexpectedResponse),
    }
}
```

#### 3.3 Import Handler

```rust
// daemon/handler.rs

impl RequestHandler {
    async fn handle_import_start(
        &self,
        source: ImportSource,
        options: ImportOptions,
    ) -> Response {
        let job_id = Uuid::new_v4();
        
        // Spawn import task
        let pipeline = self.write_pipeline.clone();
        let progress_tx = self.progress_tx.clone();
        
        tokio::spawn(async move {
            let result = run_import(source, options, pipeline, progress_tx).await;
            // Store result for later retrieval
        });
        
        Response::JobStarted { job_id }
    }
}
```

#### 3.4 Progress Reporting

```rust
// daemon/progress.rs

pub struct ProgressTracker {
    jobs: DashMap<Uuid, JobProgress>,
    subscribers: DashMap<Uuid, Vec<mpsc::Sender<Progress>>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Progress {
    pub job_id: Uuid,
    pub stage: String,
    pub current: u64,
    pub total: Option<u64>,
    pub rate: Option<f64>,  // items/sec
    pub eta: Option<Duration>,
}

// Client can poll or subscribe to progress
impl RequestHandler {
    async fn handle_job_progress(&self, job_id: Uuid) -> Response {
        match self.progress_tracker.get(&job_id) {
            Some(progress) => Response::JobProgress { 
                job_id, 
                progress: progress.clone() 
            },
            None => Response::Error {
                code: ErrorCode::JobNotFound,
                message: format!("Job {} not found", job_id),
            },
        }
    }
}
```

#### 3.5 Deliverables

- [ ] Write pipeline with batching and periodic commits
- [ ] `dindex index` streams chunks to daemon
- [ ] `dindex import` runs as background job with progress
- [ ] `dindex scrape` runs as background job with progress
- [ ] Progress reporting via `dindex status`
- [ ] Job cancellation support
- [ ] Concurrent read/write verified working

---

### Phase 4: Unify with P2P

**Goal**: Merge daemon with existing P2P node, single process for everything.

#### 4.1 Unified Start Command

```rust
// Current: dindex start (P2P node)
// New: dindex start (daemon + P2P)

Commands::Start { listen, bootstrap, .. } => {
    let daemon = Daemon::start(config.clone()).await?;
    
    // Start P2P alongside daemon
    if !no_network {
        let network = Network::new(config.network.clone()).await?;
        network.start(listen, bootstrap).await?;
        
        // Connect P2P queries to index manager
        let index_manager = daemon.index_manager();
        network.set_query_handler(move |query| {
            index_manager.search(&query.text, query.top_k)
        });
    }
    
    daemon.run().await?;
}
```

#### 4.2 Remove Standalone Mode

After Phase 4, direct index access is removed:

```rust
Commands::Search { .. } => {
    match client::commands::search(&query, top_k, format).await {
        Ok(results) => output_results(results, format),
        Err(Error::DaemonNotRunning) => {
            eprintln!("Error: Daemon not running. Start with: dindex start");
            eprintln!("Or run in foreground with: dindex start --foreground");
            std::process::exit(1);
        }
        Err(e) => return Err(e),
    }
}
```

#### 4.3 Auto-Start Option

Optional convenience feature:

```rust
// config.toml
[daemon]
auto_start = true  # Start daemon automatically if not running

// client/mod.rs
pub async fn connect_or_start() -> Result<DaemonClient> {
    match DaemonClient::connect().await {
        Ok(client) => Ok(client),
        Err(Error::DaemonNotRunning) if config.daemon.auto_start => {
            // Start daemon in background
            start_daemon_background().await?;
            // Wait for socket
            tokio::time::sleep(Duration::from_millis(500)).await;
            DaemonClient::connect().await
        }
        Err(e) => Err(e),
    }
}
```

#### 4.4 Deliverables

- [ ] `dindex start` runs daemon + P2P together
- [ ] P2P queries use shared IndexManager
- [ ] Direct index access removed from CLI commands
- [ ] Auto-start option implemented (optional)
- [ ] Documentation updated

---

### Phase 5: Polish and Optimization

**Goal**: Production-ready daemon with monitoring, resilience, and performance.

#### 5.1 Crash Recovery

```rust
// daemon/recovery.rs

impl Daemon {
    async fn recover_from_crash(&self) -> Result<()> {
        // Check for incomplete transactions
        if let Some(wal) = self.load_wal()? {
            match wal.state {
                WalState::Writing => {
                    // Rollback incomplete write
                    self.rollback_wal(&wal)?;
                }
                WalState::Committing => {
                    // Complete the commit
                    self.complete_commit(&wal)?;
                }
            }
        }
        
        // Verify index integrity
        self.index_manager.verify_integrity().await?;
        
        Ok(())
    }
}
```

#### 5.2 Metrics and Monitoring

```rust
// daemon/metrics.rs

pub struct DaemonMetrics {
    // Query metrics
    pub queries_total: Counter,
    pub query_latency: Histogram,
    
    // Write metrics
    pub chunks_indexed: Counter,
    pub commits_total: Counter,
    pub commit_latency: Histogram,
    
    // Resource metrics
    pub memory_usage: Gauge,
    pub index_size: Gauge,
    pub active_connections: Gauge,
}

// Expose via status command or optional HTTP endpoint
impl RequestHandler {
    async fn handle_status(&self) -> Response {
        Response::Status(DaemonStatus {
            uptime: self.start_time.elapsed(),
            memory_mb: get_memory_usage(),
            index_stats: self.index_manager.stats().await,
            active_jobs: self.job_tracker.active_count(),
            metrics: self.metrics.snapshot(),
        })
    }
}
```

#### 5.3 Connection Pooling (Client Side)

```rust
// client/pool.rs

pub struct ConnectionPool {
    connections: Vec<DaemonClient>,
    available: mpsc::Receiver<DaemonClient>,
    return_tx: mpsc::Sender<DaemonClient>,
}

impl ConnectionPool {
    pub async fn get(&self) -> PooledConnection {
        // Reuse existing connection or create new
    }
}
```

#### 5.4 Deliverables

- [ ] WAL for crash recovery
- [ ] Index integrity verification on startup
- [ ] Metrics collection and reporting
- [ ] Connection pooling for high-throughput clients
- [ ] Systemd service file for production deployment
- [ ] Docker compose updated for daemon mode

---

## Migration Path

### For Existing Users

1. **Phase 1-2**: No breaking changes, daemon is optional
2. **Phase 3**: Daemon recommended for concurrent operations
3. **Phase 4**: Daemon required, clear error messages guide users
4. **Phase 5**: Daemon is transparent, auto-start available

### Backwards Compatibility

During transition (Phases 1-3):
- All commands work without daemon (existing behavior)
- Daemon provides enhanced functionality when running
- Warning messages encourage daemon usage

After Phase 4:
- Daemon is required
- `--standalone` flag available for emergency direct access
- Clear migration guide in release notes

---

## Testing Strategy

### Unit Tests

- Protocol serialization/deserialization
- Request handler logic
- Write pipeline batching
- Progress tracking

### Integration Tests

```rust
#[tokio::test]
async fn test_concurrent_search_during_import() {
    let daemon = TestDaemon::start().await;
    
    // Start import in background
    let import_handle = tokio::spawn(async {
        daemon.client().import("test_data/wiki.xml").await
    });
    
    // Run searches concurrently
    for _ in 0..100 {
        let results = daemon.client().search("test query", 10).await?;
        assert!(results.is_ok());
    }
    
    import_handle.await??;
}

#[tokio::test]
async fn test_daemon_crash_recovery() {
    let daemon = TestDaemon::start().await;
    
    // Index some data
    daemon.client().index_chunks(test_chunks()).await?;
    
    // Simulate crash (kill without graceful shutdown)
    daemon.kill();
    
    // Restart and verify data integrity
    let daemon = TestDaemon::start().await;
    let stats = daemon.client().stats().await?;
    assert_eq!(stats.chunk_count, expected_count);
}
```

### Performance Tests

- IPC overhead measurement
- Throughput under concurrent load
- Memory usage over time
- Comparison with direct access baseline

---

## File Changes Summary

### New Files

```
src/daemon/
├── mod.rs
├── server.rs
├── protocol.rs
├── handler.rs
├── lifecycle.rs
├── index_manager.rs
├── write_pipeline.rs
├── progress.rs
├── metrics.rs
└── recovery.rs

src/client/
├── mod.rs
├── connection.rs
├── pool.rs
└── commands.rs
```

### Modified Files

```
src/main.rs              # Add daemon commands, update CLI commands
src/config.rs            # Add daemon config section
Cargo.toml               # Add bincode, dashmap dependencies
docker-compose.yml       # Update for daemon mode
```

### Removed (Phase 4+)

```
# Direct index access code paths in main.rs (moved to index_manager.rs)
```

---

## Dependencies

### New Dependencies

```toml
[dependencies]
bincode = "1.3"          # Binary serialization
dashmap = "5.5"          # Concurrent hashmap for job tracking
uuid = { version = "1.0", features = ["v4", "serde"] }
```

### Optional Dependencies

```toml
[features]
metrics = ["prometheus"]  # Optional metrics export

[dependencies.prometheus]
version = "0.13"
optional = true
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| IPC overhead affects search latency | Medium | Benchmark early, optimize protocol |
| Daemon crashes lose in-flight data | High | WAL for write durability |
| Complex debugging across processes | Medium | Structured logging, request IDs |
| Users forget to start daemon | Low | Clear error messages, auto-start option |
| Socket permission issues | Low | Document, provide fallback paths |

---

## Success Criteria

1. **Functional**: Search works during import/index/scrape operations
2. **Performance**: <10ms IPC overhead on search operations
3. **Reliability**: No data loss on daemon crash
4. **Usability**: Clear error messages, simple daemon management
5. **Resource**: Single embedding model instance, reduced memory usage
