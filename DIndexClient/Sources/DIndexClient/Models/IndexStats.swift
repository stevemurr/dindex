import Foundation

/// Index statistics
public struct IndexStats: Codable, Sendable {
    /// Total number of documents
    public let totalDocuments: Int

    /// Total number of chunks
    public let totalChunks: Int

    /// Vector index size in bytes
    public let vectorIndexSizeBytes: UInt64

    /// BM25 index size in bytes
    public let bm25IndexSizeBytes: UInt64

    /// Total storage size in bytes
    public let storageSizeBytes: UInt64

    enum CodingKeys: String, CodingKey {
        case totalDocuments = "total_documents"
        case totalChunks = "total_chunks"
        case vectorIndexSizeBytes = "vector_index_size_bytes"
        case bm25IndexSizeBytes = "bm25_index_size_bytes"
        case storageSizeBytes = "storage_size_bytes"
    }

    public init(
        totalDocuments: Int,
        totalChunks: Int,
        vectorIndexSizeBytes: UInt64,
        bm25IndexSizeBytes: UInt64,
        storageSizeBytes: UInt64
    ) {
        self.totalDocuments = totalDocuments
        self.totalChunks = totalChunks
        self.vectorIndexSizeBytes = vectorIndexSizeBytes
        self.bm25IndexSizeBytes = bm25IndexSizeBytes
        self.storageSizeBytes = storageSizeBytes
    }
}

/// Daemon status
public struct DaemonStatus: Codable, Sendable {
    /// Whether the daemon is running
    public let running: Bool

    /// Uptime in seconds
    public let uptimeSeconds: UInt64

    /// Memory usage in MB
    public let memoryMb: UInt64

    /// Number of active background jobs
    public let activeJobs: Int

    /// Number of pending writes
    public let pendingWrites: Int

    enum CodingKeys: String, CodingKey {
        case running
        case uptimeSeconds = "uptime_seconds"
        case memoryMb = "memory_mb"
        case activeJobs = "active_jobs"
        case pendingWrites = "pending_writes"
    }

    public init(
        running: Bool,
        uptimeSeconds: UInt64,
        memoryMb: UInt64,
        activeJobs: Int,
        pendingWrites: Int
    ) {
        self.running = running
        self.uptimeSeconds = uptimeSeconds
        self.memoryMb = memoryMb
        self.activeJobs = activeJobs
        self.pendingWrites = pendingWrites
    }
}

/// Health check response
public struct HealthResponse: Codable, Sendable {
    /// Whether the service is healthy
    public let healthy: Bool

    /// Service version
    public let version: String

    public init(healthy: Bool, version: String) {
        self.healthy = healthy
        self.version = version
    }
}
