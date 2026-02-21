import Foundation

/// A document to be indexed
public struct Document: Codable, Sendable {
    /// Document content
    public let content: String

    /// Optional document title
    public let title: String?

    /// Optional source URL
    public let url: String?

    /// Optional custom metadata (key-value pairs)
    public let metadata: [String: String]?

    public init(
        content: String,
        title: String? = nil,
        url: String? = nil,
        metadata: [String: String]? = nil
    ) {
        self.content = content
        self.title = title
        self.url = url
        self.metadata = metadata
    }
}

/// Request to index documents
public struct IndexRequest: Codable, Sendable {
    /// Documents to index
    public let documents: [Document]

    public init(documents: [Document]) {
        self.documents = documents
    }
}

/// Response from indexing documents
public struct IndexResponse: Codable, Sendable {
    /// Number of documents indexed
    public let documentsIndexed: Int

    /// Number of chunks created
    public let chunksCreated: Int

    /// Duration in milliseconds
    public let durationMs: UInt64

    enum CodingKeys: String, CodingKey {
        case documentsIndexed = "documents_indexed"
        case chunksCreated = "chunks_created"
        case durationMs = "duration_ms"
    }

    public init(documentsIndexed: Int, chunksCreated: Int, durationMs: UInt64) {
        self.documentsIndexed = documentsIndexed
        self.chunksCreated = chunksCreated
        self.durationMs = durationMs
    }
}

/// Response from commit operation
public struct CommitResponse: Codable, Sendable {
    /// Whether the commit was successful
    public let success: Bool

    public init(success: Bool) {
        self.success = success
    }
}
