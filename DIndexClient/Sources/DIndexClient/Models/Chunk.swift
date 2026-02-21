import Foundation

/// Metadata for a document chunk
public struct ChunkMetadata: Codable, Sendable, Equatable {
    /// Unique chunk identifier
    public let chunkId: String

    /// Parent document identifier
    public let documentId: String

    /// Source URL if available
    public let sourceUrl: String?

    /// Source title if available
    public let sourceTitle: String?

    enum CodingKeys: String, CodingKey {
        case chunkId = "chunk_id"
        case documentId = "document_id"
        case sourceUrl = "source_url"
        case sourceTitle = "source_title"
    }

    public init(chunkId: String, documentId: String, sourceUrl: String? = nil, sourceTitle: String? = nil) {
        self.chunkId = chunkId
        self.documentId = documentId
        self.sourceUrl = sourceUrl
        self.sourceTitle = sourceTitle
    }
}

/// A document chunk
public struct Chunk: Codable, Sendable, Equatable {
    /// The chunk content text
    public let content: String

    /// Chunk metadata
    public let metadata: ChunkMetadata

    public init(content: String, metadata: ChunkMetadata) {
        self.content = content
        self.metadata = metadata
    }
}
