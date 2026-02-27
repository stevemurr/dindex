import Foundation

/// A matching chunk within a grouped search result
public struct MatchingChunk: Codable, Sendable, Equatable {
    /// Chunk identifier
    public let chunkId: String

    /// The chunk content text
    public let content: String

    /// Relevance score (0.0 to 1.0)
    public let relevanceScore: Float

    /// Which retrieval methods matched
    public let matchedBy: [String]

    /// Section hierarchy in the document
    public let sectionHierarchy: [String]

    /// Position in document (0.0 to 1.0)
    public let positionInDoc: Float

    enum CodingKeys: String, CodingKey {
        case chunkId = "chunk_id"
        case content
        case relevanceScore = "relevance_score"
        case matchedBy = "matched_by"
        case sectionHierarchy = "section_hierarchy"
        case positionInDoc = "position_in_doc"
    }

    public init(
        chunkId: String,
        content: String,
        relevanceScore: Float,
        matchedBy: [String] = [],
        sectionHierarchy: [String] = [],
        positionInDoc: Float = 0.0
    ) {
        self.chunkId = chunkId
        self.content = content
        self.relevanceScore = relevanceScore
        self.matchedBy = matchedBy
        self.sectionHierarchy = sectionHierarchy
        self.positionInDoc = positionInDoc
    }
}

/// Search results grouped by document
public struct GroupedSearchResult: Codable, Sendable, Equatable {
    /// Parent document identifier
    public let documentId: String

    /// Source URL if available
    public let sourceUrl: String?

    /// Source title if available
    public let sourceTitle: String?

    /// Maximum relevance score among all matching chunks
    public let relevanceScore: Float

    /// Matching chunks sorted by score descending
    public let chunks: [MatchingChunk]

    enum CodingKeys: String, CodingKey {
        case documentId = "document_id"
        case sourceUrl = "source_url"
        case sourceTitle = "source_title"
        case relevanceScore = "relevance_score"
        case chunks
    }

    public init(
        documentId: String,
        sourceUrl: String? = nil,
        sourceTitle: String? = nil,
        relevanceScore: Float,
        chunks: [MatchingChunk]
    ) {
        self.documentId = documentId
        self.sourceUrl = sourceUrl
        self.sourceTitle = sourceTitle
        self.relevanceScore = relevanceScore
        self.chunks = chunks
    }
}

/// Response from a search query
public struct SearchResponse: Codable, Sendable {
    /// Search results grouped by document
    public let results: [GroupedSearchResult]

    /// Total number of unique documents matched
    public let totalDocuments: Int

    /// Total number of matching chunks across all documents
    public let totalChunks: Int

    /// Query execution time in milliseconds
    public let queryTimeMs: UInt64

    enum CodingKeys: String, CodingKey {
        case results
        case totalDocuments = "total_documents"
        case totalChunks = "total_chunks"
        case queryTimeMs = "query_time_ms"
    }

    public init(results: [GroupedSearchResult], totalDocuments: Int, totalChunks: Int, queryTimeMs: UInt64) {
        self.results = results
        self.totalDocuments = totalDocuments
        self.totalChunks = totalChunks
        self.queryTimeMs = queryTimeMs
    }
}
