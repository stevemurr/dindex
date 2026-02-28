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

    /// Best-matching sentence snippet for citations
    public let snippet: String?

    enum CodingKeys: String, CodingKey {
        case chunkId = "chunk_id"
        case content
        case relevanceScore = "relevance_score"
        case matchedBy = "matched_by"
        case sectionHierarchy = "section_hierarchy"
        case positionInDoc = "position_in_doc"
        case snippet
    }

    public init(
        chunkId: String,
        content: String,
        relevanceScore: Float,
        matchedBy: [String] = [],
        sectionHierarchy: [String] = [],
        positionInDoc: Float = 0.0,
        snippet: String? = nil
    ) {
        self.chunkId = chunkId
        self.content = content
        self.relevanceScore = relevanceScore
        self.matchedBy = matchedBy
        self.sectionHierarchy = sectionHierarchy
        self.positionInDoc = positionInDoc
        self.snippet = snippet
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

    /// 1-based citation index linking to the citations array
    public let citationIndex: Int

    /// Matching chunks sorted by score descending
    public let chunks: [MatchingChunk]

    enum CodingKeys: String, CodingKey {
        case documentId = "document_id"
        case sourceUrl = "source_url"
        case sourceTitle = "source_title"
        case relevanceScore = "relevance_score"
        case citationIndex = "citation_index"
        case chunks
    }

    public init(
        documentId: String,
        sourceUrl: String? = nil,
        sourceTitle: String? = nil,
        relevanceScore: Float,
        citationIndex: Int = 0,
        chunks: [MatchingChunk]
    ) {
        self.documentId = documentId
        self.sourceUrl = sourceUrl
        self.sourceTitle = sourceTitle
        self.relevanceScore = relevanceScore
        self.citationIndex = citationIndex
        self.chunks = chunks
    }
}

/// A citation entry bundling source metadata and the best snippet
public struct Citation: Codable, Sendable, Equatable {
    /// 1-based citation index
    public let index: Int

    /// Source title if available
    public let sourceTitle: String?

    /// Source URL if available
    public let sourceUrl: String?

    /// Best snippet from the top-scoring chunk
    public let snippet: String?

    enum CodingKeys: String, CodingKey {
        case index
        case sourceTitle = "source_title"
        case sourceUrl = "source_url"
        case snippet
    }

    public init(
        index: Int,
        sourceTitle: String? = nil,
        sourceUrl: String? = nil,
        snippet: String? = nil
    ) {
        self.index = index
        self.sourceTitle = sourceTitle
        self.sourceUrl = sourceUrl
        self.snippet = snippet
    }
}

/// Response from a search query
public struct SearchResponse: Codable, Sendable {
    /// Search results grouped by document
    public let results: [GroupedSearchResult]

    /// Citation entries for each grouped result
    public let citations: [Citation]

    /// Total number of unique documents matched
    public let totalDocuments: Int

    /// Total number of matching chunks across all documents
    public let totalChunks: Int

    /// Query execution time in milliseconds
    public let queryTimeMs: UInt64

    enum CodingKeys: String, CodingKey {
        case results
        case citations
        case totalDocuments = "total_documents"
        case totalChunks = "total_chunks"
        case queryTimeMs = "query_time_ms"
    }

    public init(results: [GroupedSearchResult], citations: [Citation] = [], totalDocuments: Int, totalChunks: Int, queryTimeMs: UInt64) {
        self.results = results
        self.citations = citations
        self.totalDocuments = totalDocuments
        self.totalChunks = totalChunks
        self.queryTimeMs = queryTimeMs
    }
}
