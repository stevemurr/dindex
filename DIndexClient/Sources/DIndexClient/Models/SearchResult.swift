import Foundation

/// A single search result
public struct SearchResult: Codable, Sendable, Equatable {
    /// The matched chunk
    public let chunk: Chunk

    /// Relevance score (0.0 to 1.0)
    public let relevanceScore: Float

    /// Which retrieval methods matched
    public let matchedBy: [String]

    enum CodingKeys: String, CodingKey {
        case chunk
        case relevanceScore = "relevance_score"
        case matchedBy = "matched_by"
    }

    public init(chunk: Chunk, relevanceScore: Float, matchedBy: [String] = []) {
        self.chunk = chunk
        self.relevanceScore = relevanceScore
        self.matchedBy = matchedBy
    }
}

/// Response from a search query
public struct SearchResponse: Codable, Sendable {
    /// Search results
    public let results: [SearchResult]

    /// Query execution time in milliseconds
    public let queryTimeMs: UInt64

    enum CodingKeys: String, CodingKey {
        case results
        case queryTimeMs = "query_time_ms"
    }

    public init(results: [SearchResult], queryTimeMs: UInt64) {
        self.results = results
        self.queryTimeMs = queryTimeMs
    }
}
