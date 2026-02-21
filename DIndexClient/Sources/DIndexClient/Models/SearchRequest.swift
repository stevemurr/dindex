import Foundation

/// Request to search the index
public struct SearchRequest: Codable, Sendable {
    /// The search query text
    public let query: String

    /// Number of results to return
    public let topK: Int

    /// Optional filters
    public let filters: SearchFilters?

    enum CodingKeys: String, CodingKey {
        case query
        case topK = "top_k"
        case filters
    }

    public init(query: String, topK: Int = 10, filters: SearchFilters? = nil) {
        self.query = query
        self.topK = topK
        self.filters = filters
    }
}

/// Optional filters for search
public struct SearchFilters: Codable, Sendable {
    /// Filter by source URL prefix
    public let sourceUrlPrefix: String?

    enum CodingKeys: String, CodingKey {
        case sourceUrlPrefix = "source_url_prefix"
    }

    public init(sourceUrlPrefix: String? = nil) {
        self.sourceUrlPrefix = sourceUrlPrefix
    }
}
