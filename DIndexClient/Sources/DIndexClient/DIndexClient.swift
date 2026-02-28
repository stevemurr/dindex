import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Client for interacting with a dindex server via HTTP
///
/// ## Example Usage
///
/// ```swift
/// let client = DIndexClient(
///     baseURL: URL(string: "http://localhost:8080")!,
///     apiKey: "your-api-key"
/// )
///
/// // Search
/// let response = try await client.search(query: "machine learning", topK: 5)
/// for group in response.results {
///     print("[\(group.relevanceScore)] \(group.sourceTitle ?? "untitled")")
///     for chunk in group.chunks {
///         print("  [\(chunk.relevanceScore)] \(chunk.content.prefix(100))")
///     }
/// }
///
/// // Check health
/// let isHealthy = try await client.health()
/// ```
public final class DIndexClient: Sendable {
    private let transport: HTTPTransport

    // MARK: - Initialization

    /// Create a client with HTTP transport
    /// - Parameters:
    ///   - baseURL: The base URL of the dindex server (e.g., http://localhost:8080)
    ///   - apiKey: Optional API key for authentication
    public init(baseURL: URL, apiKey: String? = nil) {
        self.transport = HTTPTransport(baseURL: baseURL, apiKey: apiKey)
    }

    /// Create a client with a custom URL session
    /// - Parameters:
    ///   - baseURL: The base URL of the dindex server
    ///   - apiKey: Optional API key for authentication
    ///   - session: URLSession to use for requests
    public init(baseURL: URL, apiKey: String? = nil, session: URLSession) {
        self.transport = HTTPTransport(baseURL: baseURL, apiKey: apiKey, session: session)
    }

    // MARK: - Search

    /// Search the index
    /// - Parameters:
    ///   - query: The search query text
    ///   - topK: Number of results to return (default: 10)
    ///   - filters: Optional search filters
    /// - Returns: Search response with results
    public func search(
        query: String,
        topK: Int = 10,
        filters: SearchFilters? = nil
    ) async throws -> SearchResponse {
        let request = SearchRequest(query: query, topK: topK, filters: filters)
        return try await transport.post(path: "/api/v1/search", body: request)
    }

    // MARK: - Status

    /// Check if the server is healthy
    /// - Returns: true if healthy
    public func health() async throws -> Bool {
        let response: HealthResponse = try await transport.get(path: "/api/v1/health")
        return response.healthy
    }

    /// Get daemon status
    /// - Returns: Current daemon status
    public func status() async throws -> DaemonStatus {
        try await transport.get(path: "/api/v1/status")
    }

    /// Get index statistics
    /// - Returns: Index statistics
    public func stats() async throws -> IndexStats {
        try await transport.get(path: "/api/v1/stats")
    }

    // MARK: - Indexing

    /// Index documents
    /// - Parameter documents: Documents to index
    /// - Returns: Indexing result
    public func index(documents: [Document]) async throws -> IndexResponse {
        let request = IndexRequest(documents: documents)
        return try await transport.post(path: "/api/v1/index", body: request)
    }

    /// Force commit pending writes
    /// - Returns: Commit result
    @discardableResult
    public func commit() async throws -> CommitResponse {
        try await transport.post(path: "/api/v1/index/commit")
    }

    // MARK: - Deletion

    /// Delete documents by their IDs
    /// - Parameter documentIds: Array of document IDs to delete
    /// - Returns: Delete result with counts
    @discardableResult
    public func deleteDocuments(ids documentIds: [String]) async throws -> DeleteResponse {
        let request = DeleteRequest(documentIds: documentIds)
        return try await transport.delete(path: "/api/v1/documents", body: request)
    }

    /// Delete a single document by ID
    /// - Parameter documentId: The document ID to delete
    /// - Returns: Delete result with counts
    @discardableResult
    public func deleteDocument(id documentId: String) async throws -> DeleteResponse {
        try await deleteDocuments(ids: [documentId])
    }

    /// Clear all entries from the index
    /// - Returns: Clear result with count of deleted chunks
    @discardableResult
    public func clearAll() async throws -> ClearResponse {
        try await transport.post(path: "/api/v1/index/clear")
    }
}

// MARK: - Convenience Extensions

extension DIndexClient {
    /// Search and return just the results
    /// - Parameters:
    ///   - query: The search query text
    ///   - topK: Number of results to return
    /// - Returns: Array of grouped search results
    public func searchResults(query: String, topK: Int = 10) async throws -> [GroupedSearchResult] {
        let response = try await search(query: query, topK: topK)
        return response.results
    }

    /// Index a single document
    /// - Parameter document: Document to index
    /// - Returns: Indexing result
    public func index(document: Document) async throws -> IndexResponse {
        try await index(documents: [document])
    }

    /// Index content with optional metadata
    /// - Parameters:
    ///   - content: Document content
    ///   - title: Optional title
    ///   - url: Optional source URL
    /// - Returns: Indexing result
    public func index(content: String, title: String? = nil, url: String? = nil) async throws -> IndexResponse {
        let document = Document(content: content, title: title, url: url)
        return try await index(document: document)
    }
}

// MARK: - Category-based Convenience Extensions

extension DIndexClient {
    /// Index content with category tags
    ///
    /// Categories are stored as a comma-separated string in the "categories" metadata key.
    /// This allows filtering during search using metadata_contains filter.
    ///
    /// - Parameters:
    ///   - content: Document content
    ///   - title: Optional title
    ///   - url: Optional source URL
    ///   - categories: Array of category tags (e.g., ["history", "web"])
    /// - Returns: Indexing result
    public func index(
        content: String,
        title: String? = nil,
        url: String? = nil,
        categories: [String]
    ) async throws -> IndexResponse {
        let metadata = ["categories": categories.joined(separator: ",")]
        let document = Document(content: content, title: title, url: url, metadata: metadata)
        return try await index(document: document)
    }

    /// Search with category filter
    ///
    /// Filters results to only include documents that have at least one of the specified categories.
    ///
    /// - Parameters:
    ///   - query: The search query text
    ///   - categories: Array of category tags to filter by (results must have at least one)
    ///   - topK: Number of results to return (default: 10)
    /// - Returns: Search response with filtered results
    public func search(
        query: String,
        categories: [String],
        topK: Int = 10
    ) async throws -> SearchResponse {
        let filters = SearchFilters(metadataContains: ["categories": categories])
        return try await search(query: query, topK: topK, filters: filters)
    }

    /// Search with category filter and return just the results
    ///
    /// - Parameters:
    ///   - query: The search query text
    ///   - categories: Array of category tags to filter by
    ///   - topK: Number of results to return (default: 10)
    /// - Returns: Array of grouped search results
    public func searchResults(
        query: String,
        categories: [String],
        topK: Int = 10
    ) async throws -> [GroupedSearchResult] {
        let response = try await search(query: query, categories: categories, topK: topK)
        return response.results
    }
}

// MARK: - Scraping Extensions

extension DIndexClient {
    /// Start a web scraping job
    ///
    /// - Parameters:
    ///   - urls: URLs to start scraping from
    ///   - options: Scrape options (depth, domain filtering, etc.)
    /// - Returns: Response containing the job ID
    public func startScrape(urls: [String], options: ScrapeOptions = ScrapeOptions()) async throws -> JobStartedResponse {
        let request = ScrapeRequest(urls: urls, options: options)
        return try await transport.post(path: "/api/v1/scrape", body: request)
    }

    /// Start scraping a single URL
    ///
    /// - Parameters:
    ///   - url: URL to start scraping from
    ///   - options: Scrape options
    /// - Returns: Response containing the job ID
    public func startScrape(url: String, options: ScrapeOptions = ScrapeOptions()) async throws -> JobStartedResponse {
        try await startScrape(urls: [url], options: options)
    }

    /// Get the progress of a scraping job
    ///
    /// - Parameter jobId: The job ID returned from startScrape
    /// - Returns: Current job progress
    public func getJobProgress(jobId: String) async throws -> JobProgress {
        try await transport.get(path: "/api/v1/jobs/\(jobId)")
    }

    /// Cancel a running scrape job
    ///
    /// - Parameter jobId: The job ID to cancel
    /// - Returns: Cancellation response
    @discardableResult
    public func cancelJob(jobId: String) async throws -> JobCancelResponse {
        try await transport.post(path: "/api/v1/jobs/\(jobId)/cancel")
    }

    /// Subscribe to real-time events for a scrape job via SSE
    ///
    /// - Parameter jobId: The job ID to subscribe to
    /// - Returns: An async stream of scrape events
    public func subscribeToJobEvents(jobId: String) -> AsyncThrowingStream<ScrapeEvent, Error> {
        transport.streamSSE(path: "/api/v1/jobs/\(jobId)/events")
    }
}
