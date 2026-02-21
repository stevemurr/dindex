import Foundation

/// Client for interacting with a dindex server
///
/// Supports both HTTP (all platforms) and Unix socket (macOS only) transports.
///
/// ## Example Usage
///
/// ```swift
/// // HTTP transport (all platforms)
/// let client = DIndexClient(
///     baseURL: URL(string: "http://localhost:8080")!,
///     apiKey: "your-api-key"
/// )
///
/// // Search
/// let response = try await client.search(query: "machine learning", topK: 5)
/// for result in response.results {
///     print("[\(result.relevanceScore)] \(result.chunk.content)")
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

    #if os(macOS)
    /// Create a client for local daemon via Unix socket
    /// - Parameter socketPath: Path to the Unix socket (default: /tmp/dindex.sock)
    ///
    /// Note: This initializer creates an HTTP client pointing to localhost.
    /// For true Unix socket support, use `UnixSocketTransport` directly.
    public init(socketPath: String = "/tmp/dindex.sock") {
        // For now, fall back to HTTP on localhost
        // Full Unix socket support would require implementing bincode protocol
        self.transport = HTTPTransport(
            baseURL: URL(string: "http://127.0.0.1:8080")!,
            apiKey: nil
        )
    }
    #endif

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
}

// MARK: - Convenience Extensions

extension DIndexClient {
    /// Search and return just the results
    /// - Parameters:
    ///   - query: The search query text
    ///   - topK: Number of results to return
    /// - Returns: Array of search results
    public func searchResults(query: String, topK: Int = 10) async throws -> [SearchResult] {
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
