import XCTest
@testable import DIndexClient

final class DIndexClientTests: XCTestCase {

    // MARK: - Model Tests

    func testSearchRequestEncoding() throws {
        let request = SearchRequest(query: "test query", topK: 5)
        let encoder = JSONEncoder()
        let data = try encoder.encode(request)
        let json = String(data: data, encoding: .utf8)!

        XCTAssertTrue(json.contains("\"query\":\"test query\""))
        XCTAssertTrue(json.contains("\"top_k\":5"))
    }

    func testSearchResponseDecoding() throws {
        let json = """
        {
            "results": [
                {
                    "chunk": {
                        "content": "Test content",
                        "metadata": {
                            "chunk_id": "chunk1",
                            "document_id": "doc1",
                            "source_url": "https://example.com",
                            "source_title": "Example"
                        }
                    },
                    "relevance_score": 0.95,
                    "matched_by": ["dense", "bm25"]
                }
            ],
            "query_time_ms": 42
        }
        """

        let decoder = JSONDecoder()
        let response = try decoder.decode(SearchResponse.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(response.results.count, 1)
        XCTAssertEqual(response.queryTimeMs, 42)

        let result = response.results[0]
        XCTAssertEqual(result.chunk.content, "Test content")
        XCTAssertEqual(result.relevanceScore, 0.95, accuracy: 0.001)
        XCTAssertEqual(result.matchedBy, ["dense", "bm25"])

        let metadata = result.chunk.metadata
        XCTAssertEqual(metadata.chunkId, "chunk1")
        XCTAssertEqual(metadata.documentId, "doc1")
        XCTAssertEqual(metadata.sourceUrl, "https://example.com")
        XCTAssertEqual(metadata.sourceTitle, "Example")
    }

    func testIndexStatsDecoding() throws {
        let json = """
        {
            "total_documents": 100,
            "total_chunks": 500,
            "vector_index_size_bytes": 1048576,
            "bm25_index_size_bytes": 524288,
            "storage_size_bytes": 2097152
        }
        """

        let decoder = JSONDecoder()
        let stats = try decoder.decode(IndexStats.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(stats.totalDocuments, 100)
        XCTAssertEqual(stats.totalChunks, 500)
        XCTAssertEqual(stats.vectorIndexSizeBytes, 1048576)
        XCTAssertEqual(stats.bm25IndexSizeBytes, 524288)
        XCTAssertEqual(stats.storageSizeBytes, 2097152)
    }

    func testDaemonStatusDecoding() throws {
        let json = """
        {
            "running": true,
            "uptime_seconds": 3600,
            "memory_mb": 256,
            "active_jobs": 2,
            "pending_writes": 10
        }
        """

        let decoder = JSONDecoder()
        let status = try decoder.decode(DaemonStatus.self, from: json.data(using: .utf8)!)

        XCTAssertTrue(status.running)
        XCTAssertEqual(status.uptimeSeconds, 3600)
        XCTAssertEqual(status.memoryMb, 256)
        XCTAssertEqual(status.activeJobs, 2)
        XCTAssertEqual(status.pendingWrites, 10)
    }

    func testDocumentEncoding() throws {
        let document = Document(
            content: "This is test content",
            title: "Test Document",
            url: "https://example.com/test"
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(document)

        // Decode back to verify round-trip
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(Document.self, from: data)

        XCTAssertEqual(decoded.content, "This is test content")
        XCTAssertEqual(decoded.title, "Test Document")
        XCTAssertEqual(decoded.url, "https://example.com/test")
    }

    func testIndexRequestEncoding() throws {
        let documents = [
            Document(content: "Doc 1"),
            Document(content: "Doc 2", title: "Second")
        ]
        let request = IndexRequest(documents: documents)

        let encoder = JSONEncoder()
        let data = try encoder.encode(request)
        let json = String(data: data, encoding: .utf8)!

        XCTAssertTrue(json.contains("\"documents\":"))
        XCTAssertTrue(json.contains("\"Doc 1\""))
        XCTAssertTrue(json.contains("\"Doc 2\""))
    }

    func testErrorResponseDecoding() throws {
        let json = """
        {
            "code": "SEARCH_FAILED",
            "message": "Index not found"
        }
        """

        let decoder = JSONDecoder()
        let error = try decoder.decode(ErrorResponse.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(error.code, "SEARCH_FAILED")
        XCTAssertEqual(error.message, "Index not found")
    }

    // MARK: - Error Tests

    func testDIndexErrorDescriptions() {
        let networkError = DIndexError.networkError(underlying: NSError(domain: "test", code: -1))
        XCTAssertTrue(networkError.localizedDescription.contains("Network error"))

        let serverError = DIndexError.serverError(code: "TEST", message: "Test error")
        XCTAssertTrue(serverError.localizedDescription.contains("TEST"))
        XCTAssertTrue(serverError.localizedDescription.contains("Test error"))

        let unauthorized = DIndexError.unauthorized
        XCTAssertTrue(unauthorized.localizedDescription.contains("Unauthorized"))
    }

    // MARK: - Configuration Tests

    func testHTTPConfiguration() {
        let config = DIndexConfiguration.http(
            baseURL: URL(string: "http://localhost:8080")!,
            apiKey: "test-key"
        )

        XCTAssertEqual(config.baseURL?.absoluteString, "http://localhost:8080")
        XCTAssertEqual(config.apiKey, "test-key")
        XCTAssertNil(config.socketPath)
    }

    #if os(macOS)
    func testSocketConfiguration() {
        let config = DIndexConfiguration.socket(path: "/custom/socket.sock")

        XCTAssertNil(config.baseURL)
        XCTAssertEqual(config.socketPath, "/custom/socket.sock")
    }
    #endif
}
