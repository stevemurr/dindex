import XCTest
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
@testable import DIndexClient

// MARK: - Mock URL Protocol

private final class MockURLProtocol: URLProtocol {
    nonisolated(unsafe) static var requestHandler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.requestHandler else {
            client?.urlProtocol(self, didFailWithError: URLError(.unknown))
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

private func makeMockClient() -> DIndexClient {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [MockURLProtocol.self]
    let session = URLSession(configuration: config)
    return DIndexClient(
        baseURL: URL(string: "http://localhost:8080")!,
        session: session
    )
}

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

    // MARK: - Delete/Clear Model Tests

    func testDeleteRequestEncoding() throws {
        let request = DeleteRequest(documentIds: ["doc1", "doc2", "doc3"])
        let encoder = JSONEncoder()
        let data = try encoder.encode(request)
        let json = String(data: data, encoding: .utf8)!

        XCTAssertTrue(json.contains("\"document_ids\""))
        XCTAssertTrue(json.contains("\"doc1\""))
        XCTAssertTrue(json.contains("\"doc2\""))
        XCTAssertTrue(json.contains("\"doc3\""))
    }

    func testDeleteResponseDecoding() throws {
        let json = """
        {
            "documents_deleted": 3,
            "chunks_deleted": 15,
            "duration_ms": 42
        }
        """

        let decoder = JSONDecoder()
        let response = try decoder.decode(DeleteResponse.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(response.documentsDeleted, 3)
        XCTAssertEqual(response.chunksDeleted, 15)
        XCTAssertEqual(response.durationMs, 42)
    }

    func testClearResponseDecoding() throws {
        let json = """
        {
            "chunks_deleted": 500,
            "duration_ms": 1234
        }
        """

        let decoder = JSONDecoder()
        let response = try decoder.decode(ClearResponse.self, from: json.data(using: .utf8)!)

        XCTAssertEqual(response.chunksDeleted, 500)
        XCTAssertEqual(response.durationMs, 1234)
    }

    // MARK: - Client-Level Deletion Tests

    func testDeleteDocumentsSendsCorrectRequest() async throws {
        let client = makeMockClient()
        var capturedRequest: URLRequest?

        MockURLProtocol.requestHandler = { request in
            capturedRequest = request
            let json = #"{"documents_deleted":2,"chunks_deleted":10,"duration_ms":50}"#
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200,
                httpVersion: nil, headerFields: nil
            )!
            return (response, json.data(using: .utf8)!)
        }

        let result = try await client.deleteDocuments(ids: ["doc1", "doc2"])

        // Verify request
        XCTAssertEqual(capturedRequest?.httpMethod, "DELETE")
        XCTAssertTrue(capturedRequest!.url!.path.hasSuffix("/api/v1/documents"))
        XCTAssertEqual(capturedRequest?.value(forHTTPHeaderField: "Content-Type"), "application/json")

        // Verify body contains the document IDs
        let bodyData = capturedRequest!.httpBody!
        let body = try JSONDecoder().decode(DeleteRequest.self, from: bodyData)
        XCTAssertEqual(body.documentIds, ["doc1", "doc2"])

        // Verify response
        XCTAssertEqual(result.documentsDeleted, 2)
        XCTAssertEqual(result.chunksDeleted, 10)
        XCTAssertEqual(result.durationMs, 50)
    }

    func testDeleteSingleDocument() async throws {
        let client = makeMockClient()
        var capturedBody: DeleteRequest?

        MockURLProtocol.requestHandler = { request in
            capturedBody = try JSONDecoder().decode(DeleteRequest.self, from: request.httpBody!)
            let json = #"{"documents_deleted":1,"chunks_deleted":5,"duration_ms":12}"#
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200,
                httpVersion: nil, headerFields: nil
            )!
            return (response, json.data(using: .utf8)!)
        }

        let result = try await client.deleteDocument(id: "single-doc")

        XCTAssertEqual(capturedBody?.documentIds, ["single-doc"])
        XCTAssertEqual(result.documentsDeleted, 1)
        XCTAssertEqual(result.chunksDeleted, 5)
    }

    func testClearAll() async throws {
        let client = makeMockClient()
        var capturedRequest: URLRequest?

        MockURLProtocol.requestHandler = { request in
            capturedRequest = request
            let json = #"{"chunks_deleted":500,"duration_ms":1234}"#
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 200,
                httpVersion: nil, headerFields: nil
            )!
            return (response, json.data(using: .utf8)!)
        }

        let result = try await client.clearAll()

        XCTAssertEqual(capturedRequest?.httpMethod, "POST")
        XCTAssertTrue(capturedRequest!.url!.path.hasSuffix("/api/v1/index/clear"))
        XCTAssertNil(capturedRequest?.httpBody)
        XCTAssertEqual(result.chunksDeleted, 500)
        XCTAssertEqual(result.durationMs, 1234)
    }

    func testDeleteReturnsServerError() async throws {
        let client = makeMockClient()

        MockURLProtocol.requestHandler = { request in
            let json = #"{"code":"NOT_FOUND","message":"Document not found"}"#
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 404,
                httpVersion: nil, headerFields: nil
            )!
            return (response, json.data(using: .utf8)!)
        }

        do {
            _ = try await client.deleteDocuments(ids: ["nonexistent"])
            XCTFail("Expected error to be thrown")
        } catch let error as DIndexError {
            if case .serverError(let code, let message) = error {
                XCTAssertEqual(code, "NOT_FOUND")
                XCTAssertEqual(message, "Document not found")
            } else {
                XCTFail("Expected serverError, got \(error)")
            }
        }
    }

    func testDeleteReturnsUnauthorized() async throws {
        let client = makeMockClient()

        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!, statusCode: 401,
                httpVersion: nil, headerFields: nil
            )!
            return (response, Data())
        }

        do {
            _ = try await client.deleteDocuments(ids: ["doc1"])
            XCTFail("Expected error to be thrown")
        } catch let error as DIndexError {
            if case .unauthorized = error {
                // expected
            } else {
                XCTFail("Expected unauthorized, got \(error)")
            }
        }
    }
}
