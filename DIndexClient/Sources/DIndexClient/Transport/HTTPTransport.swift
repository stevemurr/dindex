import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// HTTP transport for communicating with dindex server
///
/// Thread safety: `baseURL` and `apiKey` are immutable, `URLSession` is thread-safe,
/// and `JSONDecoder`/`JSONEncoder` are shared static instances.
public final class HTTPTransport: @unchecked Sendable {
    private static let decoder = JSONDecoder()
    private static let encoder = JSONEncoder()

    private let baseURL: URL
    private let apiKey: String?
    private let session: URLSession

    /// Create a new HTTP transport
    /// - Parameters:
    ///   - baseURL: The base URL of the dindex server (e.g., http://localhost:8080)
    ///   - apiKey: Optional API key for authentication
    ///   - session: URLSession to use (defaults to shared)
    public init(baseURL: URL, apiKey: String? = nil, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.session = session
    }

    public func get<T: Decodable>(path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        addHeaders(to: &request)

        return try await performRequest(request)
    }

    public func post<T: Decodable, B: Encodable>(path: String, body: B) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addHeaders(to: &request)

        do {
            request.httpBody = try Self.encoder.encode(body)
        } catch {
            throw DIndexError.encodingError(underlying: error)
        }

        return try await performRequest(request)
    }

    public func post<T: Decodable>(path: String) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        addHeaders(to: &request)

        return try await performRequest(request)
    }

    public func delete<T: Decodable, B: Encodable>(path: String, body: B) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        addHeaders(to: &request)

        do {
            request.httpBody = try Self.encoder.encode(body)
        } catch {
            throw DIndexError.encodingError(underlying: error)
        }

        return try await performRequest(request)
    }

    private func addHeaders(to request: inout URLRequest) {
        if let apiKey = apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
    }

    private func performRequest<T: Decodable>(_ request: URLRequest) async throws -> T {
        let data: Data
        let response: URLResponse

        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw DIndexError.networkError(underlying: error)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw DIndexError.networkError(underlying: NSError(
                domain: "DIndexClient",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid response type"]
            ))
        }

        // Handle error responses
        if httpResponse.statusCode == 401 {
            throw DIndexError.unauthorized
        }

        if httpResponse.statusCode >= 400 {
            // Try to decode error response
            if let errorResponse = try? Self.decoder.decode(ErrorResponse.self, from: data) {
                throw DIndexError.serverError(code: errorResponse.code, message: errorResponse.message)
            }
            throw DIndexError.serverError(
                code: "HTTP_\(httpResponse.statusCode)",
                message: String(data: data, encoding: .utf8) ?? "Unknown error"
            )
        }

        // Decode successful response
        do {
            return try Self.decoder.decode(T.self, from: data)
        } catch {
            throw DIndexError.decodingError(underlying: error)
        }
    }

    // MARK: - SSE Streaming

    /// Stream Server-Sent Events from the given path
    ///
    /// - Parameter path: The API path to stream from
    /// - Returns: An async stream of decoded events
    public func streamSSE<T: Decodable>(path: String) -> AsyncThrowingStream<T, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let url = baseURL.appendingPathComponent(path)
                    var request = URLRequest(url: url)
                    request.httpMethod = "GET"
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    addHeaders(to: &request)

                    #if DEBUG
                    print("[SSE] Connecting to: \(url.absoluteString)")
                    #endif

                    let (bytes, response) = try await session.bytes(for: request)

                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw DIndexError.networkError(underlying: NSError(
                            domain: "DIndexClient",
                            code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "Invalid response type"]
                        ))
                    }

                    if httpResponse.statusCode == 401 {
                        throw DIndexError.unauthorized
                    }

                    if httpResponse.statusCode >= 400 {
                        #if DEBUG
                        print("[SSE] HTTP error: \(httpResponse.statusCode)")
                        #endif
                        throw DIndexError.serverError(
                            code: "HTTP_\(httpResponse.statusCode)",
                            message: "SSE connection failed"
                        )
                    }

                    #if DEBUG
                    print("[SSE] Connected successfully, status: \(httpResponse.statusCode)")
                    #endif

                    // Parse SSE stream
                    var dataBuffer = ""

                    var lineCount = 0
                    for try await line in bytes.lines {
                        if Task.isCancelled {
                            #if DEBUG
                            print("[SSE] Task cancelled after \(lineCount) lines")
                            #endif
                            break
                        }

                        lineCount += 1

                        // event: line or empty line signals end of previous event
                        if line.hasPrefix("event:") || line.isEmpty {
                            if !dataBuffer.isEmpty {
                                if let data = dataBuffer.data(using: .utf8) {
                                    do {
                                        let event = try Self.decoder.decode(T.self, from: data)
                                        continuation.yield(event)
                                    } catch {
                                        #if DEBUG
                                        print("[SSE] Decode error: \(error), data: \(dataBuffer.prefix(200))")
                                        #endif
                                    }
                                }
                                dataBuffer = ""
                            }
                            continue
                        }

                        if line.hasPrefix("data:") {
                            let data = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)
                            if !dataBuffer.isEmpty {
                                dataBuffer += "\n"
                            }
                            dataBuffer += data
                        }
                        // Ignore comments (lines starting with :) and other fields
                    }

                    #if DEBUG
                    print("[SSE] Stream ended after \(lineCount) lines")
                    #endif
                    continuation.finish()
                } catch {
                    #if DEBUG
                    print("[SSE] Stream error: \(error)")
                    #endif
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
