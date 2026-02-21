import Foundation

/// HTTP transport for communicating with dindex server
public final class HTTPTransport: Transport, Sendable {
    private let baseURL: URL
    private let apiKey: String?
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    /// Create a new HTTP transport
    /// - Parameters:
    ///   - baseURL: The base URL of the dindex server (e.g., http://localhost:8080)
    ///   - apiKey: Optional API key for authentication
    ///   - session: URLSession to use (defaults to shared)
    public init(baseURL: URL, apiKey: String? = nil, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.session = session
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
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
            request.httpBody = try encoder.encode(body)
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
            if let errorResponse = try? decoder.decode(ErrorResponse.self, from: data) {
                throw DIndexError.serverError(code: errorResponse.code, message: errorResponse.message)
            }
            throw DIndexError.serverError(
                code: "HTTP_\(httpResponse.statusCode)",
                message: String(data: data, encoding: .utf8) ?? "Unknown error"
            )
        }

        // Decode successful response
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw DIndexError.decodingError(underlying: error)
        }
    }
}
