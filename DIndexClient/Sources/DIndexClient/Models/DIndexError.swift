import Foundation

/// Errors that can occur when interacting with DIndex
public enum DIndexError: Error, LocalizedError, Sendable {
    /// Network request failed
    case networkError(underlying: Error)

    /// Server returned an error response
    case serverError(code: String, message: String)

    /// Failed to encode request
    case encodingError(underlying: Error)

    /// Failed to decode response
    case decodingError(underlying: Error)

    /// Invalid URL provided
    case invalidURL(String)

    /// Unauthorized - invalid or missing API key
    case unauthorized

    /// Connection to Unix socket failed
    case socketConnectionFailed(path: String)

    /// Request timeout
    case timeout

    public var errorDescription: String? {
        switch self {
        case .networkError(let underlying):
            return "Network error: \(underlying.localizedDescription)"
        case .serverError(let code, let message):
            return "Server error [\(code)]: \(message)"
        case .encodingError(let underlying):
            return "Failed to encode request: \(underlying.localizedDescription)"
        case .decodingError(let underlying):
            return "Failed to decode response: \(underlying.localizedDescription)"
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        case .unauthorized:
            return "Unauthorized: Invalid or missing API key"
        case .socketConnectionFailed(let path):
            return "Failed to connect to Unix socket: \(path)"
        case .timeout:
            return "Request timed out"
        }
    }
}

/// Error response from the server
public struct ErrorResponse: Codable, Sendable {
    public let code: String
    public let message: String
}
