import Foundation

/// Configuration for the DIndex client
public struct DIndexConfiguration: Sendable {
    /// The base URL for HTTP connections
    public let baseURL: URL?

    /// API key for authentication
    public let apiKey: String?

    /// Unix socket path for local connections (macOS only)
    public let socketPath: String?

    /// Request timeout interval
    public let timeoutInterval: TimeInterval

    /// Create configuration for HTTP transport
    /// - Parameters:
    ///   - baseURL: The base URL of the dindex server
    ///   - apiKey: Optional API key for authentication
    ///   - timeoutInterval: Request timeout (default: 30 seconds)
    public static func http(
        baseURL: URL,
        apiKey: String? = nil,
        timeoutInterval: TimeInterval = 30
    ) -> DIndexConfiguration {
        DIndexConfiguration(
            baseURL: baseURL,
            apiKey: apiKey,
            socketPath: nil,
            timeoutInterval: timeoutInterval
        )
    }

    #if os(macOS)
    /// Create configuration for Unix socket transport (macOS only)
    /// - Parameters:
    ///   - socketPath: Path to the Unix socket
    ///   - timeoutInterval: Request timeout (default: 30 seconds)
    public static func socket(
        path: String = "/tmp/dindex.sock",
        timeoutInterval: TimeInterval = 30
    ) -> DIndexConfiguration {
        DIndexConfiguration(
            baseURL: nil,
            apiKey: nil,
            socketPath: path,
            timeoutInterval: timeoutInterval
        )
    }
    #endif
}
