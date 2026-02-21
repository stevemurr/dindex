#if os(macOS)
import Foundation
import Network

/// Unix socket transport for local daemon communication (macOS only)
///
/// Note: This transport provides basic connectivity checking.
/// Full bincode protocol implementation would be needed for complete
/// Unix socket support. For most use cases, the HTTP transport is recommended.
public final class UnixSocketTransport: @unchecked Sendable {
    private let socketPath: String

    /// Create a new Unix socket transport
    /// - Parameter socketPath: Path to the Unix socket (default: /tmp/dindex.sock)
    public init(socketPath: String = "/tmp/dindex.sock") {
        self.socketPath = socketPath
    }

    /// Check if the daemon socket exists
    public var socketExists: Bool {
        FileManager.default.fileExists(atPath: socketPath)
    }

    /// Check if the daemon is running by testing socket connectivity
    public func isAvailable() async -> Bool {
        guard socketExists else { return false }

        return await withCheckedContinuation { continuation in
            let endpoint = NWEndpoint.unix(path: socketPath)
            let connection = NWConnection(to: endpoint, using: .tcp)

            var didResume = false
            let resume = { (result: Bool) in
                guard !didResume else { return }
                didResume = true
                connection.cancel()
                continuation.resume(returning: result)
            }

            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    resume(true)
                case .failed, .cancelled:
                    resume(false)
                case .waiting:
                    // Timeout after 1 second
                    DispatchQueue.global().asyncAfter(deadline: .now() + 1.0) {
                        resume(false)
                    }
                default:
                    break
                }
            }

            connection.start(queue: .global())

            // Safety timeout
            DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) {
                resume(false)
            }
        }
    }
}
#endif
