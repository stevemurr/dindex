import Foundation

/// Protocol for transport implementations
public protocol Transport: Sendable {
    /// Perform a GET request
    func get<T: Decodable>(path: String) async throws -> T

    /// Perform a POST request with JSON body
    func post<T: Decodable, B: Encodable>(path: String, body: B) async throws -> T

    /// Perform a POST request without body
    func post<T: Decodable>(path: String) async throws -> T
}
