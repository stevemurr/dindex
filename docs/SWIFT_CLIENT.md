# Swift Client

A native Swift client library for iOS, macOS, and visionOS apps.

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(path: "../DIndexClient")  // or use a URL
]
```

## Usage

```swift
import DIndexClient

let client = DIndexClient(baseURL: URL(string: "http://localhost:8081")!)

// Search
let results = try await client.search(query: "machine learning", topK: 10)

// Search with filters
let filters = SearchFilters(
    metadataEquals: ["source": "arxiv"],
    metadataContains: ["category": ["ml"]]
)
let filtered = try await client.search(query: "transformers", topK: 5, filters: filters)

// Index a document
try await client.index(content: "Document text...", title: "My Doc", url: "https://example.com")

// Delete documents
try await client.deleteDocuments(ids: ["doc123", "doc456"])

// Clear the entire index
try await client.clearAll()

// Health check
let healthy = try await client.health()
```
