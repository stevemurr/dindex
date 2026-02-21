// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DIndexClient",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "DIndexClient",
            targets: ["DIndexClient"]
        )
    ],
    targets: [
        .target(
            name: "DIndexClient",
            path: "Sources/DIndexClient"
        ),
        .testTarget(
            name: "DIndexClientTests",
            dependencies: ["DIndexClient"],
            path: "Tests/DIndexClientTests"
        )
    ]
)
