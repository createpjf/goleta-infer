// swift-tools-version: 5.10
//
// goleta-infer Phase 0 spike harness.
//
// Standalone SwiftPM package that pulls in mlx-swift and benchmarks
// MLX matmul / sparse-matmul / RMSNorm against Apple's Accelerate
// (CPU SIMD baseline) and ggml-metal (the upstream PowerInfer
// inference backend).
//
// Why standalone (not part of the parent SwiftPM target):
// adding mlx-swift as a parent dependency would force every
// consumer of `goleta-infer` to download ~150MB of MLX framework
// for code that returns the Phase 0 stub. The spike answers the
// question "is MLX worth integrating" — once it does, Phase 1 W25
// promotes the chosen kernels into the parent target.
//
// Run with:
//     cd mlx-spike && swift run -c release MLXSpike

import PackageDescription

let package = Package(
    name: "MLXSpike",
    platforms: [
        // MLX-Swift requires macOS 13.3+ / iOS 16.4+.
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MLXSpike", targets: ["MLXSpike"]),
    ],
    dependencies: [
        // Apple's official MLX Swift bindings. Pinned to a recent
        // tag during the spike; Phase 1 will track main.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0"),
    ],
    targets: [
        .executableTarget(
            name: "MLXSpike",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/MLXSpike",
            swiftSettings: [
                .unsafeFlags(["-O"], .when(configuration: .release)),
            ]
        ),
    ]
)
