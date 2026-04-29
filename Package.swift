// swift-tools-version: 5.10
//
// goleta-infer SwiftPM Package.
//
// Architecture (Pattern 2 from plan §2.0 design fork):
//   1. `LlamaCore` — binary target wrapping the C/C++ xcframework
//      (ggml + ggml-cpu + ggml-metal + ggml-mlx + llama). Built by
//      `scripts/goleta-xcframework-mac.sh` for macOS arm64 only;
//      multi-platform build via the upstream `build-xcframework.sh`.
//   2. `MLXKernels` — Swift target that registers MLX-backed kernel
//      function pointers with the C-side ggml-mlx backend at runtime.
//      Without this target loaded, ggml-mlx falls through to ggml-cpu
//      (Phase 1 stub mode).
//   3. `GoletaInfer` — umbrella Swift target that consumers (Goleta
//      app + GoletaEngine) import. On `bootstrap()`, registers the
//      MLX kernel table with the backend so subsequent llama_decode
//      calls dispatch to MLX where supported.
//
// Build flow for local dev:
//   ./scripts/goleta-xcframework-mac.sh Release   # produces xcframework
//   swift build                                    # links Swift targets
//
// For CI / shipping: cut a release, upload xcframework.zip, switch
// `.binaryTarget(path:)` to `.binaryTarget(url:checksum:)`.

import PackageDescription

let package = Package(
    name: "goleta-infer",
    platforms: [
        .macOS(.v14),  // mlx-swift requires 14+
    ],
    products: [
        // Goleta consumers import this; gets both C/C++ libs + Swift MLX kernels.
        .library(name: "GoletaInfer", targets: ["GoletaInfer"]),

        // Standalone product for consumers that don't need MLX kernels
        // (testing, headless tools). Just the C/C++ libs.
        .library(name: "LlamaCore", targets: ["LlamaCore"]),
    ],
    dependencies: [
        // mlx-swift pinned at 0.21+ per Phase 0 spike validation.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0"),
    ],
    targets: [
        // -------------------------------------------------------------------
        // Binary target: ggml + llama as a prebuilt xcframework.
        //
        // Produced by ./scripts/goleta-xcframework-mac.sh and committed
        // .gitignore'd at build-apple/. Local-path .binaryTarget for now;
        // Phase 4 freeze switches to URL+checksum for distribution.
        // -------------------------------------------------------------------
        .binaryTarget(
            name: "LlamaCore",
            path: "build-apple/goleta-infer.xcframework"
        ),

        // -------------------------------------------------------------------
        // Swift target: MLX kernel implementations + @_cdecl bridges.
        //
        // Phase 1 stub: file exists but registers no kernels. Phase 2
        // fills in dense_matmul, rms_norm, rope, sdpa, kv_cache.
        // -------------------------------------------------------------------
        .target(
            name: "MLXKernels",
            dependencies: [
                "LlamaCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/MLXKernels",
            swiftSettings: [
                .unsafeFlags(["-O"], .when(configuration: .release)),
            ]
        ),

        // -------------------------------------------------------------------
        // Umbrella target Goleta imports. Pulls in C/C++ libs + Swift kernels.
        //
        // Exposes GoletaInfer.bootstrap() — call once at app launch to
        // register the MLX kernel table with the C-side ggml-mlx backend.
        // -------------------------------------------------------------------
        .target(
            name: "GoletaInfer",
            dependencies: ["LlamaCore", "MLXKernels"],
            path: "Sources/GoletaInfer"
        ),

        // -------------------------------------------------------------------
        // Tests: Swift-side validation that the kernel table registers
        // correctly with the C backend. Phase 2 task tests live here too.
        // -------------------------------------------------------------------
        .testTarget(
            name: "MLXKernelsTests",
            dependencies: ["MLXKernels"],
            path: "Tests/MLXKernelsTests"
        ),
    ]
)
