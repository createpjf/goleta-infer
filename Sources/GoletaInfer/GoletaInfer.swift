// GoletaInfer.swift — umbrella public API consumed by Goleta app + GoletaEngine.
//
// This is what `import GoletaInfer` gives you:
//   - The C/C++ llama.cpp + ggml-mlx libs (via the LlamaCore xcframework)
//   - The Swift MLXKernels module that registers Apple MLX kernels with
//     the ggml-mlx backend
//   - A thin Swift wrapper exposing the goleta-infer-specific knobs
//     (currently just bootstrap; Phase 3 adds a higher-level chat API
//     consumed by PowerInferProvider).
//
// Calling order at app launch:
//
//     import GoletaInfer
//     GoletaInfer.bootstrap()
//     // ... now llama_init_from_model() will use the MLX backend for
//     // ops with N >= 2048 (and fall back to ggml-metal otherwise).

import Foundation
import MLXKernels

public enum GoletaInfer {
    /// Wire up MLX kernels so subsequent llama_decode() calls dispatch
    /// to MLX where supported. Idempotent. Call once at app launch.
    @discardableResult
    public static func bootstrap() -> Bool {
        return MLXKernels.bootstrap()
    }

    /// True iff GoletaInfer has registered MLX kernels with the
    /// C-side backend. Useful for debug / Settings UX surfacing.
    public static var mlxKernelsActive: Bool {
        return MLXKernels.kernelsAvailable
    }
}
