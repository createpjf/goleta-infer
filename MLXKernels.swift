// MLXKernels.swift — Apple MLX kernel implementations for goleta-infer
// (Phase 0 placeholder).
//
// **STATUS: Phase 0 placeholder — NOT YET WIRED INTO Package.swift.**
//
// This file exists so Phase 1 (W25-W28) work has an obvious landing
// pad. It is intentionally absent from the SwiftPM target list:
// pulling in MLX-Swift as a dependency now would burden every
// `swift build` consumer of goleta-infer with a ~150MB download
// that does nothing today (every entry returns the stub fall-back
// path in ggml-mlx.c).
//
// What lands here in Phase 1:
//
// 1. **Imports**
//        import MLX
//        import MLXNN
//        // see https://github.com/ml-explore/mlx-swift
//
// 2. **@_cdecl bridges** matching the four sparse-activation hot-path
//    declarations in `ggml-mlx.h`. Pattern:
//
//        @_cdecl("goleta_mlx_sparse_matmul_impl")
//        public func sparseMatmulImpl(
//            ctx: OpaquePointer?,
//            dst: OpaquePointer?,
//            weightHot: OpaquePointer?,
//            input: OpaquePointer?,
//            hotIndices: UnsafePointer<Int32>?,
//            nHot: Int32
//        ) -> Bool {
//            // marshal ggml_tensor → MLX array
//            // dispatch via MLX kernel
//            // marshal back
//        }
//
//    The C side (ggml-mlx.c Phase 1 version) calls these symbols
//    instead of returning the stub `false`.
//
// 3. **Kernels**
//    - sparseMatmul(weight: MLXArray, input: MLXArray, hotIndices: MLXArray) -> MLXArray
//        Hot-neuron-only matmul. Uses MLXArray.gather for hot index
//        materialization, MLX.matmul for the reduced GEMM.
//    - predictActivation(predictorW: MLXArray, input: MLXArray) -> MLXArray
//        Small MLP that predicts which neurons are "hot" given input.
//        Tiny model, fits in unified memory — full MLX dispatch.
//    - rmsNorm(src: MLXArray, weight: MLXArray, eps: Float) -> MLXArray
//        Use MLXNN.RMSNorm directly.
//    - rope(src: MLXArray, nPast: Int, nDims: Int, mode: Int) -> MLXArray
//        MLX has rotary embedding primitives in MLXNN; thin wrapper.
//
// 4. **Quantization helpers** (Phase 1.5)
//    Bridging GGUF Q4_0 / Q4_K_M weight blocks → MLX's quantized
//    tensor format. Apple's MLX has Q4 support natively; mapping
//    needs care for the GGUF block layout vs MLX's layout.
//
// 5. **Memory pool**
//    MLX uses unified memory. Map ggml_tensor's data pointer
//    directly into an MLXArray view when the tensor is host-resident.
//    GPU-resident tensors need a copy through the MLX context's
//    device buffer.
//
// **Phase 0 today**: this file is read by hand for design-doc
// purposes and committed to the repo so the structure is visible.
// It does NOT compile (no MLX import). Adding it to Package.swift
// is a Phase 1 W25 task.
//
// **Spike measurement separately**: Phase 0 D4-5 builds a
// standalone Swift benchmark project (in `mlx-spike/` directory)
// that pulls MLX-Swift directly and times sparse_matmul against
// ggml-metal's existing Q4_K_M matmul. That's where the
// MLX-vs-Metal performance comparison happens; if MLX wins by
// ≥1.5x, Phase 1 W25 promotes the spike code into this file.

#if false
// Phase 1 placeholder — uncomment after `swift package add-dependency
// https://github.com/ml-explore/mlx-swift` lands in Phase 1 W25.

// import Foundation
// import MLX
// import MLXNN
//
// @_cdecl("goleta_mlx_sparse_matmul_impl")
// public func goleta_mlx_sparse_matmul_impl(/* ... */) -> Bool {
//     // Phase 1 implementation
//     return false
// }
//
// (etc.)
#endif
