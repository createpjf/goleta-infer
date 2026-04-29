// MLXKernels.swift — Apple MLX kernel implementations for goleta-infer.
//
// Phase 1 stub status (2026-04-29): this module compiles but registers
// no kernels. The C-side ggml-mlx backend (in the LlamaCore xcframework)
// receives a NULL kernel table and stays in stub mode. Phase 2 fills in
// the actual @_cdecl bridges.
//
// Architecture:
//
//   MLXKernels.swift (this file)
//     │
//     │  goleta_mlx_register_kernels(&table)
//     ▼
//   ggml-mlx.cpp (in LlamaCore.xcframework)
//     │
//     │  std::atomic<table*> g_mlx_kernels.store(...)
//     ▼
//   ggml graph compute dispatcher
//     - reads g_mlx_kernels at op time
//     - calls table->dense_matmul_f16(...) etc. via C function pointer
//     - on NULL: falls through to ggml-cpu / ggml-metal
//
// Phase 2 task plan (one Swift file per task, all in this directory):
//   Task 2.1: DenseMatmul.swift     -> table.dense_matmul_f16
//   Task 2.2: Q4KMDequant.swift     -> table.dequant_q4km_to_f16
//   Task 2.3: RMSNorm.swift         -> table.rms_norm
//   Task 2.4: RoPE.swift            -> table.rope
//   Task 2.5: SDPA.swift            -> table.sdpa
//   Task 2.6: KVCache.swift         -> table.kv_cache_create/append/read/destroy
//
// At Phase 2 completion, MLXKernels.bootstrap() builds the table from
// the @_cdecl symbols and calls goleta_mlx_register_kernels(&table).

import Foundation
// The xcframework's modulemap calls itself `llama` (not LlamaCore — that's
// only the SwiftPM target name). All C symbols from ggml + ggml-mlx + llama
// are reached via this single import.
import llama

public enum MLXKernels {
    /// Register the MLX kernel table with the C-side ggml-mlx backend.
    ///
    /// Phase 1: passes NULL, leaves backend in stub mode.
    /// Phase 2: builds a `goleta_mlx_kernel_table` populated with @_cdecl
    /// function pointers and registers it.
    ///
    /// Idempotent — safe to call multiple times. Returns true if the C
    /// side accepted the registration (or the deregistration request).
    @discardableResult
    public static func bootstrap() -> Bool {
        // Phase 1: no kernels yet. Explicit NULL deregisters / keeps stub.
        return goleta_mlx_register_kernels(nil)
    }

    /// True iff the C-side backend has a non-NULL kernel table loaded.
    /// Use this to verify bootstrap() ran in tests.
    public static var kernelsAvailable: Bool {
        return goleta_mlx_kernels_available()
    }
}
