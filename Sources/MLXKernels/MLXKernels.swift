// MLXKernels.swift — Apple MLX kernel implementations for goleta-infer.
//
// Phase 2 status (2026-04-30): the kernel table starts taking shape.
// Each task lights up one slot:
//
//   Task 2.1a: DenseMatmul.swift  -> table.dense_matmul_f16   ✅ DONE
//   Task 2.2:  Q4KMDequant.swift  -> table.dequant_q4km_to_f16
//   Task 2.3:  RMSNorm.swift      -> table.rms_norm
//   Task 2.4:  RoPE.swift         -> table.rope
//   Task 2.5:  SDPA.swift         -> table.sdpa
//   Task 2.6:  KVCache.swift      -> table.kv_cache_*
//
// Architecture:
//
//   MLXKernels.swift (this file) — bootstrap() builds the table
//     │
//     │  goleta_mlx_register_kernels(&table)   (C copies into static storage)
//     ▼
//   ggml-mlx.cpp (in LlamaCore.xcframework)
//     │
//     │  std::atomic<table*> g_mlx_kernels.store(...)
//     ▼
//   ggml graph_compute dispatcher (Task 2.1b)
//     - reads g_mlx_kernels at op time
//     - calls table->dense_matmul_f16(...) etc. via C function pointer
//     - on NULL or NULL fn-ptr: falls through to ggml-cpu / ggml-metal
//
// MLX evaluates lazily — bootstrap() is therefore cheap. The actual
// MLX device init happens on the first kernel call.

import Foundation
// The xcframework's modulemap calls itself `llama` (not LlamaCore — that's
// only the SwiftPM target name). All C symbols from ggml + ggml-mlx + llama
// are reached via this single import.
import llama

public enum MLXKernels {

    /// Register the MLX kernel table with the C-side ggml-mlx backend.
    ///
    /// Idempotent — safe to call multiple times; the C side serializes on
    /// a mutex and copies the table into static storage so the local struct
    /// here doesn't need to outlive the call.
    ///
    /// Returns true if the C side accepted the registration. False indicates
    /// either an ABI mismatch (recompile MLXKernels against a fresh ggml-mlx.h)
    /// or a deeper integration bug.
    @discardableResult
    public static func bootstrap() -> Bool {
        var table = makeKernelTable()
        return withUnsafePointer(to: &table) { ptr in
            goleta_mlx_register_kernels(ptr)
        }
    }

    /// Tear down: unregister kernels, return backend to stub mode. Used by
    /// tests to isolate state and during app shutdown for cleanliness.
    @discardableResult
    public static func shutdown() -> Bool {
        return goleta_mlx_register_kernels(nil)
    }

    /// True iff the C-side backend has a non-NULL kernel table loaded.
    /// Use this to verify bootstrap() ran in tests.
    public static var kernelsAvailable: Bool {
        return goleta_mlx_kernels_available()
    }

    // MARK: - Internal: table construction

    /// Build the kernel table from the @_cdecl symbols defined in this
    /// module. Phase 2 tasks add slots; nil entries mean "kernel not
    /// implemented yet — ggml-mlx falls through to ggml-cpu for that op."
    private static func makeKernelTable() -> goleta_mlx_kernel_table {
        var table = goleta_mlx_kernel_table()
        table.abi_version = Int32(GOLETA_MLX_KERNEL_ABI_VERSION)

        // Task 2.1a: dense fp16 matmul (generic primitive: out = a @ b).
        // The bridge is a @convention(c) closure constant (NOT @_cdecl) —
        // see DenseMatmul.swift for why.
        table.dense_matmul_f16 = _denseMatmulF16Bridge

        // Task 2.1b: ggml-shaped MUL_MAT (out = input @ weight.T), what
        // ggml-mlx.cpp's graph_compute actually dispatches to.
        table.mul_mat_f16_ggml = _mulMatF16GgmlBridge

        // Phase 2 remaining slots default-initialize to nil. As tasks 2.2-2.6
        // land, they add their @_cdecl symbol here. ggml-mlx checks for NULL
        // per-op before dispatching; absent kernels fall through to ggml-cpu.

        return table
    }
}
