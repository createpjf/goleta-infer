// DenseMatmul.swift — Phase 2 Task 2.1a
//
// Two layers:
//   1. `MLXKernels.denseMatmulF16(a:b:)` — Swift API. Today a thin wrapper
//      around MLX-Swift's matmul(). Phase 2.5+ may specialize for batch
//      sizes or fuse with rms_norm; the wrapper makes that swap-in
//      transparent to callers.
//   2. `goleta_mlx_dense_matmul_impl(...)` — @_cdecl bridge invoked
//      through the kernel table by ggml-mlx.cpp's graph_compute. Marshals
//      raw Float16 buffer pointers (matching ggml's tensor data layout)
//      into MLXArray, runs the kernel, copies result back. The C side
//      passes void* pointers; we cast them via UnsafeRawPointer so the
//      kernel table's `bool (*)(const void *, ...)` signature matches
//      the @_cdecl ABI without unsafeBitCast on the Swift end.

import Foundation
import MLX
import llama  // C-side ggml-mlx symbols

extension MLXKernels {
    /// Dense fp16 matmul. Output dtype matches MLX-Swift's matmul semantics
    /// (fp16 in, fp16 out for fp16 inputs). Phase 2.1a delegates directly to
    /// `MLX.matmul`; later phases may specialize.
    public static func denseMatmulF16(a: MLXArray, b: MLXArray) -> MLXArray {
        return matmul(a, b)
    }
}

// MARK: - C bridge (called from ggml-mlx.cpp graph_compute via the kernel table)

/// Implementation of the `dense_matmul_f16` slot in `goleta_mlx_kernel_table`
/// (see `ggml-mlx.h`). Declared as `@convention(c)` so it's a plain C function
/// pointer at the ABI level. ggml-mlx.cpp invokes it through the kernel
/// table — never by symbol name — so we don't need `@_cdecl`. (Avoiding
/// `@_cdecl` is intentional: `@_cdecl` causes Swift to re-emit the C symbol
/// at every reference site, producing a duplicate-symbol linker error when
/// MLXKernels.swift assigns this function to the kernel table slot.)
///
/// All buffers must be host-resident contiguous fp16. Caller owns lifetimes;
/// we only borrow during the matmul + copy back to outData.
internal let _denseMatmulF16Bridge: @convention(c) (
    UnsafeRawPointer?, Int32, Int32,
    UnsafeRawPointer?, Int32, Int32,
    UnsafeMutableRawPointer?
) -> Bool = { aData, aRows, aCols, bData, bRows, bCols, outData in
    guard let aData, let bData, let outData,
          aCols == bRows,
          aRows > 0, aCols > 0, bCols > 0 else { return false }

    let m = Int(aRows), k = Int(aCols), n = Int(bCols)

    // Reinterpret the raw pointers as Float16 buffers. Lifetimes are caller-
    // owned; we only borrow during the matmul + copy back to outData.
    let aPtr = aData.assumingMemoryBound(to: Float16.self)
    let bPtr = bData.assumingMemoryBound(to: Float16.self)
    let outPtr = outData.assumingMemoryBound(to: Float16.self)

    let aBuf = UnsafeBufferPointer(start: aPtr, count: m * k)
    let bBuf = UnsafeBufferPointer(start: bPtr, count: k * n)

    // MLXArray copies from these buffers; safe even after aBuf/bBuf go out
    // of scope.
    let aArr = MLXArray(aBuf, [m, k])
    let bArr = MLXArray(bBuf, [k, n])

    let cArr = MLXKernels.denseMatmulF16(a: aArr, b: bArr)
    eval(cArr)  // force completion before reading

    // Copy result back into ggml's output tensor buffer.
    cArr.asArray(Float16.self).withUnsafeBufferPointer { src in
        guard let srcBase = src.baseAddress else { return }
        memcpy(outPtr, srcBase, m * n * MemoryLayout<Float16>.size)
    }

    return true
}
