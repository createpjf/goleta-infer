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

// MARK: - ggml-shaped MUL_MAT bridge (Phase 2 Task 2.1b)

/// Implementation of the `mul_mat_f16_ggml` slot in `goleta_mlx_kernel_table`.
///
/// Computes `out[m, n] = sum_k input[m, k] * weight[n, k]`, i.e.
/// `out = input @ weight.T`. This is the layout ggml MUL_MAT operates on
/// natively: the C-side dispatcher in ggml-mlx.cpp passes src1 as input
/// (row-major [M, K]) and src0 as weight (row-major [N, K]) directly,
/// without any caller-side transpose.
///
/// The transpose happens inside MLX as a stride-only view (no memcpy),
/// then a single `MLX.matmul(input, weight.transposed())` runs on the GPU.
internal let _mulMatF16GgmlBridge: @convention(c) (
    UnsafeRawPointer?, Int32, Int32,        // input, M, K
    UnsafeRawPointer?, Int32, Int32,        // weight, N, K_w (must == K)
    UnsafeMutableRawPointer?
) -> Bool = { inputData, mIn, kIn, weightData, nIn, kwIn, outData in
    guard let inputData, let weightData, let outData,
          kIn == kwIn,
          mIn > 0, kIn > 0, nIn > 0 else { return false }

    let m = Int(mIn), k = Int(kIn), n = Int(nIn)

    let inputPtr  = inputData.assumingMemoryBound(to: Float16.self)
    let weightPtr = weightData.assumingMemoryBound(to: Float16.self)
    let outPtr    = outData.assumingMemoryBound(to: Float16.self)

    let inputBuf  = UnsafeBufferPointer(start: inputPtr,  count: m * k)
    let weightBuf = UnsafeBufferPointer(start: weightPtr, count: n * k)

    // input: [M, K] row-major = MLXArray with shape [m, k]
    // weight: ggml stores it as [N, K] row-major = MLXArray with shape [n, k]
    // For matmul we need weight as [K, N]; transposed() is a stride view.
    let inputArr  = MLXArray(inputBuf,  [m, k])
    let weightArr = MLXArray(weightBuf, [n, k])
    let cArr = matmul(inputArr, weightArr.transposed())
    eval(cArr)

    cArr.asArray(Float16.self).withUnsafeBufferPointer { src in
        guard let srcBase = src.baseAddress else { return }
        memcpy(outPtr, srcBase, m * n * MemoryLayout<Float16>.size)
    }

    return true
}
