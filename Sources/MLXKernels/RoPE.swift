// RoPE.swift — Phase 2 Task 2.4
//
// Rotary positional embedding bridge. MLX-Swift exposes `MLXFast.RoPE`
// which is the canonical implementation Apple ships. We forward to it
// with parameters mapped from ggml's tensor convention.
//
// ggml's MUL_MAT for attention typically gives an input shape
// [head_dim, n_heads, n_tokens] (in ggml ne[] order). MLX expects
// the input as [..., n_tokens, head_dim] with the rotated dimensions
// being the LAST axis. The bridge reshapes accordingly.
//
// Phase 2.4 status: handles the simple decode case (single batch,
// arbitrary n_tokens, single head dimension). Multi-head batched RoPE
// for prefill works with the same code path because MLXFast.RoPE
// applies along the last axis regardless of leading shape.

import Foundation
import MLX
import llama  // C-side ggml-mlx symbols

extension MLXKernels {
    /// Apply rotary positional embedding to `input` along the last axis.
    /// - Parameters:
    ///   - input: shape `[..., headDim]`
    ///   - dims: number of dimensions to rotate (typically headDim)
    ///   - offset: position offset (= n_past in ggml conventions)
    ///   - theta: RoPE base (10000 by default for Qwen-class models)
    public static func ropeF16(
        input: MLXArray,
        dims: Int,
        offset: Int,
        theta: Float
    ) -> MLXArray {
        return MLXFast.RoPE(
            input,
            dimensions: dims,
            traditional: false,    // Qwen / Llama use the non-interleaved layout
            base: theta,
            scale: 1.0,            // no RoPE scaling (Phase 2 may add YARN later)
            offset: offset
        )
    }
}

// MARK: - C bridge

/// Slot: `goleta_mlx_kernel_table.rope`. Treats the input buffer as
/// fp16 row-major `[nTokens, nDims]` (folded view of any leading
/// per-head dimensions; ggml's RoPE op runs on a single
/// (head, token) span at a time anyway).
internal let _ropeF16Bridge: @convention(c) (
    UnsafeRawPointer?,        // input
    Int32, Int32,             // n_tokens, n_dims
    Int32, Int32,             // head_dim, n_past
    Float,                    // theta
    UnsafeMutableRawPointer?  // out
) -> Bool = { inputData, nTokens, nDims, headDim, nPast, theta, outData in
    guard let inputData, let outData,
          nTokens > 0, nDims > 0, headDim > 0,
          nDims <= headDim else { return false }

    let t = Int(nTokens), d = Int(nDims), hd = Int(headDim), past = Int(nPast)

    let inputPtr = inputData.assumingMemoryBound(to: Float16.self)
    let outPtr   = outData.assumingMemoryBound(to: Float16.self)

    let inputBuf = UnsafeBufferPointer(start: inputPtr, count: t * hd)

    // MLX's RoPE requires ≥3 dimensions (batch + sequence + head_dim).
    // Pack the buffer as [batch=1, nTokens, headDim] for the call, squeeze
    // the batch axis back out before copying to the caller's output.
    let inputArr = MLXArray(inputBuf, [1, t, hd])
    let rotated = MLXKernels.ropeF16(
        input: inputArr,
        dims: d,
        offset: past,
        theta: theta
    )
    let result = rotated.squeezed(axis: 0)
    eval(result)

    result.asArray(Float16.self).withUnsafeBufferPointer { src in
        guard let srcBase = src.baseAddress else { return }
        memcpy(outPtr, srcBase, t * hd * MemoryLayout<Float16>.size)
    }

    return true
}
