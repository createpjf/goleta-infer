// SDPA.swift — Phase 2 Task 2.5
//
// Scaled dot-product attention bridge over MLX-Swift's
// `MLXFast.scaledDotProductAttention`. Apple's implementation dispatches
// to an optimized Metal kernel when the query sequence length is 1
// (i.e., the decode hot path), and handles other cases with regular
// MLX ops. Supports MHA, GQA, and MQA via the n_heads / n_kv_heads
// parameter split.
//
// Layout convention this bridge accepts:
//   queries: [n_heads,    seq_len_q,  head_dim]   (batch=1 implied)
//   keys:    [n_kv_heads, seq_len_kv, head_dim]
//   values:  [n_kv_heads, seq_len_kv, head_dim]
//   output:  [n_heads,    seq_len_q,  head_dim]
//
// MLX expects 4D arrays [B, n_heads, L, head_dim]; we add the batch
// axis here. For Phase 2.5's first pass we don't take a mask — Phase
// 2.6/2.7 add KV-cache + causal mask wiring.

import Foundation
import MLX
import llama  // C-side ggml-mlx symbols

extension MLXKernels {
    /// Run scaled dot-product attention.
    /// - Parameters:
    ///   - q/k/v: [n_heads_or_kv, seqLen, headDim]
    ///   - scale: typically 1/sqrt(headDim)
    public static func sdpaF16(
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float
    ) -> MLXArray {
        // MLX's SDPA expects 4D [batch, n_heads, seqLen, headDim].
        // Add unit batch dim with expandedDimensions.
        let q4 = q.expandedDimensions(axis: 0)
        let k4 = k.expandedDimensions(axis: 0)
        let v4 = v.expandedDimensions(axis: 0)
        let result = MLXFast.scaledDotProductAttention(
            queries: q4, keys: k4, values: v4, scale: scale, mask: nil
        )
        // Squeeze the batch axis back out → [n_heads, seqLen, headDim]
        return result.squeezed(axis: 0)
    }
}

// MARK: - C bridge

/// Slot: `goleta_mlx_kernel_table.sdpa`. Phase 2.5 treats the input
/// buffers as packed [heads, seq_len, head_dim] fp16. seq_len applies
/// to both q and kv (Phase 2.6 splits them once KV cache is wired).
internal let _sdpaF16Bridge: @convention(c) (
    UnsafeRawPointer?,        // q
    UnsafeRawPointer?,        // k
    UnsafeRawPointer?,        // v
    Int32, Int32,             // n_heads, n_kv_heads
    Int32, Int32,             // head_dim, seq_len
    UnsafeMutableRawPointer?  // out
) -> Bool = { qData, kData, vData, nHeads, nKvHeads, headDim, seqLen, outData in
    guard let qData, let kData, let vData, let outData,
          nHeads > 0, nKvHeads > 0, headDim > 0, seqLen > 0,
          (nHeads % nKvHeads) == 0 else { return false }

    let h = Int(nHeads), kh = Int(nKvHeads)
    let hd = Int(headDim), s = Int(seqLen)

    let qPtr = qData.assumingMemoryBound(to: Float16.self)
    let kPtr = kData.assumingMemoryBound(to: Float16.self)
    let vPtr = vData.assumingMemoryBound(to: Float16.self)
    let outPtr = outData.assumingMemoryBound(to: Float16.self)

    let qBuf = UnsafeBufferPointer(start: qPtr, count: h * s * hd)
    let kBuf = UnsafeBufferPointer(start: kPtr, count: kh * s * hd)
    let vBuf = UnsafeBufferPointer(start: vPtr, count: kh * s * hd)

    let qArr = MLXArray(qBuf, [h, s, hd])
    let kArr = MLXArray(kBuf, [kh, s, hd])
    let vArr = MLXArray(vBuf, [kh, s, hd])

    let scale = 1.0 / sqrt(Float(hd))
    let result = MLXKernels.sdpaF16(q: qArr, k: kArr, v: vArr, scale: scale)
    eval(result)

    result.asArray(Float16.self).withUnsafeBufferPointer { src in
        guard let srcBase = src.baseAddress else { return }
        memcpy(outPtr, srcBase, h * s * hd * MemoryLayout<Float16>.size)
    }

    return true
}
