// Q4KMDequant.swift — Phase 2 Task 2.2
//
// Dequantizes ggml's Q4_K_M block format into fp16 buffers that MLX
// matmul can consume directly.
//
// Q4_K_M block layout (must stay binary-identical to ggml's block_q4_K
// in ggml/src/ggml-common.h, otherwise dequant produces garbage):
//
//     offset  size  field
//     ------  ----  -----
//        0      2   d         fp16 super-block scale
//        2      2   dmin      fp16 super-block min
//        4     12   scales    8 sub-blocks of (scale, min) pairs, each
//                             6-bit, packed into 12 bytes via the
//                             ggml-specific layout (see _qkScaleMin)
//       16    128   qs        256 quants packed as 4-bit nibbles
//                             (low nibble is "first" element)
//     ----  ----
//      total 144 bytes per QK_K=256 elements
//
// Dequant formula per element (matches ggml's dequantize_row_q4_K):
//
//     y[i] = (d * sc) * (q[i] & 0xF) - (dmin * m)         // first 32
//     y[i] = (d * sc) * (q[i] >> 4)  - (dmin * m)         // next 32
//
// where (sc, m) for the j-th pair-of-32-element-sub-blocks comes from
// _qkScaleMin(j, scales).
//
// Phase 2.2 status: CPU-side reference dequant. fp32 intermediate, fp16
// output. Runs at host speed; suitable for the matmul path because
// MLX's unified-memory matmul reads host memory directly. Phase 2.7+ may
// move this into MLX's native quantized matmul (different block format,
// so requires a transcoder, not a dequant) — measure first.
//
// Performance notes for future optimization:
//   - The inner two loops over 32 elements are auto-vectorizable via
//     Accelerate vDSP — drop in if the perf gate misses.
//   - A 9B Q4_K_M model has ~2400 matmuls per token decoded. Per matmul
//     dequant cost is dominated by memory bandwidth (Q4_K_M weight
//     buffer read once, fp16 written once = ~3.5x expansion).

import Foundation
import MLX
import llama  // C-side ggml-mlx symbols

extension MLXKernels {

    /// Block size in elements (must equal `QK_K` in ggml-common.h = 256).
    public static let q4KMBlockElements: Int = 256
    /// Block size in bytes (must equal `sizeof(block_q4_K)` = 144).
    public static let q4KMBlockBytes: Int = 144

    /// Dequantize Q4_K_M blocks into fp16. Output buffer must hold
    /// `blockCount * q4KMBlockElements` Float16 values.
    ///
    /// - Parameter blocks: Pointer to the first byte of the first block
    ///   (caller-owned, contiguous, blockCount * 144 bytes).
    /// - Parameter blockCount: How many 256-element blocks to dequantize.
    /// - Parameter out: Destination fp16 buffer (blockCount * 256 elements).
    public static func dequantQ4KMtoF16(
        blocks: UnsafeRawPointer,
        blockCount: Int,
        out: UnsafeMutablePointer<Float16>
    ) {
        for blockIdx in 0..<blockCount {
            let blockBase = blocks.advanced(by: blockIdx * q4KMBlockBytes)
            let outBase = out.advanced(by: blockIdx * q4KMBlockElements)
            _dequantOneQ4KMBlock(blockBase: blockBase, outBase: outBase)
        }
    }
}

// MARK: - Internal: per-block dequant

/// Dequantize a single 144-byte Q4_K_M block into 256 fp16 outputs.
@inline(__always)
private func _dequantOneQ4KMBlock(
    blockBase: UnsafeRawPointer,
    outBase: UnsafeMutablePointer<Float16>
) {
    // Layout offsets within the block (must match ggml-common.h block_q4_K).
    let dPtr      = blockBase.assumingMemoryBound(to: Float16.self)            //  0
    let dminPtr   = blockBase.advanced(by: 2).assumingMemoryBound(to: Float16.self) // 2
    let scales    = blockBase.advanced(by: 4).assumingMemoryBound(to: UInt8.self)   // 4..16
    let qs        = blockBase.advanced(by: 16).assumingMemoryBound(to: UInt8.self)  // 16..144

    let d:    Float = Float(dPtr.pointee)
    let dmin: Float = Float(dminPtr.pointee)

    // The 256 elements split into 4 groups of 64. Each group consumes one
    // pair of (sc, m) sub-blocks (so 8 sub-block pairs total). Each group
    // reads 32 bytes of qs, producing 32 low-nibble + 32 high-nibble outputs.
    var qOffset = 0
    var outOffset = 0
    for groupIdx in 0..<4 {
        let isBase = groupIdx * 2

        let (sc1, m1) = _qkScaleMin(isBase + 0, scales: scales)
        let (sc2, m2) = _qkScaleMin(isBase + 1, scales: scales)
        let d1 = d * Float(sc1)
        let mm1 = dmin * Float(m1)
        let d2 = d * Float(sc2)
        let mm2 = dmin * Float(m2)

        // Low nibble → first 32 elements
        for l in 0..<32 {
            let q = qs.advanced(by: qOffset + l).pointee
            let val = d1 * Float(q & 0x0F) - mm1
            outBase.advanced(by: outOffset + l).pointee = Float16(val)
        }
        // High nibble → next 32 elements
        for l in 0..<32 {
            let q = qs.advanced(by: qOffset + l).pointee
            let val = d2 * Float(q >> 4) - mm2
            outBase.advanced(by: outOffset + 32 + l).pointee = Float16(val)
        }

        qOffset += 32
        outOffset += 64
    }
}

/// Unpack the j-th 6-bit (scale, min) pair from the 12-byte packed scales
/// array. Mirrors ggml's `get_scale_min_k4`. j ∈ 0..<8.
@inline(__always)
private func _qkScaleMin(_ j: Int, scales q: UnsafePointer<UInt8>) -> (UInt8, UInt8) {
    if j < 4 {
        let sc = q[j] & 0x3F           // bits 0..5 of byte j
        let m  = q[j + 4] & 0x3F       // bits 0..5 of byte j+4
        return (sc, m)
    } else {
        // The "other half": each scale + min has its low 4 bits in byte
        // j+4 and high 2 bits packed into the top of byte j-4.
        let sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)
        let m  = (q[j + 4] >> 4)     | ((q[j]     >> 6) << 4)
        return (sc, m)
    }
}

// MARK: - C bridge for the kernel table

/// Implementation of the `dequant_q4km_to_f16` slot in `goleta_mlx_kernel_table`.
/// Caller passes `blocks` as a contiguous Q4_K_M buffer and `out_f16` sized
/// for `block_count * 256` Float16 values.
internal let _dequantQ4KMtoF16Bridge: @convention(c) (
    UnsafeRawPointer?,        // q4km blocks
    Int32,                    // block_count
    UnsafeMutableRawPointer?  // out_f16
) -> Bool = { blocks, blockCount, outData in
    guard let blocks, let outData, blockCount > 0 else { return false }
    let outPtr = outData.assumingMemoryBound(to: Float16.self)
    MLXKernels.dequantQ4KMtoF16(
        blocks: blocks,
        blockCount: Int(blockCount),
        out: outPtr
    )
    return true
}
