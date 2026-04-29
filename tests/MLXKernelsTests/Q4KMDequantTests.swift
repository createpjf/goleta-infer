// Q4KMDequantTests.swift — Phase 2 Task 2.2
//
// The load-bearing test for the dequant path: the Swift implementation
// must produce values that match ggml's own dequant_row_q4_K within fp16
// rounding tolerance. If the layout interpretation, scale unpacking, or
// nibble ordering is off in any way, this test fails immediately.
//
// We synthesize Q4_K_M blocks by calling ggml's quantize_row_q4_K_ref via
// FFI on a known float pattern, then dequant via both ggml and our Swift
// impl, then compare element-by-element.

import XCTest
import llama
@testable import MLXKernels

// ggml-quants.h is an internal header (not exposed via the public framework
// modulemap), but the symbols are present in the binary. Declare them here
// for the test only — these are not part of the goleta-infer public API.
@_silgen_name("quantize_row_q4_K_ref")
private func quantize_row_q4_K_ref(
    _ src: UnsafePointer<Float>?,
    _ dst: OpaquePointer?,
    _ k: Int64
)

@_silgen_name("dequantize_row_q4_K")
private func dequantize_row_q4_K(
    _ src: OpaquePointer?,
    _ dst: UnsafeMutablePointer<Float>?,
    _ k: Int64
)

final class Q4KMDequantTests: XCTestCase {

    /// Round-trip equivalence: fill a 256-element block with a sine wave,
    /// quantize via ggml, dequant via Swift + ggml independently, the two
    /// dequant outputs must agree exactly (both are the same algorithm).
    func test_dequant_matches_ggml_reference_one_block() {
        let n = MLXKernels.q4KMBlockElements  // 256
        let blockCount = 1

        // Synthesize a known float pattern with non-trivial dynamic range
        // (range matters for Q4_K_M's per-sub-block scale + min encoding).
        var src = [Float](repeating: 0, count: n)
        for i in 0..<n {
            // Mix of magnitudes so each 32-element sub-block sees a
            // different range and the (scale, min) packing is exercised.
            src[i] = Float(i) * 0.013 - 1.5 + sin(Float(i) * 0.07) * 0.4
        }

        // Allocate a 144-byte Q4_K_M block. ggml's quantize_row_q4_K_ref
        // takes block_q4_K* but it's just sizeof-compatible bytes, so we
        // pass a UInt8 buffer cast through OpaquePointer.
        var qBlock = [UInt8](repeating: 0, count: MLXKernels.q4KMBlockBytes * blockCount)

        src.withUnsafeBufferPointer { srcPtr in
            qBlock.withUnsafeMutableBufferPointer { qPtr in
                quantize_row_q4_K_ref(
                    srcPtr.baseAddress,
                    OpaquePointer(qPtr.baseAddress),
                    Int64(n)
                )
            }
        }

        // ggml's reference dequant (fp32 output)
        var ggmlOut = [Float](repeating: 0, count: n)
        qBlock.withUnsafeBufferPointer { qPtr in
            ggmlOut.withUnsafeMutableBufferPointer { outPtr in
                dequantize_row_q4_K(
                    OpaquePointer(qPtr.baseAddress),
                    outPtr.baseAddress,
                    Int64(n)
                )
            }
        }

        // Swift's dequant (fp16 output)
        var swiftOut = [Float16](repeating: 0, count: n)
        qBlock.withUnsafeBufferPointer { qPtr in
            swiftOut.withUnsafeMutableBufferPointer { outPtr in
                MLXKernels.dequantQ4KMtoF16(
                    blocks: UnsafeRawPointer(qPtr.baseAddress!),
                    blockCount: blockCount,
                    out: outPtr.baseAddress!
                )
            }
        }

        // Compare. Tolerance is fp16 round-trip — Swift converts to fp16
        // (~3 decimal digits of precision) while ggml stays in fp32.
        // Q4_K_M's quant error is already much larger than fp16 epsilon,
        // so we just need to verify Swift didn't introduce ADDITIONAL
        // error beyond the fp32 → fp16 cast.
        var maxAbsDiff: Float = 0
        var maxRelDiff: Float = 0
        for i in 0..<n {
            let g = ggmlOut[i]
            let s = Float(swiftOut[i])
            let absDiff = abs(g - s)
            maxAbsDiff = max(maxAbsDiff, absDiff)
            // Relative error normalized by ggml's value magnitude
            let denom = max(abs(g), 1e-3)
            maxRelDiff = max(maxRelDiff, absDiff / denom)
        }

        // fp16 has ~3 decimal digits of precision. For values in the
        // magnitude range we tested (~1.5), abs diff under 1e-3 is fp16's
        // expected resolution.
        XCTAssertLessThan(maxAbsDiff, 5e-3,
                          "Swift dequant max abs diff \(maxAbsDiff) exceeds fp16 cast tolerance")
        XCTAssertLessThan(maxRelDiff, 5e-3,
                          "Swift dequant max rel diff \(maxRelDiff) exceeds fp16 cast tolerance")
    }

    /// Multi-block dequant: same algorithm, more blocks. Verifies the
    /// per-block stride is correct (no buffer overrun, no offset reset bug).
    func test_dequant_matches_ggml_reference_multi_block() {
        let blockCount = 4
        let n = MLXKernels.q4KMBlockElements * blockCount  // 1024

        var src = [Float](repeating: 0, count: n)
        for i in 0..<n {
            src[i] = sin(Float(i) * 0.03) * 2.0 + Float(i % 17) * 0.05
        }

        var qBlocks = [UInt8](repeating: 0, count: MLXKernels.q4KMBlockBytes * blockCount)
        src.withUnsafeBufferPointer { srcPtr in
            qBlocks.withUnsafeMutableBufferPointer { qPtr in
                quantize_row_q4_K_ref(
                    srcPtr.baseAddress,
                    OpaquePointer(qPtr.baseAddress),
                    Int64(n)
                )
            }
        }

        var ggmlOut = [Float](repeating: 0, count: n)
        qBlocks.withUnsafeBufferPointer { qPtr in
            ggmlOut.withUnsafeMutableBufferPointer { outPtr in
                dequantize_row_q4_K(
                    OpaquePointer(qPtr.baseAddress),
                    outPtr.baseAddress,
                    Int64(n)
                )
            }
        }

        var swiftOut = [Float16](repeating: 0, count: n)
        qBlocks.withUnsafeBufferPointer { qPtr in
            swiftOut.withUnsafeMutableBufferPointer { outPtr in
                MLXKernels.dequantQ4KMtoF16(
                    blocks: UnsafeRawPointer(qPtr.baseAddress!),
                    blockCount: blockCount,
                    out: outPtr.baseAddress!
                )
            }
        }

        for i in 0..<n {
            XCTAssertEqual(Float(swiftOut[i]), ggmlOut[i], accuracy: 5e-3,
                           "block-stride bug? mismatch at element \(i): swift=\(swiftOut[i]) ggml=\(ggmlOut[i])")
        }
    }

    /// C bridge round-trip — same as above but called via the
    /// @convention(c) closure that the kernel table actually uses.
    func test_cdecl_bridge_matches_ggml_reference() {
        let blockCount = 2
        let n = MLXKernels.q4KMBlockElements * blockCount

        var src = [Float](repeating: 0, count: n)
        for i in 0..<n { src[i] = Float(i) * 0.005 - 0.5 }

        var qBlocks = [UInt8](repeating: 0, count: MLXKernels.q4KMBlockBytes * blockCount)
        src.withUnsafeBufferPointer { srcPtr in
            qBlocks.withUnsafeMutableBufferPointer { qPtr in
                quantize_row_q4_K_ref(
                    srcPtr.baseAddress,
                    OpaquePointer(qPtr.baseAddress),
                    Int64(n)
                )
            }
        }

        var ggmlOut = [Float](repeating: 0, count: n)
        qBlocks.withUnsafeBufferPointer { qPtr in
            ggmlOut.withUnsafeMutableBufferPointer { outPtr in
                dequantize_row_q4_K(
                    OpaquePointer(qPtr.baseAddress),
                    outPtr.baseAddress,
                    Int64(n)
                )
            }
        }

        var bridgeOut = [Float16](repeating: 0, count: n)
        let ok = qBlocks.withUnsafeBufferPointer { qPtr -> Bool in
            bridgeOut.withUnsafeMutableBufferPointer { outPtr in
                _dequantQ4KMtoF16Bridge(
                    UnsafeRawPointer(qPtr.baseAddress),
                    Int32(blockCount),
                    UnsafeMutableRawPointer(outPtr.baseAddress)
                )
            }
        }
        XCTAssertTrue(ok)
        for i in 0..<n {
            XCTAssertEqual(Float(bridgeOut[i]), ggmlOut[i], accuracy: 5e-3)
        }
    }

    /// NULL inputs should fail gracefully.
    func test_cdecl_bridge_rejects_null() {
        XCTAssertFalse(_dequantQ4KMtoF16Bridge(nil, 1, nil))
    }

    func test_cdecl_bridge_rejects_zero_block_count() {
        var dummy = [Float16](repeating: 0, count: 1)
        var blocks = [UInt8](repeating: 0, count: 144)
        let ok = blocks.withUnsafeBufferPointer { qPtr -> Bool in
            dummy.withUnsafeMutableBufferPointer { outPtr in
                _dequantQ4KMtoF16Bridge(
                    UnsafeRawPointer(qPtr.baseAddress),
                    0,
                    UnsafeMutableRawPointer(outPtr.baseAddress)
                )
            }
        }
        XCTAssertFalse(ok)
    }
}
