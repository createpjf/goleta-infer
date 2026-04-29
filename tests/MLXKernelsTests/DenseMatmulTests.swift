// DenseMatmulTests.swift — Phase 2 Task 2.1a
//
// Two layers of test:
//   1. Swift wrapper equivalence: MLXKernels.denseMatmulF16(a, b) matches
//      MLX-Swift's own matmul(a, b). Tautological today (the wrapper just
//      delegates to mlx.matmul), but locks the public API in place. If
//      Phase 2.5+ specializes the wrapper (e.g. fuses with rms_norm), this
//      test must keep passing.
//   2. C bridge round-trip: goleta_mlx_dense_matmul_impl marshals raw
//      Float16 buffer pointers correctly. This is the bridge that
//      ggml-mlx.cpp graph_compute calls through the kernel table; if it's
//      wrong, every MLX-routed matmul produces garbage.

import XCTest
import MLX
import MLXRandom
@testable import MLXKernels

final class DenseMatmulTests: XCTestCase {

    // MARK: - Layer 1: Swift wrapper equivalence

    func test_4096x4096_matches_reference_within_1e3() {
        let m = 1, k = 4096, n = 4096
        let a = MLXRandom.normal([m, k], dtype: .float16)
        let b = MLXRandom.normal([k, n], dtype: .float16)

        let reference = matmul(a, b)
        let underTest = MLXKernels.denseMatmulF16(a: a, b: b)

        eval(reference, underTest)
        let diff = (reference - underTest).abs().max().item(Float.self)
        XCTAssertLessThan(diff, 1e-3,
                          "dense matmul deviates from MLX reference by \(diff)")
    }

    // MARK: - Layer 2: C bridge round-trip

    /// All-ones × all-ones matmul. Each output cell = k. Tests that the
    /// @_cdecl bridge correctly marshals raw Float16 buffers in and out.
    func test_cdecl_bridge_all_ones() {
        let m = 4, k = 8, n = 4
        let aBuf = [Float16](repeating: 1.0, count: m * k)
        let bBuf = [Float16](repeating: 1.0, count: k * n)
        var outBuf = [Float16](repeating: 0.0, count: m * n)

        let ok = aBuf.withUnsafeBufferPointer { aPtr -> Bool in
            bBuf.withUnsafeBufferPointer { bPtr in
                outBuf.withUnsafeMutableBufferPointer { outPtr in
                    _denseMatmulF16Bridge(
                        UnsafeRawPointer(aPtr.baseAddress),
                        Int32(m), Int32(k),
                        UnsafeRawPointer(bPtr.baseAddress),
                        Int32(k), Int32(n),
                        UnsafeMutableRawPointer(outPtr.baseAddress)
                    )
                }
            }
        }

        XCTAssertTrue(ok, "@_cdecl bridge returned false on valid inputs")
        for (i, v) in outBuf.enumerated() {
            XCTAssertEqual(Float(v), Float(k), accuracy: 1e-3,
                           "out[\(i)] = \(v), expected \(k)")
        }
    }

    /// NULL pointers should fail gracefully (return false) instead of crashing.
    func test_cdecl_bridge_rejects_null() {
        let ok = _denseMatmulF16Bridge(
            nil, 4, 4,
            nil, 4, 4,
            nil
        )
        XCTAssertFalse(ok)
    }

    /// Shape mismatch should fail. a's cols (k) must equal b's rows.
    func test_cdecl_bridge_rejects_shape_mismatch() {
        let aBuf = [Float16](repeating: 1.0, count: 16)  // 4×4
        let bBuf = [Float16](repeating: 1.0, count: 16)  // also 4×4 but caller lies
        var outBuf = [Float16](repeating: 0.0, count: 4)

        let ok = aBuf.withUnsafeBufferPointer { aPtr -> Bool in
            bBuf.withUnsafeBufferPointer { bPtr in
                outBuf.withUnsafeMutableBufferPointer { outPtr in
                    _denseMatmulF16Bridge(
                        UnsafeRawPointer(aPtr.baseAddress),
                        Int32(4), Int32(4),
                        UnsafeRawPointer(bPtr.baseAddress),
                        Int32(8), Int32(2),  // claims 8 rows but only 4 in buffer
                        UnsafeMutableRawPointer(outPtr.baseAddress)
                    )
                }
            }
        }
        XCTAssertFalse(ok, "should refuse a_cols(4) != b_rows(8)")
    }
}
