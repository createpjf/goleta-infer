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

    // MARK: - Layer 3: ggml-shaped bridge (Task 2.1b)

    /// Verify the ggml-style bridge agrees with the generic primitive.
    /// For ggml MUL_MAT semantics: out = input @ weight.T where input is
    /// row-major [M, K] and weight is row-major [N, K]. With all-ones
    /// inputs, out[i,j] = sum_k 1*1 = K, regardless of layout.
    func test_mul_mat_ggml_bridge_all_ones() {
        let m = 4, k = 8, n = 6
        let inputBuf  = [Float16](repeating: 1.0, count: m * k)
        let weightBuf = [Float16](repeating: 1.0, count: n * k)
        var outBuf    = [Float16](repeating: 0.0, count: m * n)

        let ok = inputBuf.withUnsafeBufferPointer { iPtr -> Bool in
            weightBuf.withUnsafeBufferPointer { wPtr in
                outBuf.withUnsafeMutableBufferPointer { oPtr in
                    _mulMatF16GgmlBridge(
                        UnsafeRawPointer(iPtr.baseAddress),
                        Int32(m), Int32(k),
                        UnsafeRawPointer(wPtr.baseAddress),
                        Int32(n), Int32(k),
                        UnsafeMutableRawPointer(oPtr.baseAddress)
                    )
                }
            }
        }
        XCTAssertTrue(ok)
        for v in outBuf {
            XCTAssertEqual(Float(v), Float(k), accuracy: 1e-3)
        }
    }

    /// Numerical equivalence: ggml-shaped bridge must equal the explicit
    /// formula `out[m, n] = sum_k input[m, k] * weight[n, k]` to within
    /// fp16 rounding tolerance.
    func test_mul_mat_ggml_bridge_numerical_equivalence() {
        let m = 3, k = 5, n = 4
        // Distinct values so a transposition bug would visibly fail.
        var inputBuf  = [Float16]()
        var weightBuf = [Float16]()
        for i in 0..<(m*k) { inputBuf.append(Float16(Float(i) * 0.1)) }
        for i in 0..<(n*k) { weightBuf.append(Float16(Float(i) * 0.07 + 0.3)) }
        var outBuf = [Float16](repeating: 0.0, count: m * n)

        let ok = inputBuf.withUnsafeBufferPointer { iPtr -> Bool in
            weightBuf.withUnsafeBufferPointer { wPtr in
                outBuf.withUnsafeMutableBufferPointer { oPtr in
                    _mulMatF16GgmlBridge(
                        UnsafeRawPointer(iPtr.baseAddress),
                        Int32(m), Int32(k),
                        UnsafeRawPointer(wPtr.baseAddress),
                        Int32(n), Int32(k),
                        UnsafeMutableRawPointer(oPtr.baseAddress)
                    )
                }
            }
        }
        XCTAssertTrue(ok)

        // Reference: explicit ggml MUL_MAT formula.
        for mi in 0..<m {
            for ni in 0..<n {
                var expected: Float = 0
                for ki in 0..<k {
                    expected += Float(inputBuf[mi * k + ki]) * Float(weightBuf[ni * k + ki])
                }
                let got = Float(outBuf[mi * n + ni])
                XCTAssertEqual(got, expected, accuracy: 1e-2,
                               "out[\(mi),\(ni)] = \(got), expected \(expected)")
            }
        }
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
