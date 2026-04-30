// Phase2KernelWrapperTests.swift — Tasks 2.3 (RMSNorm) + 2.4 (RoPE) + 2.5 (SDPA)
//
// Each test verifies (a) the @convention(c) bridge marshals raw fp16
// pointers correctly, and (b) the result agrees with a hand-rolled
// reference (or with MLX-Swift's own implementation called directly,
// which is what we're wrapping). MLX itself is presumed correct;
// these tests check our marshaling is right.

import XCTest
import MLX
import MLXRandom
@testable import MLXKernels

// MARK: - Task 2.3: RMSNorm

final class RMSNormBridgeTests: XCTestCase {

    /// Manual RMSNorm reference: y[i,j] = w[j] * x[i,j] / sqrt(mean(x[i,:]^2) + eps).
    /// All-ones input + all-ones weight + n_features = 4 → mean(1^2)=1 → y = 1/sqrt(1+eps) ≈ 1.
    func test_bridge_all_ones() {
        let n = 2, f = 4
        let eps: Float = 1e-5
        let inputBuf  = [Float16](repeating: 1.0, count: n * f)
        let weightBuf = [Float16](repeating: 1.0, count: f)
        var outBuf    = [Float16](repeating: 0.0, count: n * f)

        let ok = inputBuf.withUnsafeBufferPointer { iPtr -> Bool in
            weightBuf.withUnsafeBufferPointer { wPtr in
                outBuf.withUnsafeMutableBufferPointer { oPtr in
                    _rmsNormF16Bridge(
                        UnsafeRawPointer(iPtr.baseAddress),
                        UnsafeRawPointer(wPtr.baseAddress),
                        Int32(n), Int32(f), eps,
                        UnsafeMutableRawPointer(oPtr.baseAddress)
                    )
                }
            }
        }
        XCTAssertTrue(ok)
        let expected = Float(1.0) / sqrt(Float(1.0) + eps)
        for v in outBuf {
            XCTAssertEqual(Float(v), expected, accuracy: 1e-3)
        }
    }

    /// Numerical equivalence with manual reference. Random input, known weight,
    /// compare bridge output vs Swift loop computing the same formula.
    func test_bridge_matches_manual_reference() {
        let n = 3, f = 8
        let eps: Float = 1e-6

        var inputF: [Float] = []
        for i in 0..<(n * f) { inputF.append(sin(Float(i) * 0.21) * 1.5 + 0.3) }
        let inputBuf  = inputF.map { Float16($0) }
        let weightBuf = (0..<f).map { Float16(0.5 + Float($0) * 0.1) }
        var outBuf    = [Float16](repeating: 0, count: n * f)

        _ = inputBuf.withUnsafeBufferPointer { iPtr in
            weightBuf.withUnsafeBufferPointer { wPtr in
                outBuf.withUnsafeMutableBufferPointer { oPtr in
                    _rmsNormF16Bridge(
                        UnsafeRawPointer(iPtr.baseAddress),
                        UnsafeRawPointer(wPtr.baseAddress),
                        Int32(n), Int32(f), eps,
                        UnsafeMutableRawPointer(oPtr.baseAddress)
                    )
                }
            }
        }

        // Manual reference (fp32 throughout for clarity)
        for row in 0..<n {
            var sumSq: Float = 0
            for j in 0..<f {
                let v = Float(inputBuf[row * f + j])
                sumSq += v * v
            }
            let invRms = 1.0 / sqrt(sumSq / Float(f) + eps)
            for j in 0..<f {
                let expected = Float(weightBuf[j]) * Float(inputBuf[row * f + j]) * invRms
                let got = Float(outBuf[row * f + j])
                // fp16 rounding tolerance is roughly 1e-3 relative for values ~1.
                XCTAssertEqual(got, expected, accuracy: 5e-3,
                               "row=\(row) j=\(j) got=\(got) expected=\(expected)")
            }
        }
    }

    func test_bridge_rejects_null() {
        XCTAssertFalse(_rmsNormF16Bridge(nil, nil, 4, 4, 1e-5, nil))
    }
}

// MARK: - Task 2.4: RoPE

final class RoPEBridgeTests: XCTestCase {

    /// Identity at sequence position 0 with single token: the only
    /// position in the sequence is i=0, and at offset=0 the absolute
    /// position is also 0, so all rotation angles are 0 → identity.
    /// (For multi-token sequences, only token 0 is identity; tokens
    /// 1, 2, ... rotate by non-zero angles regardless of offset.)
    func test_bridge_first_token_is_identity() {
        let nTokens = 1, headDim = 8
        var inputBuf = [Float16]()
        for i in 0..<headDim { inputBuf.append(Float16(Float(i) * 0.01 + 0.05)) }
        var outBuf = [Float16](repeating: 0, count: headDim)

        let ok = inputBuf.withUnsafeBufferPointer { iPtr -> Bool in
            outBuf.withUnsafeMutableBufferPointer { oPtr in
                _ropeF16Bridge(
                    UnsafeRawPointer(iPtr.baseAddress),
                    Int32(nTokens), Int32(headDim),
                    Int32(headDim), Int32(0),         // offset=0 + nTokens=1 → angle=0
                    10000.0,
                    UnsafeMutableRawPointer(oPtr.baseAddress)
                )
            }
        }
        XCTAssertTrue(ok)
        for i in 0..<inputBuf.count {
            XCTAssertEqual(Float(outBuf[i]), Float(inputBuf[i]), accuracy: 1e-3,
                           "first-token RoPE should be identity at i=\(i)")
        }
    }

    /// Real rotation: nDims = headDim, offset varies. We don't reimplement
    /// RoPE math — just verify the bridge produces the same result as
    /// calling MLXFast.RoPE directly with identical args.
    func test_bridge_matches_direct_mlx_call() {
        let nTokens = 3, headDim = 8, nDims = 8, offset = 5
        let theta: Float = 10000.0

        var inputBuf = [Float16]()
        for i in 0..<(nTokens * headDim) {
            inputBuf.append(Float16(sin(Float(i) * 0.13) + 0.5))
        }
        var bridgeOut = [Float16](repeating: 0, count: nTokens * headDim)

        _ = inputBuf.withUnsafeBufferPointer { iPtr in
            bridgeOut.withUnsafeMutableBufferPointer { oPtr in
                _ropeF16Bridge(
                    UnsafeRawPointer(iPtr.baseAddress),
                    Int32(nTokens), Int32(nDims),
                    Int32(headDim), Int32(offset),
                    theta,
                    UnsafeMutableRawPointer(oPtr.baseAddress)
                )
            }
        }

        // Direct MLX call — must use 3D input since MLX RoPE needs >= 3 dims.
        // Shape it the same way the bridge does for an apples-to-apples test.
        let directInput = MLXArray(inputBuf, [1, nTokens, headDim])
        let directRotated = MLXFast.RoPE(
            directInput,
            dimensions: nDims, traditional: false,
            base: theta, scale: 1.0, offset: offset
        )
        let directOut = directRotated.squeezed(axis: 0)
        eval(directOut)
        let directResultBuf: [Float16] = directOut.asArray(Float16.self)

        for i in 0..<bridgeOut.count {
            XCTAssertEqual(Float(bridgeOut[i]), Float(directResultBuf[i]), accuracy: 5e-3,
                           "bridge / MLX disagree at i=\(i)")
        }
    }

    func test_bridge_rejects_null() {
        XCTAssertFalse(_ropeF16Bridge(nil, 4, 8, 8, 0, 10000.0, nil))
    }
}

// MARK: - Task 2.5: SDPA

final class SDPABridgeTests: XCTestCase {

    /// Smoke test: identical q/k/v values → output should equal v
    /// (after softmax over k.T @ q rows that all match equally weighted).
    /// Specifically when all q/k/v are the same constant, attention
    /// reduces to averaging which equals the constant.
    func test_bridge_uniform_values() {
        let nHeads = 2, nKvHeads = 2, headDim = 4, seqLen = 3
        let count = nHeads * seqLen * headDim
        let qBuf = [Float16](repeating: 0.5, count: count)
        let kBuf = [Float16](repeating: 0.5, count: count)
        let vBuf = [Float16](repeating: 0.5, count: count)
        var outBuf = [Float16](repeating: 0.0, count: count)

        let ok = qBuf.withUnsafeBufferPointer { qP -> Bool in
            kBuf.withUnsafeBufferPointer { kP in
                vBuf.withUnsafeBufferPointer { vP in
                    outBuf.withUnsafeMutableBufferPointer { oP in
                        _sdpaF16Bridge(
                            UnsafeRawPointer(qP.baseAddress),
                            UnsafeRawPointer(kP.baseAddress),
                            UnsafeRawPointer(vP.baseAddress),
                            Int32(nHeads), Int32(nKvHeads),
                            Int32(headDim), Int32(seqLen),
                            UnsafeMutableRawPointer(oP.baseAddress)
                        )
                    }
                }
            }
        }
        XCTAssertTrue(ok)
        for v in outBuf {
            XCTAssertEqual(Float(v), 0.5, accuracy: 1e-3)
        }
    }

    /// Bridge agrees with calling MLXFast.scaledDotProductAttention directly
    /// on the same inputs. Verifies the marshaling + reshape is correct.
    func test_bridge_matches_direct_mlx_call() {
        let nHeads = 4, nKvHeads = 2, headDim = 8, seqLen = 3

        // GQA: nHeads (4) > nKvHeads (2), ratio 2:1.
        let qCount  = nHeads    * seqLen * headDim
        let kvCount = nKvHeads  * seqLen * headDim
        let qBuf  = (0..<qCount ).map  { Float16(sin(Float($0) * 0.07) + 0.3) }
        let kBuf  = (0..<kvCount).map { Float16(cos(Float($0) * 0.11) + 0.2) }
        let vBuf  = (0..<kvCount).map { Float16(sin(Float($0) * 0.05) - 0.1) }
        var bridgeOut = [Float16](repeating: 0, count: qCount)

        _ = qBuf.withUnsafeBufferPointer { qP in
            kBuf.withUnsafeBufferPointer { kP in
                vBuf.withUnsafeBufferPointer { vP in
                    bridgeOut.withUnsafeMutableBufferPointer { oP in
                        _sdpaF16Bridge(
                            UnsafeRawPointer(qP.baseAddress),
                            UnsafeRawPointer(kP.baseAddress),
                            UnsafeRawPointer(vP.baseAddress),
                            Int32(nHeads), Int32(nKvHeads),
                            Int32(headDim), Int32(seqLen),
                            UnsafeMutableRawPointer(oP.baseAddress)
                        )
                    }
                }
            }
        }

        // Direct call with the same logical layout
        let qArr = MLXArray(qBuf,  [nHeads,   seqLen, headDim]).expandedDimensions(axis: 0)
        let kArr = MLXArray(kBuf,  [nKvHeads, seqLen, headDim]).expandedDimensions(axis: 0)
        let vArr = MLXArray(vBuf,  [nKvHeads, seqLen, headDim]).expandedDimensions(axis: 0)
        let scale = 1.0 / sqrt(Float(headDim))
        let directOut = MLXFast.scaledDotProductAttention(
            queries: qArr, keys: kArr, values: vArr,
            scale: scale, mask: nil
        ).squeezed(axis: 0)
        eval(directOut)
        let directBuf: [Float16] = directOut.asArray(Float16.self)

        for i in 0..<bridgeOut.count {
            XCTAssertEqual(Float(bridgeOut[i]), Float(directBuf[i]), accuracy: 5e-3)
        }
    }

    func test_bridge_rejects_null() {
        XCTAssertFalse(_sdpaF16Bridge(nil, nil, nil, 4, 4, 8, 8, nil))
    }

    func test_bridge_rejects_non_divisible_heads() {
        // n_heads must be a multiple of n_kv_heads for GQA/MQA.
        var dummyQ = [Float16](repeating: 0.0, count: 100)
        var dummyK = [Float16](repeating: 0.0, count: 100)
        var dummyV = [Float16](repeating: 0.0, count: 100)
        var dummyOut = [Float16](repeating: 0.0, count: 100)

        let ok = dummyQ.withUnsafeBufferPointer { qP -> Bool in
            dummyK.withUnsafeBufferPointer { kP in
                dummyV.withUnsafeBufferPointer { vP in
                    dummyOut.withUnsafeMutableBufferPointer { oP in
                        _sdpaF16Bridge(
                            UnsafeRawPointer(qP.baseAddress),
                            UnsafeRawPointer(kP.baseAddress),
                            UnsafeRawPointer(vP.baseAddress),
                            5, 2,  // 5 heads, 2 kv-heads — not divisible
                            8, 1,
                            UnsafeMutableRawPointer(oP.baseAddress)
                        )
                    }
                }
            }
        }
        XCTAssertFalse(ok)
    }
}
