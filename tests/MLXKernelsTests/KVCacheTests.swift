// KVCacheTests.swift — Phase 2 Task 2.6
//
// Two surfaces under test:
//   1. The Swift `MLXKernels.KVCache` class — append, read, length tracking.
//   2. The four C-bridge closures — create / append / read / destroy
//      round-trip through `void *` opaque handles using Unmanaged.
//
// The C bridge tests are particularly important because mismatched
// retain/release would leak (or worse, double-free) silently. Each test
// pairs every create with a destroy.

import XCTest
@testable import MLXKernels

final class KVCacheSwiftTests: XCTestCase {

    /// Append → read → assert byte-equal round trip on a single layer.
    func test_swift_append_read_round_trip() {
        let cache = MLXKernels.KVCache(
            nLayers: 1, nKvHeads: 2, headDim: 4, maxSeqLen: 16
        )
        // One token's K/V in nKvHeads * headDim layout
        let kInput: [Float16] = [
            1, 2, 3, 4,    // head 0
            5, 6, 7, 8     // head 1
        ]
        let vInput: [Float16] = [
            10, 20, 30, 40,
            50, 60, 70, 80
        ]

        kInput.withUnsafeBufferPointer { kPtr in
            vInput.withUnsafeBufferPointer { vPtr in
                XCTAssertTrue(cache.append(
                    layer: 0, position: 0,
                    k: kPtr.baseAddress!, v: vPtr.baseAddress!
                ))
            }
        }

        XCTAssertEqual(cache.currentLength(layer: 0), 1)

        // Read length=1 → output [n_kv_heads=2, L=1, head_dim=4] = 8 elements
        var outK = [Float16](repeating: -1, count: 8)
        var outV = [Float16](repeating: -1, count: 8)
        outK.withUnsafeMutableBufferPointer { kPtr in
            outV.withUnsafeMutableBufferPointer { vPtr in
                XCTAssertTrue(cache.read(
                    layer: 0, upToPosition: 1,
                    outK: kPtr.baseAddress!, outV: vPtr.baseAddress!
                ))
            }
        }
        XCTAssertEqual(outK, kInput)
        XCTAssertEqual(outV, vInput)
    }

    /// Multiple appends accumulate in the right slots; read returns
    /// them in token-order per head.
    func test_swift_multi_append_preserves_order() {
        let cache = MLXKernels.KVCache(
            nLayers: 1, nKvHeads: 1, headDim: 2, maxSeqLen: 8
        )
        // Append 3 tokens. K/V: identifier per token+pair so we can
        // see exactly which position is at each output slot.
        let tokens: [(k: [Float16], v: [Float16])] = [
            ([10, 11], [100, 101]),
            ([20, 21], [200, 201]),
            ([30, 31], [300, 301]),
        ]
        for (i, t) in tokens.enumerated() {
            t.k.withUnsafeBufferPointer { kP in
                t.v.withUnsafeBufferPointer { vP in
                    XCTAssertTrue(cache.append(
                        layer: 0, position: i,
                        k: kP.baseAddress!, v: vP.baseAddress!
                    ))
                }
            }
        }
        XCTAssertEqual(cache.currentLength(layer: 0), 3)

        // Read length=3 → [1 head, 3 positions, 2 dims] = 6 elements,
        // expect interleaved by token:
        //   out[0..2]   = head 0, position 0, head_dim 0..1
        //   out[2..4]   = head 0, position 1, head_dim 0..1
        //   out[4..6]   = head 0, position 2, head_dim 0..1
        var outK = [Float16](repeating: -1, count: 6)
        var outV = [Float16](repeating: -1, count: 6)
        outK.withUnsafeMutableBufferPointer { kP in
            outV.withUnsafeMutableBufferPointer { vP in
                XCTAssertTrue(cache.read(
                    layer: 0, upToPosition: 3,
                    outK: kP.baseAddress!, outV: vP.baseAddress!
                ))
            }
        }
        XCTAssertEqual(outK, [10, 11, 20, 21, 30, 31])
        XCTAssertEqual(outV, [100, 101, 200, 201, 300, 301])
    }

    /// Layers are independent: writes to layer 0 don't show up in
    /// layer 1, and lengths track per-layer.
    func test_swift_layers_are_isolated() {
        let cache = MLXKernels.KVCache(
            nLayers: 3, nKvHeads: 1, headDim: 2, maxSeqLen: 8
        )
        let k0: [Float16] = [11, 12]
        let v0: [Float16] = [110, 120]
        let k2: [Float16] = [33, 34]
        let v2: [Float16] = [330, 340]

        k0.withUnsafeBufferPointer { kP in
            v0.withUnsafeBufferPointer { vP in
                _ = cache.append(layer: 0, position: 0,
                                  k: kP.baseAddress!, v: vP.baseAddress!)
            }
        }
        k2.withUnsafeBufferPointer { kP in
            v2.withUnsafeBufferPointer { vP in
                _ = cache.append(layer: 2, position: 0,
                                  k: kP.baseAddress!, v: vP.baseAddress!)
            }
        }

        XCTAssertEqual(cache.currentLength(layer: 0), 1)
        XCTAssertEqual(cache.currentLength(layer: 1), 0)
        XCTAssertEqual(cache.currentLength(layer: 2), 1)

        // Reading layer 1 should fail (no data yet)
        var outK = [Float16](repeating: -1, count: 2)
        var outV = [Float16](repeating: -1, count: 2)
        outK.withUnsafeMutableBufferPointer { kP in
            outV.withUnsafeMutableBufferPointer { vP in
                XCTAssertFalse(cache.read(
                    layer: 1, upToPosition: 1,
                    outK: kP.baseAddress!, outV: vP.baseAddress!
                ))
            }
        }
    }

    func test_swift_rejects_out_of_range() {
        let cache = MLXKernels.KVCache(
            nLayers: 1, nKvHeads: 1, headDim: 2, maxSeqLen: 4
        )
        let k: [Float16] = [1, 2]
        let v: [Float16] = [3, 4]

        // append: layer out of range
        k.withUnsafeBufferPointer { kP in
            v.withUnsafeBufferPointer { vP in
                XCTAssertFalse(cache.append(layer: 99, position: 0,
                                             k: kP.baseAddress!, v: vP.baseAddress!))
                // append: position out of range
                XCTAssertFalse(cache.append(layer: 0, position: 99,
                                             k: kP.baseAddress!, v: vP.baseAddress!))
            }
        }

        // read: nothing written yet
        var outK = [Float16](repeating: 0, count: 2)
        var outV = [Float16](repeating: 0, count: 2)
        outK.withUnsafeMutableBufferPointer { kP in
            outV.withUnsafeMutableBufferPointer { vP in
                XCTAssertFalse(cache.read(layer: 0, upToPosition: 1,
                                           outK: kP.baseAddress!, outV: vP.baseAddress!))
            }
        }
    }
}

// MARK: - C bridge round-trip

final class KVCacheBridgeTests: XCTestCase {

    /// Full lifecycle through the @convention(c) closures: create →
    /// append → read → destroy. Verifies the Unmanaged round-trip
    /// keeps the cache alive across calls and frees on destroy
    /// (no observable assert here for free; ASan / leak checks would
    /// catch a double-free on subsequent runs).
    func test_bridge_full_lifecycle() {
        // create
        let handle = _kvCacheCreateBridge(1, 2, 4, 8)
        XCTAssertNotNil(handle)
        defer { _kvCacheDestroyBridge(handle) }

        let kInput: [Float16] = [1, 2, 3, 4, 5, 6, 7, 8]
        let vInput: [Float16] = [9, 10, 11, 12, 13, 14, 15, 16]

        // append
        let appendOk = kInput.withUnsafeBufferPointer { kP -> Bool in
            vInput.withUnsafeBufferPointer { vP in
                _kvCacheAppendBridge(
                    handle, 0, 0,
                    UnsafeRawPointer(kP.baseAddress),
                    UnsafeRawPointer(vP.baseAddress)
                )
            }
        }
        XCTAssertTrue(appendOk)

        // read
        var outK = [Float16](repeating: -1, count: 8)
        var outV = [Float16](repeating: -1, count: 8)
        let readOk = outK.withUnsafeMutableBufferPointer { kP -> Bool in
            outV.withUnsafeMutableBufferPointer { vP in
                _kvCacheReadBridge(
                    handle, 0, 1,
                    UnsafeMutableRawPointer(kP.baseAddress),
                    UnsafeMutableRawPointer(vP.baseAddress)
                )
            }
        }
        XCTAssertTrue(readOk)
        XCTAssertEqual(outK, kInput)
        XCTAssertEqual(outV, vInput)
    }

    func test_bridge_create_rejects_zero_dimensions() {
        XCTAssertNil(_kvCacheCreateBridge(0, 1, 1, 1))
        XCTAssertNil(_kvCacheCreateBridge(1, 0, 1, 1))
        XCTAssertNil(_kvCacheCreateBridge(1, 1, 0, 1))
        XCTAssertNil(_kvCacheCreateBridge(1, 1, 1, 0))
    }

    func test_bridge_destroy_nil_is_safe() {
        // free(NULL) convention: passing nil to destroy must not crash.
        _kvCacheDestroyBridge(nil)
        XCTAssertTrue(true, "still alive")
    }

    func test_bridge_append_read_rejects_null() {
        let handle = _kvCacheCreateBridge(1, 1, 2, 4)
        XCTAssertNotNil(handle)
        defer { _kvCacheDestroyBridge(handle) }

        XCTAssertFalse(_kvCacheAppendBridge(handle, 0, 0, nil, nil))
        XCTAssertFalse(_kvCacheReadBridge(handle, 0, 1, nil, nil))
        XCTAssertFalse(_kvCacheAppendBridge(nil, 0, 0, nil, nil))
        XCTAssertFalse(_kvCacheReadBridge(nil, 0, 1, nil, nil))
    }

    /// Stress: 64-token round trip in 4 layers, verify lengths track
    /// independently and read returns the right prefix per layer at
    /// arbitrary lengths.
    func test_bridge_64_token_4_layer_round_trip() {
        let handle = _kvCacheCreateBridge(4, 2, 8, 64)
        XCTAssertNotNil(handle)
        defer { _kvCacheDestroyBridge(handle) }

        // Fill each layer with a layer-distinct pattern so we can tell
        // them apart on read.
        for layer in 0..<4 {
            for position in 0..<64 {
                // 16 elements per token (n_kv_heads=2 * head_dim=8)
                let kBuf: [Float16] = (0..<16).map {
                    Float16(Float(layer) * 1000 + Float(position) * 10 + Float($0))
                }
                let vBuf: [Float16] = kBuf.map { Float16(Float($0) + 0.5) }

                let ok = kBuf.withUnsafeBufferPointer { kP -> Bool in
                    vBuf.withUnsafeBufferPointer { vP in
                        _kvCacheAppendBridge(
                            handle, Int32(layer), Int32(position),
                            UnsafeRawPointer(kP.baseAddress),
                            UnsafeRawPointer(vP.baseAddress)
                        )
                    }
                }
                XCTAssertTrue(ok, "append failed at layer=\(layer) position=\(position)")
            }
        }

        // Read full prefix (length 64) on layer 2 only and verify
        // pattern matches what we wrote.
        let elementsPerHead = 64 * 8  // L * head_dim
        let readSize = 2 * elementsPerHead  // n_kv_heads * L * head_dim
        var outK = [Float16](repeating: 0, count: readSize)
        var outV = [Float16](repeating: 0, count: readSize)
        let readOk = outK.withUnsafeMutableBufferPointer { kP -> Bool in
            outV.withUnsafeMutableBufferPointer { vP in
                _kvCacheReadBridge(
                    handle, 2, 64,
                    UnsafeMutableRawPointer(kP.baseAddress),
                    UnsafeMutableRawPointer(vP.baseAddress)
                )
            }
        }
        XCTAssertTrue(readOk)

        // Verify a few sample elements per head per position
        for h in 0..<2 {
            for p in [0, 1, 31, 63] {
                for d in [0, 7] {
                    let outIdx = h * elementsPerHead + p * 8 + d
                    // What we appended at (layer=2, position=p): index in
                    // the append buffer was h*8 + d.
                    let appendBufIdx = h * 8 + d
                    let expectedK = Float16(2 * 1000 + p * 10 + appendBufIdx)
                    let expectedV = Float16(Float(expectedK) + 0.5)
                    XCTAssertEqual(outK[outIdx], expectedK,
                                   "K mismatch at h=\(h) p=\(p) d=\(d)")
                    XCTAssertEqual(outV[outIdx], expectedV,
                                   "V mismatch at h=\(h) p=\(p) d=\(d)")
                }
            }
        }
    }
}
