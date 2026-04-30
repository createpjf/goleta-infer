// KVCache.swift — Phase 2 Task 2.6
//
// Per-layer key/value cache for autoregressive decode. Each layer holds
// pre-allocated K and V buffers of shape [n_kv_heads, max_seq_len, head_dim]
// in fp16. Append writes one token's K/V at the current position;
// read returns the [n_kv_heads, length, head_dim] prefix that SDPA
// (Task 2.5) consumes directly with no transpose.
//
// Lifetime: the cache is exposed to C as a `void *` handle. The four
// kernel-table slots (`kv_cache_create/append/read/destroy`) round-trip
// through Swift's `Unmanaged` pointer machinery:
//
//     create  -> Unmanaged.passRetained(cache).toOpaque()    // +1 ref
//     append  -> takeUnretainedValue()                         // no ref change
//     read    -> takeUnretainedValue()                         // no ref change
//     destroy -> takeRetainedValue()                           // -1 ref → ARC frees
//
// The C side never knows the Swift type; the void* is a stable opaque
// token. Mismatched create/destroy is a memory leak, but every Phase 2
// test exercises both ends to catch that.
//
// What this Phase 2.6 does NOT yet do:
//   - Multi-batch (assumes batch=1, the decode-only path)
//   - Sliding window / eviction (caller responsible for capacity)
//   - Storage as MLXArray (Phase 2.7 may move heavier ops onto GPU;
//     for now plain Swift arrays + memcpy is simpler and correct)

import Foundation
import MLX
import llama  // C-side ggml-mlx symbols

// MARK: - Public Swift type

extension MLXKernels {

    /// Per-layer K/V cache for autoregressive decode. Allocated once
    /// at session start; appended to every decoded token; read by
    /// attention to feed the SDPA kernel.
    public final class KVCache {
        public let nLayers: Int
        public let nKvHeads: Int
        public let headDim: Int
        public let maxSeqLen: Int

        /// Per-layer K and V buffers laid out as
        /// `[n_kv_heads, max_seq_len, head_dim]` row-major fp16.
        /// Indexing: `keys[layer][h * maxSeqLen * headDim + p * headDim + d]`
        var keys:   [[Float16]]
        var values: [[Float16]]

        /// Number of positions written per layer (high-water mark).
        var lengths: [Int]

        public init(nLayers: Int, nKvHeads: Int, headDim: Int, maxSeqLen: Int) {
            precondition(nLayers > 0 && nKvHeads > 0 && headDim > 0 && maxSeqLen > 0)
            self.nLayers   = nLayers
            self.nKvHeads  = nKvHeads
            self.headDim   = headDim
            self.maxSeqLen = maxSeqLen
            let perLayer = nKvHeads * maxSeqLen * headDim
            self.keys    = Array(repeating: Array(repeating: 0, count: perLayer), count: nLayers)
            self.values  = Array(repeating: Array(repeating: 0, count: perLayer), count: nLayers)
            self.lengths = Array(repeating: 0, count: nLayers)
        }

        /// Write a single token's K and V at the given position in the
        /// given layer. Updates the layer's length high-water mark.
        public func append(
            layer: Int,
            position: Int,
            k: UnsafePointer<Float16>,
            v: UnsafePointer<Float16>
        ) -> Bool {
            guard layer >= 0, layer < nLayers,
                  position >= 0, position < maxSeqLen else { return false }

            let perHead = headDim
            for h in 0..<nKvHeads {
                let dstOffset = h * maxSeqLen * headDim + position * headDim
                let srcOffset = h * headDim
                keys[layer].withUnsafeMutableBufferPointer { keyBuf in
                    memcpy(
                        keyBuf.baseAddress!.advanced(by: dstOffset),
                        k.advanced(by: srcOffset),
                        perHead * MemoryLayout<Float16>.size
                    )
                }
                values[layer].withUnsafeMutableBufferPointer { valBuf in
                    memcpy(
                        valBuf.baseAddress!.advanced(by: dstOffset),
                        v.advanced(by: srcOffset),
                        perHead * MemoryLayout<Float16>.size
                    )
                }
            }

            // Track high-water mark. `position` may equal lengths[layer]
            // (sequential append) or revisit an earlier slot (regen),
            // we just keep the max.
            if position + 1 > lengths[layer] {
                lengths[layer] = position + 1
            }
            return true
        }

        /// Copy the prefix `[n_kv_heads, upToPosition, head_dim]` into the
        /// caller's output buffers in SDPA-ready layout (no transpose).
        /// `upToPosition` is the count of tokens to read (1..lengths[layer]).
        public func read(
            layer: Int,
            upToPosition L: Int,
            outK: UnsafeMutablePointer<Float16>,
            outV: UnsafeMutablePointer<Float16>
        ) -> Bool {
            guard layer >= 0, layer < nLayers,
                  L > 0, L <= lengths[layer] else { return false }

            let perHeadCount = L * headDim
            for h in 0..<nKvHeads {
                let srcOffset = h * maxSeqLen * headDim
                let dstOffset = h * L * headDim
                keys[layer].withUnsafeBufferPointer { keyBuf in
                    memcpy(
                        outK.advanced(by: dstOffset),
                        keyBuf.baseAddress!.advanced(by: srcOffset),
                        perHeadCount * MemoryLayout<Float16>.size
                    )
                }
                values[layer].withUnsafeBufferPointer { valBuf in
                    memcpy(
                        outV.advanced(by: dstOffset),
                        valBuf.baseAddress!.advanced(by: srcOffset),
                        perHeadCount * MemoryLayout<Float16>.size
                    )
                }
            }
            return true
        }

        /// Inspect length without reading data — useful for tests.
        public func currentLength(layer: Int) -> Int {
            return (layer >= 0 && layer < nLayers) ? lengths[layer] : 0
        }
    }
}

// MARK: - C bridge closures (kernel table slots)

/// Slot: `goleta_mlx_kernel_table.kv_cache_create`. Allocates a Swift
/// KVCache, wraps in Unmanaged with +1 retain, returns the opaque pointer.
/// Caller MUST eventually call `kv_cache_destroy` to release the ref.
internal let _kvCacheCreateBridge: @convention(c) (
    Int32, Int32, Int32, Int32     // n_layers, n_kv_heads, head_dim, max_seq_len
) -> UnsafeMutableRawPointer? = { nLayers, nKvHeads, headDim, maxSeqLen in
    guard nLayers > 0, nKvHeads > 0, headDim > 0, maxSeqLen > 0 else { return nil }
    let cache = MLXKernels.KVCache(
        nLayers: Int(nLayers),
        nKvHeads: Int(nKvHeads),
        headDim: Int(headDim),
        maxSeqLen: Int(maxSeqLen)
    )
    return UnsafeMutableRawPointer(Unmanaged.passRetained(cache).toOpaque())
}

/// Slot: `goleta_mlx_kernel_table.kv_cache_append`.
internal let _kvCacheAppendBridge: @convention(c) (
    UnsafeMutableRawPointer?,      // cache handle
    Int32, Int32,                  // layer, position
    UnsafeRawPointer?,             // k buffer
    UnsafeRawPointer?              // v buffer
) -> Bool = { cacheHandle, layer, position, kData, vData in
    guard let cacheHandle, let kData, let vData else { return false }
    let cache = Unmanaged<MLXKernels.KVCache>
        .fromOpaque(cacheHandle)
        .takeUnretainedValue()
    let kPtr = kData.assumingMemoryBound(to: Float16.self)
    let vPtr = vData.assumingMemoryBound(to: Float16.self)
    return cache.append(
        layer: Int(layer),
        position: Int(position),
        k: kPtr, v: vPtr
    )
}

/// Slot: `goleta_mlx_kernel_table.kv_cache_read`.
internal let _kvCacheReadBridge: @convention(c) (
    UnsafeMutableRawPointer?,      // cache handle
    Int32, Int32,                  // layer, up_to_position (count, not index)
    UnsafeMutableRawPointer?,      // out_k
    UnsafeMutableRawPointer?       // out_v
) -> Bool = { cacheHandle, layer, upTo, outKData, outVData in
    guard let cacheHandle, let outKData, let outVData else { return false }
    let cache = Unmanaged<MLXKernels.KVCache>
        .fromOpaque(cacheHandle)
        .takeUnretainedValue()
    let outK = outKData.assumingMemoryBound(to: Float16.self)
    let outV = outVData.assumingMemoryBound(to: Float16.self)
    return cache.read(
        layer: Int(layer),
        upToPosition: Int(upTo),
        outK: outK, outV: outV
    )
}

/// Slot: `goleta_mlx_kernel_table.kv_cache_destroy`. Consumes the +1
/// refcount granted by create, dropping the cache when ARC counts hit 0.
/// Passing nil is safe and a no-op (matches free(NULL) convention).
internal let _kvCacheDestroyBridge: @convention(c) (
    UnsafeMutableRawPointer?
) -> Void = { cacheHandle in
    guard let cacheHandle else { return }
    Unmanaged<MLXKernels.KVCache>
        .fromOpaque(cacheHandle)
        .release()
}
