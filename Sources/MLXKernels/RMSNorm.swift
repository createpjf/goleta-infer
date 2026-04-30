// RMSNorm.swift — Phase 2 Task 2.3
//
// Wraps MLX-Swift's optimized RMSNorm kernel for the goleta-infer
// kernel-table slot. RMSNorm normalizes the last axis of the input:
//
//     y[..., j] = weight[j] * x[..., j] / sqrt(mean(x[..., :]^2) + eps)
//
// where the mean is over the feature dim (last axis). MLX's
// `MLXFast.rmsNorm` does this with a Metal kernel optimized for the
// Apple Silicon GPU when the feature dim is "large enough" (the
// crossover with CPU/Accelerate isn't documented, but the spike's
// matmul thresholding logic is the closest analog — assume MLX wins
// for n_features >= 1024).

import Foundation
import MLX
import llama  // C-side ggml-mlx symbols

extension MLXKernels {
    /// Compute RMSNorm over the last axis of `input`. `input` shape
    /// `[nElements, nFeatures]`, `weight` shape `[nFeatures]`, output
    /// same shape as input. All buffers fp16.
    public static func rmsNormF16(
        input: MLXArray, weight: MLXArray, eps: Float
    ) -> MLXArray {
        return MLXFast.rmsNorm(input, weight: weight, eps: eps)
    }
}

// MARK: - C bridge

/// Slot: `goleta_mlx_kernel_table.rms_norm`. Marshals a row-major fp16
/// `[nElements, nFeatures]` input plus a fp16 `[nFeatures]` weight into
/// MLXArrays, runs RMSNorm, copies the result back.
internal let _rmsNormF16Bridge: @convention(c) (
    UnsafeRawPointer?,        // input
    UnsafeRawPointer?,        // weight
    Int32, Int32,             // n_elements, n_features
    Float,                    // eps
    UnsafeMutableRawPointer?  // out
) -> Bool = { inputData, weightData, nElems, nFeats, eps, outData in
    guard let inputData, let weightData, let outData,
          nElems > 0, nFeats > 0 else { return false }

    let n = Int(nElems), f = Int(nFeats)

    let inputPtr  = inputData.assumingMemoryBound(to: Float16.self)
    let weightPtr = weightData.assumingMemoryBound(to: Float16.self)
    let outPtr    = outData.assumingMemoryBound(to: Float16.self)

    let inputBuf  = UnsafeBufferPointer(start: inputPtr,  count: n * f)
    let weightBuf = UnsafeBufferPointer(start: weightPtr, count: f)

    let inputArr  = MLXArray(inputBuf,  [n, f])
    let weightArr = MLXArray(weightBuf, [f])

    let result = MLXKernels.rmsNormF16(input: inputArr, weight: weightArr, eps: eps)
    eval(result)

    result.asArray(Float16.self).withUnsafeBufferPointer { src in
        guard let srcBase = src.baseAddress else { return }
        memcpy(outPtr, srcBase, n * f * MemoryLayout<Float16>.size)
    }

    return true
}
