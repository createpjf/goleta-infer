// goleta-infer Phase 0 spike — MLX matmul microbenchmark
//
// Question this answers:
//   Does Apple MLX hit ≥1.5× Accelerate-on-CPU on the matrix shapes
//   PowerInfer's hot path actually uses?
//
// The shapes we care about (Bamboo-7B-DPO at f16):
//   FFN gate/up/down:  4096 × 11008
//   Attention QKV:     4096 × 4096
//   Sparse hot path:   4096 × ~1100  (≈ 10% of FFN width)
//
// The benchmark times pure matmul throughput (no model loading,
// no quantization). It's a microbench — it answers "is the kernel
// fast enough" before we invest in the full integration.
//
// Pass criterion (umbrella plan §13.3 Phase 0 Gate):
//   MLX p50 latency ≤ 0.66× Accelerate p50 latency
//   (i.e. ≥1.5× speedup)
//
// Fail criterion → revert §13, fall back to "wait for FLock Pocket".

import Foundation
import MLX
import MLXRandom
import Accelerate

// MARK: - Shapes mirroring Bamboo-7B-DPO hot path

struct Shape: CustomStringConvertible {
    let name: String
    let m: Int    // batch (1 = single token decode, the inference hot path)
    let k: Int    // input dim
    let n: Int    // output dim
    var description: String { "\(name) [\(m)x\(k)] @ [\(k)x\(n)]" }
}

let shapes: [Shape] = [
    .init(name: "attn_qkv_decode",  m: 1, k: 4096, n: 4096),
    .init(name: "ffn_gate_decode",  m: 1, k: 4096, n: 11008),
    .init(name: "ffn_down_decode",  m: 1, k: 11008, n: 4096),
    .init(name: "sparse_10pct",     m: 1, k: 4096, n: 1100),    // 10% hot
    .init(name: "sparse_30pct",     m: 1, k: 4096, n: 3300),    // 30% hot
    // Prefill-style batch=128 row to see how MLX scales w/ batch.
    .init(name: "attn_qkv_prefill", m: 128, k: 4096, n: 4096),
]

let warmupIters = 5
let timedIters = 50

// MARK: - Accelerate baseline (cblas_sgemm on CPU SIMD)

func benchmarkAccelerate(shape: Shape) -> Double {
    let aSize = shape.m * shape.k
    let bSize = shape.k * shape.n
    let cSize = shape.m * shape.n

    var a = [Float](repeating: 0, count: aSize)
    var b = [Float](repeating: 0, count: bSize)
    var c = [Float](repeating: 0, count: cSize)

    // Fill with deterministic noise; LCG keeps it cheap & repeatable.
    var seed: UInt32 = 0xC0FFEE
    func rng() -> Float {
        seed = seed &* 1664525 &+ 1013904223
        return Float(seed) / Float(UInt32.max) - 0.5
    }
    for i in 0..<aSize { a[i] = rng() }
    for i in 0..<bSize { b[i] = rng() }

    // Warm up (caches, branch predictors).
    for _ in 0..<warmupIters {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(shape.m), Int32(shape.n), Int32(shape.k),
            1.0, a, Int32(shape.k),
            b, Int32(shape.n),
            0.0, &c, Int32(shape.n)
        )
    }

    var elapsed: [Double] = []
    elapsed.reserveCapacity(timedIters)
    for _ in 0..<timedIters {
        let t0 = DispatchTime.now()
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(shape.m), Int32(shape.n), Int32(shape.k),
            1.0, a, Int32(shape.k),
            b, Int32(shape.n),
            0.0, &c, Int32(shape.n)
        )
        let t1 = DispatchTime.now()
        elapsed.append(Double(t1.uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000) // ms
    }
    return p50(elapsed)
}

// MARK: - MLX (Apple Silicon GPU/Neural Engine via MLX framework)

func benchmarkMLX(shape: Shape) -> Double {
    // f16 to match what GGUF Q4 dequant produces in the hot path.
    let a = MLXRandom.normal([shape.m, shape.k], dtype: .float16)
    let b = MLXRandom.normal([shape.k, shape.n], dtype: .float16)

    // Warm up (compile shaders, allocate buffers).
    for _ in 0..<warmupIters {
        let c = matmul(a, b)
        eval(c) // force completion
    }

    var elapsed: [Double] = []
    elapsed.reserveCapacity(timedIters)
    for _ in 0..<timedIters {
        let t0 = DispatchTime.now()
        let c = matmul(a, b)
        eval(c)
        let t1 = DispatchTime.now()
        elapsed.append(Double(t1.uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000)
    }
    return p50(elapsed)
}

// MARK: - Stats

func p50(_ samples: [Double]) -> Double {
    let sorted = samples.sorted()
    return sorted[sorted.count / 2]
}

func p95(_ samples: [Double]) -> Double {
    let sorted = samples.sorted()
    return sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
}

// MARK: - Report

print("goleta-infer Phase 0 spike — MLX vs Accelerate matmul (p50 latency, ms)")
print("Hardware: Apple Silicon (\(getHardwareDescription()))")
print("Iterations: warmup=\(warmupIters), timed=\(timedIters)")
print()
// Swift's String(format:) only accepts %@ for Swift strings (not %s, that's C-string).
// Pad columns by hand so the table stays aligned.
func padR(_ s: String, _ w: Int) -> String {
    s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
}
func padL(_ s: String, _ w: Int) -> String {
    s.count >= w ? s : String(repeating: " ", count: w - s.count) + s
}

print("\(padR("shape", 22)) | \(padL("Accelerate", 12)) | \(padL("MLX", 12)) | \(padL("speedup", 10)) | verdict")
print(String(repeating: "-", count: 80))

let gateThreshold = 1.5

var allPassed = true
for shape in shapes {
    let accelP50 = benchmarkAccelerate(shape: shape)
    let mlxP50 = benchmarkMLX(shape: shape)
    let speedup = accelP50 / mlxP50
    let verdict: String
    if speedup >= gateThreshold {
        verdict = "✓ pass"
    } else if speedup >= 1.0 {
        verdict = "× under-gate (\(String(format: "%.2f", speedup))×)"
        allPassed = false
    } else {
        verdict = "✗ slower than Accelerate"
        allPassed = false
    }
    let accelStr = String(format: "%.3f ms", accelP50)
    let mlxStr   = String(format: "%.3f ms", mlxP50)
    let spStr    = String(format: "%.2f×", speedup)
    print("\(padR(shape.name, 22)) | \(padL(accelStr, 12)) | \(padL(mlxStr, 12)) | \(padL(spStr, 10)) | \(verdict)")
}

print()
if allPassed {
    print("✅ Phase 0 Gate: PASS — every shape ≥\(gateThreshold)× Accelerate.")
    print("   Recommend proceeding to Phase 1 W25 (real kernel implementation).")
} else {
    print("⚠️ Phase 0 Gate: PARTIAL or FAIL.")
    print("   Decision rule (umbrella plan §13.3): if any decode shape misses gate,")
    print("   revert §13 and fall back to P3 (wait for FLock Pocket).")
    print("   Re-evaluation triggers in §13.6 specify when to revisit.")
}

func getHardwareDescription() -> String {
    var size: size_t = 0
    sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
    var model = [CChar](repeating: 0, count: size)
    sysctlbyname("machdep.cpu.brand_string", &model, &size, nil, 0)
    return String(cString: model)
}
