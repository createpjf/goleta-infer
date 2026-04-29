# goleta-infer Phase 0 spike

Standalone benchmark harness that answers a single question:

> **Does Apple MLX hit ≥1.5× Accelerate-on-CPU on the matrix shapes
> PowerInfer's hot path actually uses?**

If yes → Phase 1 W25 starts: real `ggml-mlx` kernel implementation in
`MLXKernels.swift` + production-quality C bridge in `ggml-mlx.c`.

If no → revert §13 of the umbrella plan, fall back to P3 (wait for
FLock Pocket to ship the inference engine; Goleta talks to it via
existing OpenAI-compat HTTP).

## Run

**Important: `swift run` does not work.** SwiftPM CLI cannot compile
Metal shaders, so the resulting binary segfaults at runtime with
"Failed to load default metallib". Per mlx-swift's own README this
is by design — use `xcodebuild`:

```bash
cd mlx-spike
# Xcode 26+ also needs the Metal Toolchain component:
#   xcodebuild -downloadComponent MetalToolchain   # ~688 MB, one-time
xcodebuild -scheme MLXSpike -configuration Release \
    -destination 'platform=macOS,arch=arm64' \
    -derivedDataPath ./.xcbuild build
./.xcbuild/Build/Products/Release/MLXSpike
```

The xcodebuild path produces `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`
next to the executable, which MLX needs at runtime. First run pulls
`mlx-swift` (~150MB), compiles ~30 .metal kernels, then prints a
markdown-friendly comparison table:

```
shape                  | Accelerate   | MLX          | speedup    | verdict
--------------------------------------------------------------------------------
attn_qkv_decode        |  4.123 ms    |  2.341 ms    |    1.76×   | ✓ pass
ffn_gate_decode        |   ...
```

## What the spike measures

Six matmul shapes drawn from Bamboo-7B-DPO at f16:

| Shape | Why |
|---|---|
| `attn_qkv_decode` 4096×4096 | Single-token attention projection — every decoded token hits this 3× |
| `ffn_gate_decode` 4096×11008 | FFN up-projection — the heavy matmul in every decoder block |
| `ffn_down_decode` 11008×4096 | FFN down-projection — same |
| `sparse_10pct` 4096×1100 | What sparse activation reduces FFN to in the hot path (10% of neurons) |
| `sparse_30pct` 4096×3300 | More relaxed sparsity threshold |
| `attn_qkv_prefill` 128×4096×4096 | Prefill batch — see how MLX scales |

Three of the six (decode shapes with batch=1) are the real bottleneck.
Sparse hot-path shapes (10% / 30%) are the actual win for PowerInfer-
style sparse activation — if MLX wins decisively on those, the
integration is worth it.

## What the spike does NOT measure

- Full token generation throughput (no model loading, no KV cache,
  no attention masks). That's Phase 0 D5 / Phase 1 territory.
- Quantized matmul (Q4_0, Q4_K_M). MLX has Q4 support but the
  microbench uses f16 to keep the comparison clean. Phase 1 covers
  quantized paths.
- Memory pressure. MLX is unified-memory; Accelerate uses host RAM
  copies. Real-world delta may differ.

## Pass / fail rules

- **Pass**: every decode shape (`attn_qkv_decode`, `ffn_gate_decode`,
  `ffn_down_decode`) hits ≥1.5× Accelerate. Sparse shapes are
  bonus signal but not gate criteria — if MLX matmul itself is
  competitive, sparse on top will be too.
- **Partial pass**: some shapes hit gate, others don't. Inspect the
  failing ones — usually a kernel-size threshold MLX can't optimize
  past. Decide case by case.
- **Fail**: any decode shape is slower than Accelerate (i.e.
  speedup < 1.0×). Revert §13.

## Phase 0 Gate decision flow

```
                 ┌─────────────────┐
                 │  swift run      │
                 └────────┬────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
       all pass       partial pass    any decode
       (≥1.5× all)    (≥1.5× some)    < 1.0×
            │             │             │
            ▼             ▼             ▼
        Phase 1       case-by-case    Revert §13
        W25 GO!       review          → P3 (Pocket)
                                      → escape hatch
                                        (P1) docs
```

## Next: Phase 0 D5 — full inference benchmark

After the microbench gates pass, Phase 0 D5 runs:

1. Download Bamboo-7B-DPO Q4_K_M GGUF (~4GB) via `hf` CLI:
   ```bash
   hf download PowerInfer/Bamboo-DPO-v0.1-gguf Bamboo-DPO-v0.1.Q4_K_M.gguf \
       --local-dir ~/.goleta/models/
   ```

2. Build llama-bench from this fork:
   ```bash
   cmake -S . -B build -DLLAMA_METAL=ON -DLLAMA_BUILD_EXAMPLES=ON
   cmake --build build --target llama-bench -j 8
   ```

3. Baseline: ggml-metal tokens/sec
   ```bash
   ./build/bin/llama-bench -m ~/.goleta/models/Bamboo-DPO-v0.1.Q4_K_M.gguf \
       -p 512 -n 128 -t 1
   ```

4. Compare with the (yet-to-be-built) MLX path through goleta-infer.
   That's Phase 1 territory — D5 just confirms the baseline number
   is in the expected ballpark (~15-20 t/s on M2).

If both gates pass, the §13 timeline locks: Phase 1 W25-W28, Phase 2
W29-W30, Phase 3 W31-W32. Beta freeze pushed from W22 → W32 = launch
slips ~10 weeks. User accepted that trade-off when committing to §13
on 2026-04-29.
