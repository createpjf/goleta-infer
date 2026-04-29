# Phase 1 ggml-metal Baseline (Task 1.4)

**Hardware:** Apple M4 Max (per Phase 0 spike audit)
**Model:** `unsloth/Qwen3.5-9B-GGUF` Q4_K_M, 5,680,522,464 bytes
**Backend:** ggml-metal (no MLX yet — `GGML_USE_MLX` defined but
            `ggml_backend_mlx_reg` reports 0 devices in Phase 1 stub)
**Build:** `b8974-fdf36005f` (commit `a304bf14a` + the .partial → .gguf rename)
**Smoke test command:**
```
tests/manual/qwen35-smoke.sh
```

## Result: PASS

The model loads, registers all expected modalities (`text` only — no
mmproj loaded as documented in plan §3a), runs deterministically, and
produces the correct final answer "4" to the prompt
"What is 2 plus 2? Answer with just the number, no explanation."

Reasoning preamble (`Thinking Process:` block) appears before the
final answer because Qwen 3.5 9B is a reasoning model. This is
expected behavior; users who want raw answers should set the system
prompt to disable thinking (or use the `/no_think` directive in the
prompt — both are documented Qwen 3.5 conventions and are not part
of the smoke test contract).

## Performance baseline (ggml-metal, M4 Max, batch=1 decode)

| Metric | Rate |
|---|---|
| Prompt processing | 267.0 t/s |
| Decode generation | **47.3 t/s** |

The decode generation rate (47.3 t/s) is the binding number for the
Phase 2 perf gate (plan §6 success criteria):

- Phase 2 freeze gate: MLX decode ≥ **1.4× = 66.2 t/s**
- Phase 0 spike speedups (1.59–2.41× on dense decode shapes) project
  this to land at ~75–114 t/s once kernels are real

Prefill (prompt processing) gate is ≥ 3× this = 801 t/s. Spike showed
4.5× on `attn_qkv_prefill`, so headroom is comfortable.

## Reference output

Saved at `tests/manual/qwen35-smoke.expected.txt` for Phase 2 Task 2.8
deterministic equivalence check (`LLAMA_BACKEND=mlx` should produce
byte-identical output at temp=0, seed=42).
