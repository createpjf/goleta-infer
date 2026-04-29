#!/bin/bash
# goleta-infer Phase 1 smoke test: load Qwen 3.5 9B Q4_K_M GGUF via the
# CPU+Metal path (no ggml-mlx involved — that's a Phase 2 path).
#
# Usage:
#   tests/manual/qwen35-smoke.sh
#
# Env overrides:
#   MODEL — path to GGUF (default: ~/.goleta/models/Qwen3.5-9B-Q4_K_M.gguf)
#   BIN   — path to llama-cli (default: ./build/bin/llama-cli)
#
# Pass criterion: stdout contains the answer "4" to the prompt
# "What is 2 plus 2? Answer with just the number."
# Run is fully deterministic (seed=42, temp=0).
#
# Phase 2 will add a sibling script with LLAMA_BACKEND=mlx envvar.
set -euo pipefail

MODEL=${MODEL:-$HOME/.goleta/models/Qwen3.5-9B-Q4_K_M.gguf}
BIN=${BIN:-./build/bin/llama-cli}

if [ ! -f "$MODEL" ]; then
    echo "ERROR: missing model at $MODEL" >&2
    echo "Run: hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/.goleta/models/" >&2
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "ERROR: missing llama-cli at $BIN" >&2
    echo "Run: cmake --build build --target llama-cli -j 8" >&2
    exit 1
fi

# Qwen 3.5 is a reasoning model that emits a Thinking Process before the
# final answer. Allow 512 tokens so the actual answer makes it out.
"$BIN" -m "$MODEL" \
    -p "What is 2 plus 2? Answer with just the number, no explanation." \
    -n 512 \
    --temp 0 \
    --seed 42 \
    --no-display-prompt \
    --no-warmup \
    -st \
    2>/dev/null
