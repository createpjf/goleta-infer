// goleta-infer ggml-mlx backend — Apple MLX dense decode kernels.
//
// This header exposes the public C API for the MLX backend. The actual
// kernel work (matmul, RMSNorm, RoPE, SDPA, KV cache) is delegated to
// MLXKernels.swift via @_cdecl bridges; this file only defines the
// ggml-backend interface that ggml-backend-reg.cpp picks up at runtime.
//
// Phase 1 status (2026-04-29):
//   - Stub implementation: ggml_backend_mlx_reg() returns a valid registry
//     reporting 0 devices. The runtime registers the backend but never
//     dispatches to it; ops fall through to ggml-metal / ggml-cpu.
//   - Phase 2 (W20-W22 per plan §2) replaces the stub with a real device
//     enumeration + supports_op + graph_compute that calls into Swift.
//
// See plan: docs/plans/2026-04-29-goleta-infer-qwen35-path-x.md (Goleta repo).

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Construct a backend instance for the default MLX device. Returns NULL
// during Phase 1 (no devices reported); Phase 2 returns a real backend.
GGML_BACKEND_API ggml_backend_t ggml_backend_mlx_init(void);

// Type predicate: true iff `backend` was returned by ggml_backend_mlx_init().
GGML_BACKEND_API bool ggml_backend_is_mlx(ggml_backend_t backend);

// Backend registry entry point. Always returns a non-NULL registry so the
// global registration in ggml-backend-reg.cpp stays consistent across
// builds, even when no MLX device is available at runtime.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_mlx_reg(void);

#ifdef __cplusplus
}
#endif
