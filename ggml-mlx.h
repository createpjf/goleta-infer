// ggml-mlx — Apple MLX backend for goleta-infer (Phase 0 stub).
//
// Status: STUB. The functions declared here will be implemented by
// `MLXKernels.swift` (in the SwiftPM target) via Swift's `@_cdecl`
// export mechanism. Until Phase 1 lands real kernels, every entry
// point returns a `not-implemented` error and the runtime backend
// dispatcher should fall through to ggml-metal / CPU.
//
// Why a separate backend instead of extending ggml-metal:
// MLX is Apple's higher-level framework with auto-quantization,
// unified-memory tensor scheduling, and a Swift-first API surface.
// ggml-metal speaks Metal Shading Language directly. We want the
// high-level path (MLX) for the sparse-activation hot path
// (predict-then-gather hot neurons + sparse matmul) where MLX's
// fused operators outperform hand-written MSL kernels at our
// engineering budget.
//
// Backend layout mirrors ggml-metal.h — same lifecycle, same
// graph_compute hook, same tensor mapping API. Differences are
// confined to the kernel implementations behind the API.
//
// Phase 0 contract (this file): declarations only. Real bodies
// land in Phase 1 (W25-W28).

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

// Max buffers / commands kept identical to ggml-metal so a future
// runtime can pick either backend without mismatched limits.
#define GGML_MLX_MAX_BUFFERS         64
#define GGML_MLX_MAX_COMMAND_BUFFERS 32

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// Lifecycle
//

struct ggml_mlx_context;

// Phase 0 stub: returns NULL and logs a warning. Real impl will
// initialize the MLX device handle, command queue, and shader cache.
struct ggml_mlx_context * ggml_mlx_init(int n_cb);

// Releases an MLX context. Safe to call with NULL (Phase 0 path).
void ggml_mlx_free(struct ggml_mlx_context * ctx);

// Logging mirror of ggml_metal_log_set_callback. Plumbed in Phase 0
// so user-code wiring doesn't need to change between stub and real.
void ggml_mlx_log_set_callback(ggml_log_callback log_callback, void * user_data);

//
// Memory mapping
//
// Phase 0: all six functions below return false / no-op. The real
// implementations will live in MLXKernels.swift and bridge via
// `@_cdecl`. Sketched here so headers compile cleanly under
// `-DLLAMA_MLX=ON` and so the runtime backend table can register
// MLX as a candidate device that simply declines every operation
// today.

// Map a host buffer into the MLX device's view. Returns false in
// Phase 0 — caller must fall back to ggml-metal / CPU.
bool ggml_mlx_add_buffer(
        struct ggml_mlx_context * ctx,
        const char * name,
        void   * data,
        size_t   size,
        size_t   max_size);

// Set / get a tensor's bytes via the MLX context. No-op in Phase 0.
void ggml_mlx_set_tensor(struct ggml_mlx_context * ctx, struct ggml_tensor * t);
void ggml_mlx_get_tensor(struct ggml_mlx_context * ctx, struct ggml_tensor * t);

//
// Graph compute
//

// Phase 0: returns immediately without computing. Caller MUST check
// the return value: false = backend declined, fall back to a real
// backend (ggml-metal / CPU).
bool ggml_mlx_graph_compute(struct ggml_mlx_context * ctx, struct ggml_cgraph * gf);

//
// Sparse-activation hot path (Phase 1+ targets)
//
// These are the four operations PowerInfer's hot path needs that
// would benefit most from MLX. Phase 0 declarations only — Phase 1
// W25 implements them in MLXKernels.swift via @_cdecl.
//

// Sparse matmul on the predicted-hot neurons. Inputs are pre-gathered
// hot indices + their weights. Phase 0: returns false (not impl).
bool ggml_mlx_sparse_matmul(
        struct ggml_mlx_context * ctx,
        struct ggml_tensor      * dst,
        const struct ggml_tensor * weight_hot,
        const struct ggml_tensor * input,
        const int               * hot_indices,
        int                       n_hot);

// Activation predictor (small MLP that decides which neurons are
// "hot"). Phase 0 stub.
bool ggml_mlx_predict_activation(
        struct ggml_mlx_context * ctx,
        struct ggml_tensor      * predictor_out,
        const struct ggml_tensor * predictor_w,
        const struct ggml_tensor * input);

// RMSNorm fused kernel. Phase 0 stub.
bool ggml_mlx_rms_norm(
        struct ggml_mlx_context * ctx,
        struct ggml_tensor      * dst,
        const struct ggml_tensor * src,
        const struct ggml_tensor * weight,
        float                     eps);

// Rotary position embedding. Phase 0 stub.
bool ggml_mlx_rope(
        struct ggml_mlx_context * ctx,
        struct ggml_tensor      * dst,
        const struct ggml_tensor * src,
        int                       n_past,
        int                       n_dims,
        int                       mode);

#ifdef __cplusplus
}
#endif
