// ggml-mlx.c — Apple MLX backend stub for goleta-infer (Phase 0).
//
// Every entry point in this file is a deliberate stub. The runtime
// backend dispatcher should treat any `false` return / NULL ctx as
// "MLX declined; fall back to the next available backend (ggml-metal
// → CPU)". Phase 0 spike measurement happens in the SwiftPM target
// (a standalone benchmark project) that exercises MLX kernels
// directly without going through this dispatcher — the spike's job
// is to decide whether real kernels are worth implementing here.
//
// When Phase 1 lands, the real implementation will be a thin C-side
// dispatch that bridges to MLXKernels.swift via `@_cdecl` symbols.
// The Swift impl owns the MLX framework calls; this file owns the
// ggml-backend ABI surface and parameter marshalling.

#include "ggml-mlx.h"

#include <stdio.h>
#include <stdlib.h>

// Phase 0 logging: stderr by default. Real impl will route through
// the registered ggml_log_callback for parity with ggml-metal.
static ggml_log_callback s_log_callback = NULL;
static void *            s_log_user_data = NULL;

#define GGML_MLX_LOG_WARN(msg) do {                                       \
    if (s_log_callback) {                                                 \
        s_log_callback(GGML_LOG_LEVEL_WARN, "[ggml-mlx] " msg "\n",       \
                       s_log_user_data);                                  \
    } else {                                                              \
        fprintf(stderr, "[ggml-mlx] " msg "\n");                          \
    }                                                                     \
} while (0)

void ggml_mlx_log_set_callback(ggml_log_callback log_callback, void * user_data) {
    s_log_callback = log_callback;
    s_log_user_data = user_data;
}

struct ggml_mlx_context {
    // Phase 0 placeholder — exists only so callers can hold an
    // opaque pointer. Real impl will track MLX device handle,
    // command queue, kernel pipeline cache, mapped buffer table.
    int placeholder;
};

struct ggml_mlx_context * ggml_mlx_init(int n_cb) {
    (void)n_cb;
    GGML_MLX_LOG_WARN("init: Phase 0 stub — MLX backend not implemented yet, callers must fall back to ggml-metal");
    // Return NULL so the runtime treats MLX as unavailable.
    // Phase 1 W25 wires real init.
    return NULL;
}

void ggml_mlx_free(struct ggml_mlx_context * ctx) {
    // Tolerate NULL — this is the only path Phase 0 callers see.
    if (ctx) free(ctx);
}

bool ggml_mlx_add_buffer(
        struct ggml_mlx_context * ctx,
        const char * name,
        void   * data,
        size_t   size,
        size_t   max_size) {
    (void)ctx; (void)name; (void)data; (void)size; (void)max_size;
    return false;
}

void ggml_mlx_set_tensor(struct ggml_mlx_context * ctx, struct ggml_tensor * t) {
    (void)ctx; (void)t;
}

void ggml_mlx_get_tensor(struct ggml_mlx_context * ctx, struct ggml_tensor * t) {
    (void)ctx; (void)t;
}

bool ggml_mlx_graph_compute(struct ggml_mlx_context * ctx, struct ggml_cgraph * gf) {
    (void)ctx; (void)gf;
    return false;
}

//
// Sparse-activation hot-path stubs. All return false in Phase 0.
//

bool ggml_mlx_sparse_matmul(
        struct ggml_mlx_context * ctx,
        struct ggml_tensor      * dst,
        const struct ggml_tensor * weight_hot,
        const struct ggml_tensor * input,
        const int               * hot_indices,
        int                       n_hot) {
    (void)ctx; (void)dst; (void)weight_hot; (void)input;
    (void)hot_indices; (void)n_hot;
    return false;
}

bool ggml_mlx_predict_activation(
        struct ggml_mlx_context * ctx,
        struct ggml_tensor      * predictor_out,
        const struct ggml_tensor * predictor_w,
        const struct ggml_tensor * input) {
    (void)ctx; (void)predictor_out; (void)predictor_w; (void)input;
    return false;
}

bool ggml_mlx_rms_norm(
        struct ggml_mlx_context * ctx,
        struct ggml_tensor      * dst,
        const struct ggml_tensor * src,
        const struct ggml_tensor * weight,
        float                     eps) {
    (void)ctx; (void)dst; (void)src; (void)weight; (void)eps;
    return false;
}

bool ggml_mlx_rope(
        struct ggml_mlx_context * ctx,
        struct ggml_tensor      * dst,
        const struct ggml_tensor * src,
        int                       n_past,
        int                       n_dims,
        int                       mode) {
    (void)ctx; (void)dst; (void)src;
    (void)n_past; (void)n_dims; (void)mode;
    return false;
}
