// goleta-infer ggml-mlx backend — Phase 1 stub implementation.
//
// This file lives at ggml/src/ggml-mlx/ggml-mlx.cpp and is wired into
// the build via ggml_add_backend(MLX) in ../CMakeLists.txt.
//
// What this file does today (Phase 1):
//   - Provides a valid ggml_backend_reg_t with 0 devices.
//   - ggml-backend-reg.cpp registers it conditionally on GGML_USE_MLX.
//   - At runtime the registry sees "no devices" and falls through to
//     ggml-metal / ggml-cpu without ever dispatching to MLX.
//
// What this file will do in Phase 2 (W20-W22 per plan §2):
//   - Real device enumeration (one logical device per Apple GPU).
//   - supports_op() returns true for the dense decode shapes the
//     spike validated as wins (matmul N>=2048, RMSNorm, RoPE, SDPA).
//   - graph_compute() marshals ggml_tensor → MLXArray and calls
//     into MLXKernels.swift via @_cdecl bridges.
//
// Reference templates: ggml/src/ggml-blas/ggml-blas.cpp (closest
// shape — single logical device, no GPU resources to manage in the
// stub) and ggml/src/ggml-metal/ggml-metal.cpp (more complete, used
// once kernels land).

#include "ggml-mlx.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <atomic>
#include <cstring>

// ---------------------------------------------------------------------------
// Kernel registration (called from MLXKernels.swift at module-load time)
// ---------------------------------------------------------------------------

namespace {
    // Global, atomically swapped. nullptr = no Swift kernels registered yet,
    // backend stays in Phase 1 stub mode (0 devices).
    std::atomic<const goleta_mlx_kernel_table *> g_mlx_kernels{nullptr};
}

extern "C" bool goleta_mlx_register_kernels(const goleta_mlx_kernel_table * table) {
    if (table == nullptr) {
        // Deregister: used by tests + during teardown to force stub mode.
        g_mlx_kernels.store(nullptr, std::memory_order_release);
        GGML_LOG_INFO("ggml-mlx: kernels deregistered\n");
        return true;
    }

    if (table->abi_version != GOLETA_MLX_KERNEL_ABI_VERSION) {
        GGML_LOG_ERROR("ggml-mlx: kernel ABI mismatch (got %d, expected %d) — refusing registration\n",
                       table->abi_version, GOLETA_MLX_KERNEL_ABI_VERSION);
        return false;
    }

    g_mlx_kernels.store(table, std::memory_order_release);
    GGML_LOG_INFO("ggml-mlx: kernels registered (ABI v%d)\n", table->abi_version);
    return true;
}

extern "C" bool goleta_mlx_kernels_available(void) {
    return g_mlx_kernels.load(std::memory_order_acquire) != nullptr;
}

// ---------------------------------------------------------------------------
// Backend registry interface (mandatory entry points)
// ---------------------------------------------------------------------------

static const char * ggml_backend_mlx_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "MLX";
}

static size_t ggml_backend_mlx_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    // Phase 1: report zero devices so the runtime never tries to
    // dispatch ops to us. Phase 2 returns 1 (or N for multi-GPU Mac
    // Pro) once init() actually constructs an MLXContext.
    return 0;
}

static ggml_backend_dev_t ggml_backend_mlx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    GGML_UNUSED(index);
    // Unreachable while get_device_count returns 0, but the registry
    // contract requires the function pointer to be non-NULL.
    GGML_ABORT("ggml-mlx: Phase 1 stub has no devices; ggml_backend_mlx_reg_get_device should not be called");
}

static void * ggml_backend_mlx_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    // Phase 1 exposes no extension procs. Phase 2 will route
    // "ggml_backend_mlx_set_dispatch_threshold" et al. through here.
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_mlx_reg_i = {
    /* .get_name         = */ ggml_backend_mlx_reg_get_name,
    /* .get_device_count = */ ggml_backend_mlx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_mlx_reg_get_device,
    /* .get_proc_address = */ ggml_backend_mlx_reg_get_proc_address,
};

// ---------------------------------------------------------------------------
// Public API (declared in ggml/include/ggml-mlx.h)
// ---------------------------------------------------------------------------

ggml_backend_reg_t ggml_backend_mlx_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_mlx_reg_i,
        /* .context     = */ nullptr,
    };

    return &reg;
}

ggml_backend_t ggml_backend_mlx_init(void) {
    // Phase 1 stub: never produce a real backend. Callers should
    // check the return value and fall back to ggml-metal / ggml-cpu.
    GGML_LOG_INFO("ggml-mlx: Phase 1 stub — no MLX backend instantiated\n");
    return nullptr;
}

bool ggml_backend_is_mlx(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    return std::strcmp(ggml_backend_name(backend), "MLX") == 0;
}

GGML_BACKEND_DL_IMPL(ggml_backend_mlx_reg)
