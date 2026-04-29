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
#include <mutex>

// ---------------------------------------------------------------------------
// Kernel registration (called from MLXKernels.swift at module-load time)
// ---------------------------------------------------------------------------
//
// Concurrency model:
//   - Registration is rare (typically once at app startup) and serializes
//     through g_mlx_register_mutex while it copies the caller's table into
//     static storage. This keeps the read path lock-free.
//   - Reads are hot (every ggml graph_compute checks if MLX is available)
//     and use a single std::atomic<const goleta_mlx_kernel_table *>.load()
//     against g_mlx_table_storage's address.
//   - Storing into static storage frees the caller of lifetime concerns —
//     Swift can pass a stack-allocated struct from bootstrap() and it's
//     safe even after bootstrap returns.

namespace {
    // Static storage for the registered table. Copied at registration time;
    // pointer to this storage lives forever. Either zero-initialized (stub
    // mode) or holds the most recently registered Swift kernel table.
    goleta_mlx_kernel_table g_mlx_table_storage{};

    // Atomic pointer: NULL = stub mode, non-NULL = active. Always points
    // either at &g_mlx_table_storage or nullptr; never at caller storage.
    std::atomic<const goleta_mlx_kernel_table *> g_mlx_kernels{nullptr};

    // Serializes registration writes (very cold path).
    std::mutex g_mlx_register_mutex;
}

extern "C" bool goleta_mlx_register_kernels(const goleta_mlx_kernel_table * table) {
    std::lock_guard<std::mutex> lock(g_mlx_register_mutex);

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

    // Copy into static storage so caller's storage doesn't have to outlive
    // the registration. Swift bootstrap() is now safe to pass &localTable.
    g_mlx_table_storage = *table;
    g_mlx_kernels.store(&g_mlx_table_storage, std::memory_order_release);
    GGML_LOG_INFO("ggml-mlx: kernels registered (ABI v%d)\n", table->abi_version);
    return true;
}

extern "C" bool goleta_mlx_kernels_available(void) {
    return g_mlx_kernels.load(std::memory_order_acquire) != nullptr;
}

// ---------------------------------------------------------------------------
// Backend implementation (Phase 2 Task 2.1b)
//
// Models on ggml-blas: a host-memory "boost" backend that intercepts a
// narrow set of ops it can accelerate (currently MUL_MAT only) and lets
// everything else fall through to ggml-cpu / ggml-metal. Buffer type is
// shared with ggml-cpu so tensors don't need copies.
//
// Per-shape dispatch threshold (Phase 0 spike): MLX wins when N >= 2048,
// loses for smaller N. We claim MUL_MAT only above that threshold and
// only for fp16 inputs (Phase 2.1b kernel coverage).
// ---------------------------------------------------------------------------

namespace {
    // Dispatch threshold: at and above this output size, MLX matmul beats
    // Accelerate by 1.5-4.5× per the Phase 0 spike data. Below this threshold
    // MLX command-buffer dispatch overhead dominates (e.g., 0.48× on the
    // 4096×1100 sparse_10pct shape) and we let ggml-cpu handle it.
    constexpr int64_t kMlxMulMatMinN = 2048;
}

struct ggml_backend_mlx_context {
    // Phase 2.1b context is intentionally minimal — MLX manages its own
    // device + command buffer state behind the kernel table function pointers.
    // Phase 2.6 (KV cache) will add a per-context cache handle here.
    int placeholder = 0;
};

static bool ggml_backend_mlx_can_run_mul_mat(const struct ggml_tensor * op) {
    const auto * src0 = op->src[0];
    const auto * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    // Kernel coverage: fp16-only matmul in Phase 2.1b.
    // Phase 2.2 will extend this to Q4_K_M (dequant on the fly).
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F16) {
        return false;
    }

    // Shape constraint: 2D contiguous, K matches between the two operands.
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {  // K dim must match
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        // 4D / batched matmul not yet supported; needs broadcasting logic.
        return false;
    }

    // Per-shape threshold: only claim MUL_MAT when N >= kMlxMulMatMinN.
    // src0 layout in ggml is [K, N] so N == ne01.
    const int64_t N = src0->ne[1];
    if (N < kMlxMulMatMinN) {
        return false;
    }

    // Output dtype: ggml MUL_MAT produces fp32 by convention; we synthesize
    // fp16 inside the kernel and the caller has to convert. Phase 2.1b only
    // claims when dst is also fp16 (skip the conversion path for now).
    if (op->type != GGML_TYPE_F16) {
        return false;
    }

    return true;
}

static void ggml_backend_mlx_mul_mat(const struct ggml_tensor * dst,
                                     const goleta_mlx_kernel_table * kernels) {
    const auto * src0 = dst->src[0];   // weight, ggml shape [K, N]
    const auto * src1 = dst->src[1];   // input,  ggml shape [K, M]

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    // Cap to int32 since the kernel table signature uses int (Phase 2 may
    // bump this to int64 once we hit a real model with > 2B elements).
    GGML_ASSERT(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX);

    const bool ok = kernels->mul_mat_f16_ggml(
        src1->data, (int32_t)M, (int32_t)K,
        src0->data, (int32_t)N, (int32_t)K,
        dst->data
    );
    GGML_ASSERT(ok && "mul_mat_f16_ggml kernel returned false on a shape we claimed to support");
}

// ---------------------------------------------------------------------------
// ggml_backend_i — the "compute the graph" interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_mlx_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "MLX";
}

static void ggml_backend_mlx_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_mlx_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_mlx_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    // The kernel table is published lock-free; load once per graph compute.
    const auto * kernels = g_mlx_kernels.load(std::memory_order_acquire);
    if (kernels == nullptr) {
        // Should never happen: ggml only schedules nodes here if init()
        // succeeded, which requires kernels to be available. Treat as a
        // logic bug.
        GGML_LOG_ERROR("ggml-mlx: graph_compute called with no kernels registered\n");
        return GGML_STATUS_FAILED;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_mlx_mul_mat(node, kernels);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // Pass-through ops: ggml's scheduler handles these, we just
                // need to claim them in supports_op so the graph stays on
                // our backend without unnecessary device hops.
                break;

            default:
                GGML_ABORT("ggml-mlx: scheduler dispatched op %s but graph_compute "
                           "doesn't handle it (supports_op accepted too widely?)",
                           ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_mlx_i = {
    /* .get_name                = */ ggml_backend_mlx_get_name,
    /* .free                    = */ ggml_backend_mlx_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_mlx_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_mlx_guid(void) {
    // Stable GUID for goleta-infer's MLX backend. Generated 2026-04-30 via
    // `python -c 'import uuid; print(uuid.uuid4().hex)'`. Don't change this
    // — backend detection in ggml_backend_is_mlx() relies on it.
    static ggml_guid guid = {
        0x6c, 0x4d, 0x58, 0x9b, 0xa1, 0xc7, 0x4f, 0x32,
        0x88, 0xea, 0x29, 0x1d, 0xb5, 0x73, 0xfe, 0x40
    };
    return &guid;
}

// ---------------------------------------------------------------------------
// ggml_backend_device_i — what ggml's scheduler queries to decide whether
// to put an op on this device
// ---------------------------------------------------------------------------

static const char * ggml_backend_mlx_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "MLX";
}

static const char * ggml_backend_mlx_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Apple MLX (goleta-infer)";
}

static void ggml_backend_mlx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // Unified memory; no separate device pool to report.
    GGML_UNUSED(dev);
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_mlx_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // ACCEL = "boost" device that accelerates specific ops; main backend
    // (ggml-cpu/metal) handles everything else. Same model as ggml-blas.
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_mlx_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_mlx_device_get_name(dev);
    props->description = ggml_backend_mlx_device_get_description(dev);
    props->type        = ggml_backend_mlx_device_get_type(dev);
    ggml_backend_mlx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_mlx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_mlx_init();
}

static ggml_backend_buffer_type_t ggml_backend_mlx_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // Share buffer type with ggml-cpu: both consume host-resident contiguous
    // memory and the kernel reads it directly via the registered Swift bridge.
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_mlx_device_buffer_from_host_ptr(
    ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_mlx_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    // No kernels = nothing supported. ggml's scheduler will route everywhere
    // else and our graph_compute is never called.
    if (g_mlx_kernels.load(std::memory_order_acquire) == nullptr) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            // Pass-through ops: claim them so consecutive MUL_MATs stay
            // on this device without a round-trip through ggml-cpu.
            return true;

        case GGML_OP_MUL_MAT:
            return ggml_backend_mlx_can_run_mul_mat(op);

        default:
            return false;
    }
}

static bool ggml_backend_mlx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_mlx_device_i = {
    /* .get_name             = */ ggml_backend_mlx_device_get_name,
    /* .get_description      = */ ggml_backend_mlx_device_get_description,
    /* .get_memory           = */ ggml_backend_mlx_device_get_memory,
    /* .get_type             = */ ggml_backend_mlx_device_get_type,
    /* .get_props            = */ ggml_backend_mlx_device_get_props,
    /* .init_backend         = */ ggml_backend_mlx_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_mlx_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_mlx_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_mlx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_mlx_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------------------------------------------------------------------
// ggml_backend_reg_i — top-level registry, advertises one device when
// kernels are available
// ---------------------------------------------------------------------------

static const char * ggml_backend_mlx_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "MLX";
}

static size_t ggml_backend_mlx_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    // No kernels registered = 0 devices, ggml never tries to dispatch to us.
    // Once MLXKernels.bootstrap() runs, advertise one logical device.
    return goleta_mlx_kernels_available() ? 1 : 0;
}

static ggml_backend_dev_t ggml_backend_mlx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_mlx_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &device;
}

static void * ggml_backend_mlx_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
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
    if (!goleta_mlx_kernels_available()) {
        GGML_LOG_INFO("ggml-mlx: no kernels registered — backend not instantiated. "
                      "Call MLXKernels.bootstrap() (Swift) before llama_backend_init.\n");
        return nullptr;
    }

    auto * ctx = new ggml_backend_mlx_context{};
    auto * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_mlx_guid(),
        /* .iface   = */ ggml_backend_mlx_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_mlx_reg(), 0),
        /* .context = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_mlx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_mlx_guid());
}

GGML_BACKEND_DL_IMPL(ggml_backend_mlx_reg)
