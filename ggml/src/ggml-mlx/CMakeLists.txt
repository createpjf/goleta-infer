# goleta-infer ggml-mlx backend build wiring.
#
# Phase 1: stub C++ backend only (returns 0 devices). The Apple MLX
# Swift kernels live in MLXKernels.swift inside the SwiftPM target
# and are linked in via xcodebuild, not CMake. CMake's job is just to
# compile the C-side ggml-backend-impl.h shim and let it call the
# Swift @_cdecl symbols at link time.
#
# Phase 2: when MLXKernels.swift becomes real, this file gains
# `target_link_libraries(ggml-mlx PRIVATE MLXKernels)` and a
# ${CMAKE_OSX_ARCHITECTURES} guard for arm64 only.
#
# Required: macOS 14+ (mlx-swift platform requirement). On other
# platforms the parent ggml_add_backend(MLX) macro short-circuits
# because GGML_MLX defaults OFF off-Apple.

if (NOT APPLE)
    message(FATAL_ERROR "GGML_MLX requires Apple Silicon (macOS 14+ / iOS 17+).")
endif()

ggml_add_backend_library(ggml-mlx
                         ggml-mlx.cpp
                        )

# Foundation framework is needed even for the stub (NSLog, base ObjC
# runtime symbols pulled in by future Swift bridging in Phase 2).
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
target_link_libraries(ggml-mlx PRIVATE ${FOUNDATION_LIBRARY})

message(STATUS "ggml-mlx: Phase 1 stub backend wired (no kernels yet)")
