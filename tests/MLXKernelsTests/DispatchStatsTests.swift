// DispatchStatsTests.swift — Phase 2 Task 2.7
//
// Verifies the dispatch-counter API surface (Swift accessor + reset).
// Actual counter increments happen inside ggml-mlx.cpp's graph_compute,
// which only runs when ggml's scheduler dispatches an op to the MLX
// backend — that path is exercised by Phase 3 integration tests
// (PowerInferProvider end-to-end). For Phase 2.7 we just confirm the
// API itself is wired correctly.

import XCTest
@testable import MLXKernels

final class DispatchStatsTests: XCTestCase {

    /// reset() zeros every counter; subsequent snapshot reflects that.
    func test_reset_zeros_all_counters() {
        MLXKernels.resetDispatchStats()
        let stats = MLXKernels.dispatchStats
        XCTAssertEqual(stats.mulMatF16Dispatched, 0)
        XCTAssertEqual(stats.mulMatQ4KMDispatched, 0)
        XCTAssertEqual(stats.mulMatRejectedBelowN, 0)
        XCTAssertEqual(stats.mulMatRejectedUnsupported, 0)
    }

    /// Snapshot is a value type — taking it twice without intervening
    /// dispatcher activity yields equal stats.
    func test_snapshot_is_consistent() {
        MLXKernels.resetDispatchStats()
        let a = MLXKernels.dispatchStats
        let b = MLXKernels.dispatchStats
        XCTAssertEqual(a, b)
    }

    /// Computed convenience properties.
    func test_total_helpers() {
        MLXKernels.resetDispatchStats()
        let s = MLXKernels.dispatchStats
        XCTAssertEqual(s.totalDispatched, 0)
        XCTAssertEqual(s.totalRejected, 0)
    }

    /// Bootstrap doesn't dispatch any ops by itself.
    /// (Confirms the architecture: registration is dormant until ggml's
    /// scheduler routes an op through the MLX backend.)
    func test_bootstrap_alone_does_not_increment_dispatcher() {
        MLXKernels.resetDispatchStats()
        MLXKernels.bootstrap()
        defer { MLXKernels.shutdown() }
        XCTAssertEqual(MLXKernels.dispatchStats.totalDispatched, 0,
                       "bootstrap registers kernels but does not dispatch any ops")
    }
}
