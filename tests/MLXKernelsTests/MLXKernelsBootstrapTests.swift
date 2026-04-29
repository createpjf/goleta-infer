// MLXKernelsBootstrapTests.swift
//
// Verifies the C/Swift kernel-table registration handshake.
//
// Phase 1 (commit 1aac2fe11) tested that bootstrap+NULL kept stub mode
// — that test is updated in Phase 2 Task 2.1a because bootstrap() now
// builds and registers a real table. The handshake is the same, only
// the assertions about kernelsAvailable flip.

import XCTest
@testable import MLXKernels

final class MLXKernelsBootstrapTests: XCTestCase {

    override func tearDown() {
        // Each test should leave the backend in a known state for the next.
        MLXKernels.shutdown()
        super.tearDown()
    }

    /// bootstrap() now registers a populated table. After Task 2.1a at
    /// least one slot (dense_matmul_f16) is non-nil; kernelsAvailable
    /// must reflect that.
    func test_bootstrap_registers_table() {
        let ok = MLXKernels.bootstrap()
        XCTAssertTrue(ok, "C side should accept the populated kernel table")
        XCTAssertTrue(
            MLXKernels.kernelsAvailable,
            "After bootstrap the backend should report kernels available"
        )
    }

    /// shutdown() returns the backend to stub mode (NULL table). Used by
    /// app teardown and by tests that want isolation.
    func test_shutdown_returns_to_stub_mode() {
        MLXKernels.bootstrap()
        XCTAssertTrue(MLXKernels.kernelsAvailable)

        let ok = MLXKernels.shutdown()
        XCTAssertTrue(ok)
        XCTAssertFalse(
            MLXKernels.kernelsAvailable,
            "After shutdown the backend should be back in stub mode"
        )
    }

    /// Both entry points are idempotent. Multiple bootstrap()s leave
    /// kernels active; multiple shutdown()s leave the backend in stub.
    func test_bootstrap_and_shutdown_are_idempotent() {
        MLXKernels.bootstrap()
        MLXKernels.bootstrap()
        MLXKernels.bootstrap()
        XCTAssertTrue(MLXKernels.kernelsAvailable)

        MLXKernels.shutdown()
        MLXKernels.shutdown()
        XCTAssertFalse(MLXKernels.kernelsAvailable)
    }
}
