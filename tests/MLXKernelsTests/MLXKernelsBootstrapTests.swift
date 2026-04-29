// MLXKernelsBootstrapTests.swift
//
// Phase 1 verification: the C/Swift bridge compiles, links, and the
// registration entry-point round-trips correctly. No actual kernels
// run yet — those tests land in Phase 2 (Tasks 2.1-2.6).

import XCTest
@testable import MLXKernels

final class MLXKernelsBootstrapTests: XCTestCase {

    /// Phase 1 stub: bootstrap returns true (deregistration of NULL table
    /// is accepted by the C side) and kernelsAvailable stays false.
    func test_phase1_bootstrap_keeps_stub_mode() {
        let ok = MLXKernels.bootstrap()
        XCTAssertTrue(ok, "C side should accept NULL deregistration as success")
        XCTAssertFalse(
            MLXKernels.kernelsAvailable,
            "Phase 1 ships no kernels; backend must remain in stub mode"
        )
    }

    /// Bootstrap is safe to call multiple times.
    func test_bootstrap_is_idempotent() {
        MLXKernels.bootstrap()
        MLXKernels.bootstrap()
        MLXKernels.bootstrap()
        XCTAssertFalse(MLXKernels.kernelsAvailable)
    }
}
