// =============================================================================
// Test: AD stamp wall-clock micro-benchmark (Phase 6 of
// add-automatic-differentiation)
// =============================================================================
//
// Times a tight loop of `MOSFET::stamp_jacobian` calls — the build flag
// `PULSIM_USE_AD_STAMP` selects manual or AD; this test measures per-stamp
// wall-clock for whichever path is compiled in. Run the test in both modes
// and compare the printed numbers to compute the AD overhead ratio (Gate
// G.3 of the change: ≤30 % per-stamp overhead).
//
// We use the MOSFET because its templated residual involves both `Real`
// arithmetic and `tanh`/region-selection logic — representative of the
// nonlinear-device cost structure. Loop count is sized so the test
// completes in well under 1 s on the CI baseline machine.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/v1/components/mosfet.hpp"

#include <Eigen/Sparse>
#include <array>
#include <chrono>

using namespace pulsim::v1;

TEST_CASE("AD stamp micro-benchmark on MOSFET in saturation",
          "[ad][perf][benchmark][.disabled-by-default]") {
    // Saturation-region operating point so the residual takes the most
    // expensive branch (kp · vov² · (1 + λ·vds) with full multiplication).
    MOSFET::Params params{};
    params.vth = 2.0;
    params.kp = 0.1;
    params.lambda = 0.01;
    params.is_nmos = true;
    MOSFET m(params, "M_perf");

    Eigen::VectorXd x(3);
    x << 5.0, 10.0, 0.0;          // gate, drain, source

    constexpr int kIterations = 100'000;

    // Warmup pass — JIT/cache effects should not bias the timed loop.
    Eigen::SparseMatrix<Real> J_warm(3, 3);
    Eigen::VectorXd f_warm = Eigen::VectorXd::Zero(3);
    std::array<Index, 3> nodes_warm{0, 1, 2};
    for (int i = 0; i < 1000; ++i) {
        J_warm.setZero();
        f_warm.setZero();
        m.stamp_jacobian(J_warm, f_warm, x, nodes_warm);
    }

    // Timed loop.
    Eigen::SparseMatrix<Real> J(3, 3);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    std::array<Index, 3> nodes{0, 1, 2};

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        J.setZero();
        f.setZero();
        m.stamp_jacobian(J, f, x, nodes);
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double total_seconds = std::chrono::duration<double>(t1 - t0).count();
    const double ns_per_stamp = total_seconds * 1e9 / kIterations;

#ifdef PULSIM_USE_AD_STAMP
    const char* mode = "AD (PULSIM_USE_AD_STAMP=ON)";
#else
    const char* mode = "manual (default)";
#endif

    UNSCOPED_INFO("=== MOSFET stamp micro-benchmark ===");
    UNSCOPED_INFO("Build mode    : " << mode);
    UNSCOPED_INFO("Iterations    : " << kIterations);
    UNSCOPED_INFO("Total wall    : " << total_seconds << " s");
    UNSCOPED_INFO("ns / stamp    : " << ns_per_stamp);

    // Sanity: stamp must produce non-zero Jacobian (otherwise the compiler
    // could elide the loop body).
    CHECK(J.coeff(1, 1) != 0.0);
    // No timing assertion — the comparison ratio between modes is the
    // useful number; we report it via INFO and let the caller compare
    // build runs.
    CHECK(ns_per_stamp > 0.0);
}

TEST_CASE("PWL Ideal mode bypasses AD stamp regardless of build flag",
          "[ad][perf][phase6][bypass]") {
    // In Ideal mode, MOSFET::stamp_jacobian_impl dispatches to
    // stamp_jacobian_ideal which is a pure constant-conductance stamp —
    // no AD partials, no tanh, no derivative-of-conductance term. This
    // contract holds regardless of how the build was configured: the
    // PULSIM_USE_AD_STAMP flag only swaps the *Behavioral* branch.
    MOSFET::Params params{};
    params.vth = 2.0;
    params.kp = 0.1;
    params.is_nmos = true;
    params.g_on = 1e3;
    params.g_off = 1e-12;
    MOSFET m(params, "M_ideal");
    m.set_switching_mode(SwitchingMode::Ideal);
    m.commit_pwl_state(true);

    Eigen::SparseMatrix<Real> J(3, 3);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << 5.0, 8.0, 0.0;
    std::array<Index, 3> nodes{0, 1, 2};

    m.stamp_jacobian(J, f, x, nodes);

    // Pure linear conductance form — drain row carries (gds=g_on, -g_on)
    // on (drain, source); gate row entries are zero (no gm coupling in
    // Ideal mode). This shape rules out the AD path having executed.
    CHECK(J.coeff(1, 1) == params.g_on);
    CHECK(J.coeff(1, 2) == -params.g_on);
    CHECK(J.coeff(1, 0) == 0.0);  // No gate coupling in Ideal mode.
}
