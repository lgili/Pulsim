// =============================================================================
// DSED — consistent reinitialization at mode changes
// =============================================================================
//
// v2.0 Phase 3, item 2. The projection moves fast STABLE modal
// components to their quasi-static values and preserves everything
// else exactly. These tests pin the algebra on systems small enough
// to verify by hand; the end-to-end gate (a DCM buck that used to
// grind 10M steps now runs and matches the pwl engine) lives in
// python/tests/test_dsed_diode_events.py.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/dsed/event_projection.hpp"

using namespace pulsim;
using namespace pulsim::dsed;
using Catch::Approx;

TEST_CASE("Projection: fast mode snaps to quasi-static, slow stays",
          "[v2][dsed][projection]") {
    // Decoupled by construction: A = diag(-1e12, -100).
    DenseMatrix A = DenseMatrix::Zero(2, 2);
    A(0, 0) = -1e12;
    A(1, 1) = -100.0;
    Vector b(2);
    b << 3e12, 5.0;      // quasi-static of mode 0: -b/λ = 3.0

    Vector x(2);
    x << 7.0, 42.0;

    const Vector out =
        project_onto_slow_manifold(A, b, x, /*fast=*/1e7);
    REQUIRE(out[0] == Approx(3.0).margin(1e-9));    // snapped
    REQUIRE(out[1] == Approx(42.0).margin(1e-12));  // untouched
}

TEST_CASE("Projection: paralleled caps equalize conserving charge",
          "[v2][dsed][projection]") {
    // Two caps joined by a conductance g (a switch that just
    // closed): C·dv/dt = ±g·(v_other − v_own). Eigenmodes: total
    // charge (λ = 0, slow) and the difference (λ = −g(1/C1 + 1/C2),
    // fast). The projection must land BOTH voltages on the
    // charge-conserving value (C1·v1 + C2·v2)/(C1 + C2) — the
    // audit's argmin‖Δx‖_M answer, reached here through nothing but
    // the spectral split.
    const Real C1 = 1e-6, C2 = 3e-6, g = 1e3;   // 1 mΩ switch
    DenseMatrix A(2, 2);
    A << -g / C1,  g / C1,
          g / C2, -g / C2;
    const Vector b = Vector::Zero(2);

    Vector x(2);
    x << 10.0, 2.0;                              // unequal at closure

    const Vector out =
        project_onto_slow_manifold(A, b, x, /*fast=*/1e6);
    const Real v_eq = (C1 * 10.0 + C2 * 2.0) / (C1 + C2);  // 4 V
    REQUIRE(out[0] == Approx(v_eq).margin(1e-9));
    REQUIRE(out[1] == Approx(v_eq).margin(1e-9));
    // Charge before == charge after, to machine precision.
    REQUIRE(C1 * out[0] + C2 * out[1]
             == Approx(C1 * 10.0 + C2 * 2.0).margin(1e-15));
}

TEST_CASE("Projection: resonant ringing is physics, not stiffness",
          "[v2][dsed][projection]") {
    // Pure oscillator at 1e9 rad/s: |λ| is enormous but Re(λ) = 0.
    // Projecting it would delete a physical LC ring — the criterion
    // is the REAL part only.
    DenseMatrix A(2, 2);
    A << 0.0, 1e9,
        -1e9, 0.0;
    const Vector b = Vector::Zero(2);
    Vector x(2);
    x << 1.0, -2.0;

    const Vector out =
        project_onto_slow_manifold(A, b, x, /*fast=*/1e6);
    REQUIRE(out[0] == Approx(1.0).margin(1e-12));
    REQUIRE(out[1] == Approx(-2.0).margin(1e-12));
}

TEST_CASE("Projection: no fast mode means exact identity",
          "[v2][dsed][projection]") {
    // The common case — every CCM converter — must cost one
    // eigenvalue scan and change nothing.
    DenseMatrix A(2, 2);
    A << -100.0,  50.0,
          25.0, -200.0;
    Vector b(2);
    b << 1.0, 2.0;
    Vector x(2);
    x << 3.0, 4.0;

    const Vector out =
        project_onto_slow_manifold(A, b, x, /*fast=*/1e7);
    REQUIRE(out[0] == x[0]);
    REQUIRE(out[1] == x[1]);
}

TEST_CASE("Projection: an unstable mode is left alone",
          "[v2][dsed][projection]") {
    // Passive circuits do not produce these, but a wrong projection
    // of one would HIDE a divergence the user needs to see.
    DenseMatrix A = DenseMatrix::Zero(2, 2);
    A(0, 0) = +1e12;
    A(1, 1) = -100.0;
    const Vector b = Vector::Zero(2);
    Vector x(2);
    x << 0.5, 7.0;

    const Vector out =
        project_onto_slow_manifold(A, b, x, /*fast=*/1e7);
    REQUIRE(out[0] == Approx(0.5).margin(1e-12));
    REQUIRE(out[1] == Approx(7.0).margin(1e-12));
}

TEST_CASE("Projection: complex fast pair projects to a real state",
          "[v2][dsed][projection]") {
    // A heavily damped oscillator: λ = −1e9 ± j·1e6. Both members of
    // the conjugate pair classify fast (same Re), so the projected
    // state must come back real and at the pair's quasi-static point
    // (here 0, since b = 0), with the slow mode preserved.
    DenseMatrix A = DenseMatrix::Zero(3, 3);
    A(0, 0) = -1e9;  A(0, 1) =  1e6;
    A(1, 0) = -1e6;  A(1, 1) = -1e9;
    A(2, 2) = -10.0;
    const Vector b = Vector::Zero(3);
    Vector x(3);
    x << 2.0, -1.0, 5.0;

    const Vector out =
        project_onto_slow_manifold(A, b, x, /*fast=*/1e7);
    REQUIRE(out[0] == Approx(0.0).margin(1e-9));
    REQUIRE(out[1] == Approx(0.0).margin(1e-9));
    REQUIRE(out[2] == Approx(5.0).margin(1e-12));
}

// -----------------------------------------------------------------
// v2.0 Phase 3 item 4 — the stiffness detector's power iteration.
// The dispatch decision needs |λ|max to within a few percent of a
// threshold; the full O(n³) eigensolve was costing more than the
// steps it routed.
// -----------------------------------------------------------------

#include "pulsim/dsed/stiffness_detector.hpp"

TEST_CASE("Power iteration finds |λ|max within a few percent",
          "[v2][dsed][stiffness]") {
    // n > 64 forces the power path. Diagonal-dominant construction
    // with a known dominant REAL eigenvalue.
    const int n = 100;
    DenseMatrix A = DenseMatrix::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = -Real{10} * (i + 1);          // λ up to -1000
        if (i + 1 < n) {
            A(i, i + 1) = Real{3};
            A(i + 1, i) = -Real{2};
        }
    }
    dsed::StiffnessDetector det(10.0);
    const Real est = det.lambda_max(/*mode_id=*/0, A);
    // Reference from the dense solver.
    Eigen::EigenSolver<DenseMatrix> es(A);
    Real ref = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        ref = std::max(ref, std::abs(es.eigenvalues()[i]));
    }
    REQUIRE(est == Approx(ref).epsilon(0.05));
}

TEST_CASE("Power iteration handles a dominant COMPLEX pair",
          "[v2][dsed][stiffness]") {
    // Dominant eigenvalue is ±j·1e6 (an LC ring): the single-step
    // norm ratio oscillates; the two-step ratio settles on the
    // modulus. n > 64 to force the power path.
    const int n = 80;
    DenseMatrix A = DenseMatrix::Zero(n, n);
    A(0, 1) = 1e6;
    A(1, 0) = -1e6;
    for (int i = 2; i < n; ++i) {
        A(i, i) = -Real{100} * (i + 1);         // all ≪ 1e6
    }
    dsed::StiffnessDetector det(10.0);
    const Real est = det.lambda_max(/*mode_id=*/1, A);
    REQUIRE(est == Approx(1e6).epsilon(0.05));
}

TEST_CASE("Small systems keep the exact dense eigensolve",
          "[v2][dsed][stiffness]") {
    DenseMatrix A(2, 2);
    A << 0.0, 1e4, -1e4, 0.0;
    dsed::StiffnessDetector det(10.0);
    REQUIRE(det.lambda_max(2, A) == Approx(1e4).margin(1e-6));
}
