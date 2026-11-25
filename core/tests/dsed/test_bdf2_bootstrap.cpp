// =============================================================================
// fix-dsed-auto-integrator-selection — the BDF2 start is backward Euler
// =============================================================================
//
// The first BDF2 step used to be a Crank–Nicolson bootstrap, which is
// not L-stable: on y' = (100 − y)/τ from y = 0 with h/τ = 100 it gave
// y₁ = 100·2z/(1 + z), z = 50: 196.08 for a 100 V target. Backward
// Euler gives 100·(h/τ)/(1 + h/τ).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/dsed/bdf2_integrator.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <cmath>

using pulsim::Real;
using pulsim::Vector;
using pulsim::DenseMatrix;
using namespace pulsim::dsed;

namespace {

/// y' = (100 − y)/τ as A = [−1/τ], b = [100/τ].
struct Rc {
    Real tau;
    DenseMatrix A() const { DenseMatrix a(1, 1); a(0, 0) = -1.0 / tau; return a; }
    Vector b() const { Vector v(1); v(0) = 100.0 / tau; return v; }
};

Real run_to(const Rc& rc, Real h, int n_steps, Real* y1 = nullptr, Real* y2 = nullptr) {
    const DenseMatrix A = rc.A();
    const Vector b = rc.b();
    auto b_fn = [&b](Real) -> Vector { return b; };
    BDF2State st;
    Vector y = Vector::Zero(1);
    for (int k = 0; k < n_steps; ++k) {
        auto [y_new, err] = bdf2_step(A, b_fn, k * h, y, h, st);
        y = y_new;
        if (k == 0 && y1) *y1 = y(0);
        if (k == 1 && y2) *y2 = y(0);
    }
    return y(0);
}

}  // namespace

TEST_CASE("BDF2 bootstrap is backward Euler (L-stable start)",
          "[dsed][bdf2][fix-dsed-auto-integrator-selection]") {
    Real y1 = 0, y2 = 0;
    (void)run_to(Rc{1.0}, 100.0, 2, &y1, &y2);    // h/tau = 100
    // y1 = 100*100/101; y2 from (1 + 200/3) y2 = 4/3 y1 + 2/3*100*100.
    const Real y1_want = 100.0 * 100.0 / 101.0;                      // 99.0099
    const Real y2_want = (4.0 / 3.0 * y1_want + 2.0 / 3.0 * 100.0 * 100.0)
                       / (1.0 + 2.0 / 3.0 * 100.0);                  // 100.4731
    REQUIRE(y1 == Catch::Approx(y1_want).epsilon(1e-9));
    REQUIRE(y2 == Catch::Approx(y2_want).epsilon(1e-9));
    REQUIRE(y1 == Catch::Approx(99.0099).epsilon(1e-6));
    REQUIRE(y2 == Catch::Approx(100.4731).epsilon(1e-6));

    (void)run_to(Rc{1.0}, 10.0, 1, &y1);           // h/tau = 10
    REQUIRE(y1 == Catch::Approx(100.0 * 10.0 / 11.0).epsilon(1e-9));  // 90.9091
}

TEST_CASE("BDF2 with the backward-Euler start is still second order",
          "[dsed][bdf2][fix-dsed-auto-integrator-selection]") {
    // Non-stiff: h/tau = 0.01, integrate to t = 1 (tau = 1).
    const Rc rc{1.0};
    const Real exact = 100.0 * (1.0 - std::exp(-1.0));
    const Real e_h  = std::abs(run_to(rc, 0.01, 100) - exact);
    const Real e_h2 = std::abs(run_to(rc, 0.005, 200) - exact);
    INFO("e(h) = " << e_h << "  e(h/2) = " << e_h2);
    REQUIRE(e_h / e_h2 == Catch::Approx(4.0).epsilon(0.10));
}
