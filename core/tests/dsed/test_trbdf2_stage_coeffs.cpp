// =============================================================================
// TR-BDF2 second-stage coefficients — the exact order conditions
// =============================================================================
//
// A wrong c2/c3 is invisible to almost every other test. The
// CONDUCTANCE block is untouched by it (c1 alone sets that), so the
// matrix keeps its sparsity, Newton keeps converging quadratically,
// and the run produces a smooth, plausible waveform — it simply
// converges to a DIFFERENT LIMIT. A convergence study can catch that,
// but only noisily and slowly, and only where a fine reference
// exists.
//
// These identities catch it exactly and instantly. The stage-2
// formula
//
//     dX/dt|_{n+1}  ~  (c1*X_{n+1} + c2*X_gamma + c3*X_n) / h
//
// applied on [t_n, t_n + h] with the intermediate point at
// t_n + gamma*h must be EXACT for X = 1, X = t and X = t^2 — three
// equations that pin the three coefficients uniquely. The fourth
// identity, c1 == 2/gamma, is the one the whole composite method
// rests on: it is what makes the stage-2 derivative coefficient
// c1/h equal the trapezoidal stage's 2/(gamma*h), so ONE matrix
// factor serves both stages.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/trbdf2_stage.hpp"

#include <cmath>

using namespace pulsim;
using Catch::Approx;

TEST_CASE("TR-BDF2 stage-2 coefficients satisfy the exact order "
          "conditions", "[v2][trbdf2][coeffs]") {
    const auto k = pulsim::pwl::trbdf2_coeffs();
    const Real g = k.gamma;

    // Order 0 — a constant state has no derivative. This is the one
    // the header asserts at runtime; pinned here too so the runtime
    // check itself cannot rot.
    CHECK(k.c1 + k.c2 + k.c3 == Approx(0.0).margin(1e-14));
    CHECK(pulsim::pwl::trbdf2_coeffs_consistent());

    // Order 1 — X = t, with t_n = 0 and h = 1 (the conditions are
    // scale-free, so h drops out): X_n = 0, X_gamma = gamma,
    // X_{n+1} = 1, and the exact derivative is 1.
    CHECK(k.c1 * 1.0 + k.c2 * g + k.c3 * 0.0 == Approx(1.0).epsilon(1e-13));

    // Order 2 — X = t^2: X_gamma = gamma^2, X_{n+1} = 1, and the
    // exact derivative at t_{n+1} = 1 is 2.
    CHECK(k.c1 * 1.0 + k.c2 * g * g + k.c3 * 0.0
          == Approx(2.0).epsilon(1e-13));

    // The shared-factor identity. Break this and the two stages need
    // two different factorizations — the method still converges, it
    // just costs twice as much and silently stops being TR-BDF2.
    CHECK(k.c1 == Approx(2.0 / g).epsilon(1e-13));

    // gamma = 2 - sqrt(2) is what makes the identity above hold; if
    // a future edit "simplifies" it to 1/2 or 1/3, the check above
    // fires, but pin the value too so the failure names its cause.
    CHECK(g == Approx(2.0 - std::sqrt(2.0)).epsilon(1e-15));
}

TEST_CASE("TR-BDF2 stage-2 formula reproduces a quadratic exactly at "
          "an arbitrary h", "[v2][trbdf2][coeffs]") {
    // The conditions above are stated at h = 1. Re-run them at a
    // physically-sized step on a quadratic with all three terms
    // present, which is what the device stamps actually evaluate.
    const auto k = pulsim::pwl::trbdf2_coeffs();
    const Real h = 3.7e-7;
    const Real t_n = 1.25e-3;

    // X(t) = a + b*t + c*t^2, dX/dt = b + 2*c*t.
    const Real a = 0.4, b = -21.0, c = 1.3e5;
    auto X = [&](Real t) { return a + b * t + c * t * t; };

    const Real approx = (k.c1 * X(t_n + h)
                         + k.c2 * X(t_n + k.gamma * h)
                         + k.c3 * X(t_n)) / h;
    const Real exact = b + 2.0 * c * (t_n + h);
    CHECK(approx == Approx(exact).epsilon(1e-9));

    // And it is NOT accidentally exact for a cubic — that would mean
    // the coefficients had drifted onto some other method.
    auto Y = [&](Real t) { return t * t * t; };
    const Real approx3 = (k.c1 * Y(t_n + h)
                          + k.c2 * Y(t_n + k.gamma * h)
                          + k.c3 * Y(t_n)) / h;
    const Real exact3 = 3.0 * (t_n + h) * (t_n + h);
    CHECK(approx3 != Approx(exact3).epsilon(1e-14));
}
