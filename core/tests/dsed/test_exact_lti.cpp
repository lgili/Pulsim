// =============================================================================
// DSED — exact LTI stepping
// =============================================================================
//
// v2.0 Phase 3, item 2. The closed form must match the analytic
// solution at ANY h — the entire value proposition is that there is
// no stability region and no truncation error to manage.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/dsed/exact_lti.hpp"

#include <cmath>

using namespace pulsim;
using namespace pulsim::dsed;
using Catch::Approx;

TEST_CASE("Exact: RC decay at a tenth, ten, and a thousand taus",
          "[v2][dsed][exact_lti]") {
    // dx/dt = -(x - V)/tau, analytic x(h) = V + (x0 - V)·e^{-h/tau}.
    const Real tau = 1e-6, V = 5.0, x0 = 1.0;
    DenseMatrix A(1, 1);
    A(0, 0) = -1.0 / tau;
    Vector b(1);
    b << V / tau;
    Vector x(1);
    x << x0;

    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    for (const Real h : {0.1 * tau, 10.0 * tau, 1000.0 * tau}) {
        const Vector out = exact_advance(e, x, b, h);
        const Real ref = V + (x0 - V) * std::exp(-h / tau);
        INFO("h/tau = " << h / tau);
        REQUIRE(out[0] == Approx(ref).epsilon(1e-12).margin(1e-12));
    }
}

TEST_CASE("Exact: a pure integrator mode integrates exactly",
          "[v2][dsed][exact_lti]") {
    // lambda = 0 — the phi1 series branch. x(h) = x0 + b·h, exactly,
    // where the naive (e^s - 1)/s form would divide zero by zero.
    DenseMatrix A = DenseMatrix::Zero(1, 1);
    Vector b(1);
    b << 3.0;
    Vector x(1);
    x << 7.0;

    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    const Vector out = exact_advance(e, x, b, 2.5);
    REQUIRE(out[0] == Approx(7.0 + 3.0 * 2.5).margin(1e-12));
}

TEST_CASE("Exact: a 5e9-rad/s mode crossed in ONE 10 µs step",
          "[v2][dsed][exact_lti]") {
    // THE case that kills explicit integrators: lambda = -5e9 (the
    // DCM idle mode's L·g_off). DOPRI5's stability region pins
    // h below ~6.6e-10 s forever; the exact step takes h = 1e-5 —
    // fifteen thousand stability limits at once — and lands exactly
    // on the equilibrium.
    DenseMatrix A(1, 1);
    A(0, 0) = -5e9;
    Vector b(1);
    b << 5e9 * 0.25;      // equilibrium at -b/lambda = 0.25
    Vector x(1);
    x << 100.0;

    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    const Vector out = exact_advance(e, x, b, 1e-5);
    REQUIRE(out[0] == Approx(0.25).margin(1e-12));
}

TEST_CASE("Exact: an LC ring keeps amplitude and phase",
          "[v2][dsed][exact_lti]") {
    // Undamped rotation at omega = 1e7: after an arbitrary h the
    // state must be the initial vector rotated by omega·h, energy
    // preserved to machine precision — where an explicit method
    // spirals out and an implicit one spirals in.
    const Real w = 1e7;
    DenseMatrix A(2, 2);
    A << 0.0,   w,
         -w,  0.0;
    const Vector b = Vector::Zero(2);
    Vector x(2);
    x << 1.0, 0.0;

    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    const Real h = 3.7e-6;                 // ~5.9 full turns
    const Vector out = exact_advance(e, x, b, h);
    REQUIRE(out[0] == Approx(std::cos(w * h)).margin(1e-9));
    REQUIRE(out[1] == Approx(-std::sin(w * h)).margin(1e-9));
    REQUIRE(out.squaredNorm() == Approx(1.0).margin(1e-12));
}

TEST_CASE("Exact: h = 0 is the identity",
          "[v2][dsed][exact_lti]") {
    DenseMatrix A(2, 2);
    A << -100.0, 20.0, 5.0, -300.0;
    Vector b(2);
    b << 1.0, -2.0;
    Vector x(2);
    x << 4.0, 9.0;
    const auto e = make_exact_lti(A);
    REQUIRE(e.valid);
    const Vector out = exact_advance(e, x, b, 0.0);
    REQUIRE(out[0] == Approx(4.0).margin(1e-13));
    REQUIRE(out[1] == Approx(9.0).margin(1e-13));
}

// -----------------------------------------------------------------
// v2.0 Phase 3 item 3 — the SINE-augmented stepper, end to end
// through the builder adapter. A series RC driven by a sine has a
// closed-form response; one exact step across many time constants
// must land on it, DC offset and phase included.
// -----------------------------------------------------------------

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/dsed/builder_adapter.hpp"
#include "pulsim/pwl/cache.hpp"

TEST_CASE("Exact: sine-driven RC steps to the analytic response",
          "[v2][dsed][exact_lti][augmented]") {
    const Real R = 1e3, C = 1e-6;                 // tau = 1 ms
    const Real A_amp = 5.0, f = 60.0, phase = 0.7, v_dc = 2.0;
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vac", "in", "gnd",
                                v_dc, A_amp, f, phase);
    b.add_resistor("R1", "in", "n1", R);
    b.add_capacitor("C1", "n1", "gnd", C);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(),
                                             cache};
    topology::SwitchStateMask m(0);
    ad.set_mask(m);
    REQUIRE(ad.has_exact_step());       // sine-only drive qualifies

    // Analytic v_C for v_C(0) = 0 under
    // V(t) = v_dc + A·sin(wt + phase):
    //   steady state: v_dc + Vp·sin(wt + phase − theta),
    //   Vp = A/sqrt(1+(wRC)^2), theta = atan(wRC);
    //   transient sized to cancel it at t = 0.
    const Real w = 2.0 * std::numbers::pi_v<Real> * f;
    const Real tau = R * C;
    const Real theta = std::atan(w * tau);
    const Real Vp = A_amp / std::sqrt(1.0 + w * tau * w * tau);
    auto v_ref = [&](Real t) {
        const Real ss = v_dc + Vp * std::sin(w * t + phase - theta);
        const Real ss0 = v_dc + Vp * std::sin(phase - theta);
        return ss - ss0 * std::exp(-t / tau);
    };

    Vector x0(1);
    x0 << 0.0;
    // ONE step of 5 ms (five taus, a third of a mains cycle), then
    // a second from there — exactness must compose across steps.
    const Vector x1 = ad.exact_advance_state(0.0, x0, 5e-3);
    REQUIRE(x1[0] == Approx(v_ref(5e-3)).margin(1e-9));
    const Vector x2 = ad.exact_advance_state(5e-3, x1, 7e-3);
    REQUIRE(x2[0] == Approx(v_ref(12e-3)).margin(1e-9));
}

TEST_CASE("Exact: PWM or pulse sources decline the exact path",
          "[v2][dsed][exact_lti][augmented]") {
    // b(t) with steps is not a finite oscillator sum — the numeric
    // path owns it, and claiming otherwise would be a wrong answer,
    // not a slow one.
    builder::CircuitBuilder b;
    b.add_pwm_voltage_source("Vg", "in", "gnd", 5.0, 0.0, 1e5, 0.5);
    b.add_resistor("R1", "in", "n1", 1e3);
    b.add_capacitor("C1", "n1", "gnd", 1e-6);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    dsed::NativeCircuitBuilderAdapter<> ad{b.graph(), b.pool(),
                                             cache};
    topology::SwitchStateMask m(0);
    ad.set_mask(m);
    REQUIRE_FALSE(ad.has_exact_step());
}
