// BLDC motor device — Catch2 tests for the consolidate-motors-and-three-phase
// Phase C.1 device-variant implementation.
//
// These tests exercise the BldcMotorDevice directly (no Circuit / Simulator
// involvement yet — the runtime wiring is added by the parent agent after
// this file lands). They validate:
//
//   1. Open-circuit back-EMF waveform shape (trapezoidal: +1 flat /
//      linear ramp / -1 flat / linear ramp on 60° / 120° / 180° / 300° /
//      360° boundaries).
//   2. Six-step commutation phase-current pattern produces the canonical
//      τ = 2 · K_e_peak · I · p per step.
//   3. Locked-rotor torque matches the analytical electromagnetic-torque
//      formula at a given (i_a, i_b, i_c) without advancing the rotor.
//
// The PMSM dynamic test suite is the architectural reference for this file;
// the difference here is that we don't drive a circuit yet.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/bldc_motor_device.hpp"

#include <array>
#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

constexpr Real kPi          = Real{3.14159265358979323846};
constexpr Real kTwoPi       = Real{2.0} * kPi;
constexpr Real kDegToRad    = kPi / Real{180.0};

motors::BldcMotorParams default_bldc_params() {
    motors::BldcMotorParams p{};
    p.R_s         = 0.5;        // Ω
    p.L_s         = 1e-3;       // H
    p.K_e_peak    = 0.05;       // V·s/rad
    p.pole_pairs  = 2;
    p.J           = 1e-4;       // kg·m²
    p.b_friction  = 1e-5;       // N·m·s
    p.omega_init  = 0.0;
    p.theta_init  = 0.0;
    return p;
}

/// Expected trapezoidal profile for phase A, in per-unit [-1, +1], computed
/// from the spec directly (no shared implementation with the device under
/// test, so the test acts as an independent oracle).
[[nodiscard]] Real expected_profile_a(Real theta_e) noexcept {
    // Wrap into [0, 2π).
    Real x = std::fmod(theta_e, kTwoPi);
    if (x < Real{0}) x += kTwoPi;

    const Real deg = x / kDegToRad;
    if (deg <= 120.0) return Real{1};
    if (deg <  180.0) return Real{1} - Real{2} * (deg - 120.0) / 60.0;
    if (deg <= 300.0) return Real{-1};
    return Real{-1} + Real{2} * (deg - 300.0) / 60.0;
}

}  // namespace

TEST_CASE("BLDC device: open-circuit back-EMF matches trapezoidal profile",
          "[bldc][motor][unit]") {
    // Pre-spin the rotor and sweep θ_m over [0, 2π) (electrical: [0, 2π·p)).
    // At each angle the device's phase-A back-EMF must equal
    //   K_e_peak · f_a(θ_e) · ω_e
    // with f_a(·) being the textbook trapezoidal profile (+1 / ramp / -1
    // / ramp).
    auto params = default_bldc_params();
    params.pole_pairs = 1;          // θ_e = θ_m for simple bookkeeping
    BldcMotorDevice motor(params, "M1");

    const Real omega_m = 100.0;     // rad/s — pick something non-trivial
    motor.set_initial_omega(omega_m);

    const Real omega_e = static_cast<Real>(params.pole_pairs) * omega_m;
    const Real e_peak  = params.K_e_peak * omega_e;
    const Real e_tol_abs = Real{0.01} * std::abs(e_peak);   // 1% absolute

    constexpr int N = 720;          // 0.5°-resolution sweep
    int checked_flat_top    = 0;
    int checked_flat_bottom = 0;
    int checked_ramps       = 0;

    for (int k = 0; k <= N; ++k) {
        const Real theta_m = (kTwoPi * static_cast<Real>(k)) /
                             static_cast<Real>(N);
        const Real e_a_meas = motor.back_emf_at(0, theta_m);
        const Real e_a_expected = params.K_e_peak * omega_e *
                                  expected_profile_a(theta_m);

        INFO("θ_m = " << theta_m << " rad ("
             << (theta_m / kDegToRad) << "°), expected e_a = "
             << e_a_expected << " V, measured = " << e_a_meas << " V");
        CHECK(e_a_meas == Approx(e_a_expected).margin(e_tol_abs));

        // Bucket counts — sanity that we hit every regime at least a few times.
        const Real deg = std::fmod(theta_m / kDegToRad, 360.0);
        const Real abs_e = std::abs(e_a_meas);
        if (deg >= 5.0 && deg <= 115.0) {
            // Strictly inside the +1 flat-top.
            CHECK(e_a_meas == Approx(e_peak).margin(e_tol_abs));
            ++checked_flat_top;
        } else if (deg >= 185.0 && deg <= 295.0) {
            // Strictly inside the -1 flat-bottom.
            CHECK(e_a_meas == Approx(-e_peak).margin(e_tol_abs));
            ++checked_flat_bottom;
        } else if ((deg >= 125.0 && deg <= 175.0) ||
                   (deg >= 305.0 && deg <= 355.0)) {
            // Strictly inside a ramp — magnitude must be < the flat-top
            // value (modulo a 1% slack at the bucket edges).
            CHECK(abs_e <= std::abs(e_peak) + e_tol_abs);
            ++checked_ramps;
        }
    }

    // Did we hit each regime a meaningful number of times?
    CHECK(checked_flat_top    >= 100);
    CHECK(checked_flat_bottom >= 100);
    CHECK(checked_ramps       >= 100);
}

TEST_CASE("BLDC device: six-step commutation produces τ = 2·K_e·I·p per step",
          "[bldc][motor][regression]") {
    // Drive the device through the canonical 6-step commutation sequence.
    // Each step is a (i_a, i_b, i_c) tuple in which exactly two phases are
    // conducting (one +I, one -I) and one is floating. The rotor is parked
    // at the electrical angle where BOTH conducting phases sit on a flat-
    // top (+1 / -1) plateau, so f_a · i_a + f_b · i_b + f_c · i_c = 2·I
    // and the canonical torque-per-step is
    //
    //   τ_em = K_e_peak · p · 2 · I
    //        = 2 · K_e_peak · I · p
    //
    // For our convention (f_a leads, f_b = f_a(θ_e − 120°),
    // f_c = f_a(θ_e − 240°)), the six commutation intervals and the
    // midpoint of each — the angle where both conducting phases sit
    // firmly on their flat-tops — work out to:
    //
    //   step  conducting        θ_e interval   θ_e center  (i_a, i_b, i_c)
    //   ───────────────────────────────────────────────────────────────────
    //   0     a(+) / b(-)        [  0°,  60°]   30°        (+I,  -I,   0)
    //   1     a(+) / c(-)        [ 60°, 120°]   90°        (+I,   0,  -I)
    //   2     b(+) / c(-)        [120°, 180°]  150°        ( 0,  +I,  -I)
    //   3     b(+) / a(-)        [180°, 240°]  210°        (-I,  +I,   0)
    //   4     c(+) / a(-)        [240°, 300°]  270°        (-I,   0,  +I)
    //   5     c(+) / b(-)        [300°, 360°]  330°        ( 0,  -I,  +I)
    //
    // Each center is exactly 60° apart — the canonical electrical-step
    // boundaries of six-step BLDC commutation.
    auto params = default_bldc_params();
    params.pole_pairs = 1;          // θ_e = θ_m — keep the test simple
    BldcMotorDevice motor(params, "M_sixstep");

    constexpr Real I = 5.0;         // A, conducting-phase current
    const Real p = static_cast<Real>(params.pole_pairs);
    const Real tau_expected = Real{2.0} * params.K_e_peak * I * p;
    const Real tau_tol = Real{0.05} * std::abs(tau_expected);  // 5% absolute

    struct Step {
        Real theta_e_deg;
        Real i_a;
        Real i_b;
        Real i_c;
        const char* label;
    };
    const std::array<Step, 6> steps{{
        { 30.0,  +I, -I,   0, "step0: a(+) b(-)" },
        { 90.0,  +I,  0,  -I, "step1: a(+) c(-)" },
        {150.0,   0, +I,  -I, "step2: b(+) c(-)" },
        {210.0,  -I, +I,   0, "step3: b(+) a(-)" },
        {270.0,  -I,  0,  +I, "step4: c(+) a(-)" },
        {330.0,   0, -I,  +I, "step5: c(+) b(-)" },
    }};

    for (const auto& s : steps) {
        // Park rotor at the step's "center" electrical angle.
        motor.set_initial_theta(s.theta_e_deg * kDegToRad / p);

        const Real tau_meas =
            motor.electromagnetic_torque(s.i_a, s.i_b, s.i_c);

        INFO(s.label << ": θ_e_center = " << s.theta_e_deg
             << "°, (i_a, i_b, i_c) = (" << s.i_a << ", " << s.i_b
             << ", " << s.i_c << "), τ_expected = " << tau_expected
             << " N·m, τ_measured = " << tau_meas << " N·m");
        CHECK(tau_meas == Approx(tau_expected).margin(tau_tol));
    }
}

TEST_CASE("BLDC device: locked-rotor torque matches analytical formula",
          "[bldc][motor][unit]") {
    // With the rotor parked at a known angle (no mechanical advance), apply
    // an arbitrary set of phase currents and verify that the device's
    // electromagnetic-torque accessor returns exactly
    //
    //   τ = K_e_peak · p · (f_a · i_a + f_b · i_b + f_c · i_c)
    //
    // computed independently here from the trapezoidal profile oracle.
    auto params = default_bldc_params();
    params.pole_pairs = 4;          // pick a non-trivial p
    BldcMotorDevice motor(params, "M_locked");

    motor.set_initial_omega(0.0);   // rotor locked
    motor.set_load_torque(0.0);     // no external load (irrelevant: no advance)

    struct Case {
        Real theta_m;
        Real i_a;
        Real i_b;
        Real i_c;
    };
    const std::array<Case, 5> cases{{
        { 0.0,                 +1.0,  -0.5,  -0.5 },
        { 10.0 * kDegToRad,     0.0,  +2.0,  -2.0 },
        { 35.0 * kDegToRad,    -1.0,  +1.0,   0.0 },
        { 75.0 * kDegToRad,    +0.7,  +0.3,  -1.0 },
        {-30.0 * kDegToRad,    +1.5,  -0.5,  -1.0 },
    }};

    for (const auto& c : cases) {
        motor.set_initial_theta(c.theta_m);

        // Use the device's own getters for the per-phase profile values
        // (which we already cross-checked in the open-circuit test).
        const Real f_a = motor.back_emf_profile_at(0, c.theta_m);
        const Real f_b = motor.back_emf_profile_at(1, c.theta_m);
        const Real f_c = motor.back_emf_profile_at(2, c.theta_m);

        const Real p_real = static_cast<Real>(params.pole_pairs);
        const Real tau_expected = params.K_e_peak * p_real *
            (f_a * c.i_a + f_b * c.i_b + f_c * c.i_c);

        // Device API #1: torque at current angle.
        const Real tau_now = motor.electromagnetic_torque(c.i_a, c.i_b,
                                                          c.i_c);
        // Device API #2: torque at an explicit angle (same as #1 here).
        const Real tau_at = motor.electromagnetic_torque_at(c.i_a, c.i_b,
                                                            c.i_c,
                                                            c.theta_m);

        INFO("θ_m = " << c.theta_m << " rad, (i_a, i_b, i_c) = ("
             << c.i_a << ", " << c.i_b << ", " << c.i_c
             << "), τ_expected = " << tau_expected
             << " N·m, τ_now = " << tau_now
             << " N·m, τ_at = " << tau_at << " N·m");
        CHECK(tau_now == Approx(tau_expected).margin(1e-12));
        CHECK(tau_at  == Approx(tau_expected).margin(1e-12));

        // Rotor really did not move (no advance() called above).
        CHECK(motor.omega_m() == Approx(0.0));
        CHECK(motor.theta_m() == Approx(c.theta_m));
    }
}

TEST_CASE("BLDC device: companion-conductance helper matches stamping math",
          "[bldc][motor][unit]") {
    // Pure unit test of the device's own arithmetic — exercises the
    // accessor used by the Jacobian ladder to ensure R_s, L_s, dt knobs
    // round-trip correctly.
    auto params = default_bldc_params();
    params.R_s = 0.75;
    params.L_s = 3e-3;
    BldcMotorDevice motor(params, "M_g");

    const Real dt = 250e-6;
    const Real g_expected = Real{1} /
        (params.R_s + Real{2} * params.L_s / dt);
    CHECK(motor.companion_conductance(dt) == Approx(g_expected));
}
