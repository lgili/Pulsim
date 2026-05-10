// =============================================================================
// add-motor-models — Phase 1 / 2 / 3 / 5.2 / 7 / 9 tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/motors/dc_motor.hpp"
#include "pulsim/v1/motors/frame_transforms.hpp"
#include "pulsim/v1/motors/mechanical.hpp"
#include "pulsim/v1/motors/pmsm.hpp"
#include "pulsim/v1/motors/pmsm_foc.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim::v1;
using namespace pulsim::v1::motors;
using Catch::Approx;

// -----------------------------------------------------------------------------
// Phase 1: mechanical primitives
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1: shaft + flywheel + step torque first-order response",
          "[v1][motors][phase1][shaft]") {
    Shaft shaft;
    shaft.J = 1e-3;
    shaft.b_friction = 1e-3;
    shaft.omega = 0.0;

    // Apply 0.5 N·m for many small steps; ω_ss = τ/b = 500 rad/s. The
    // mechanical time constant is τ_m = J/b = 1 s, so we run 5·τ_m to
    // settle within ≤ 1 % of the asymptote.
    constexpr Real tau = 0.5;
    constexpr Real dt  = 1e-4;
    constexpr int N    = 50000;        // 5.0 s = 5·τ_m
    for (int i = 0; i < N; ++i) {
        shaft.advance(tau, dt);
    }
    const Real omega_ss_expected = tau / shaft.b_friction;
    CHECK(shaft.omega == Approx(omega_ss_expected).epsilon(0.02));
}

TEST_CASE("Phase 1: gearbox reflects torque/speed correctly",
          "[v1][motors][phase1][gearbox]") {
    GearBox gb{.ratio = 10.0, .efficiency = 0.95};
    CHECK(gb.omega_out(100.0) == Approx(10.0));
    CHECK(gb.torque_out(0.5) == Approx(0.5 * 10.0 * 0.95));
    CHECK(gb.reflect_load(50.0) == Approx(50.0 / (10.0 * 0.95)));
}

TEST_CASE("Phase 1: fan load is proportional to ω·|ω|",
          "[v1][motors][phase1][fan_load]") {
    FanLoad load{.k = 1e-4};
    CHECK(load.load_torque(100.0)  == Approx(1e-4 * 100.0 * 100.0));
    CHECK(load.load_torque(-100.0) == Approx(-1e-4 * 100.0 * 100.0));
}

// -----------------------------------------------------------------------------
// Phase 2: Park / Clarke
// -----------------------------------------------------------------------------

TEST_CASE("Phase 2: Clarke + inverse round-trips identity",
          "[v1][motors][phase2][clarke]") {
    constexpr Real a = 1.0;
    constexpr Real b = -0.5;
    constexpr Real c = -0.5;
    const auto [alpha, beta] = clarke(a, b, c);
    const auto [a2, b2, c2] = inverse_clarke(alpha, beta);
    // Note: amplitude-invariant Clarke is *not* invertible 1:1 on
    // a 3-phase signal with a non-zero zero-sequence component.
    // For balanced (a + b + c = 0) the inverse returns the original.
    CHECK(a2 == Approx(a).margin(1e-12));
    CHECK(b2 == Approx(b).margin(1e-12));
    CHECK(c2 == Approx(c).margin(1e-12));
}

TEST_CASE("Phase 2: Park + inverse Park round-trips identity",
          "[v1][motors][phase2][park]") {
    constexpr Real alpha = 1.5;
    constexpr Real beta  = 0.2;
    constexpr Real theta = 1.234;     // arbitrary angle
    const auto [d, q]  = park(alpha, beta, theta);
    const auto [a2, b2] = inverse_park(d, q, theta);
    CHECK(a2 == Approx(alpha).margin(1e-12));
    CHECK(b2 == Approx(beta).margin(1e-12));
}

TEST_CASE("Phase 2: balanced 3φ at θ_e produces zero in d when aligned with α",
          "[v1][motors][phase2][park][physics]") {
    // For a balanced sinusoidal three-phase signal at angle θ:
    //   a = cos(θ), b = cos(θ - 2π/3), c = cos(θ + 2π/3)
    // After Clarke + Park at the same θ, we get d = 1, q = 0 (the
    // textbook synchronously-rotating frame snapshot).
    const Real theta = 0.85;
    const Real two_pi_3 = Real{2} * std::numbers::pi_v<Real> / Real{3};
    const Real a = std::cos(theta);
    const Real b = std::cos(theta - two_pi_3);
    const Real c = std::cos(theta + two_pi_3);
    const auto [d, q] = abc_to_dq(a, b, c, theta);
    CHECK(d == Approx(1.0).margin(1e-12));
    CHECK(q == Approx(0.0).margin(1e-12));
}

// -----------------------------------------------------------------------------
// Phase 3: PMSM no-load + locked-rotor + steady state
// -----------------------------------------------------------------------------

TEST_CASE("Phase 3.4: PMSM no-load back-EMF matches ψ_PM · ω_e",
          "[v1][motors][phase3][pmsm][no_load][gate_G4]") {
    PmsmParams p;
    p.Rs = 0.5;
    p.Ld = 5e-3;
    p.Lq = 5e-3;
    p.psi_pm = 0.1;
    p.pole_pairs = 4;
    p.J = 1e-3;
    p.b_friction = 0.0;
    p.omega_init = 100.0;             // 100 rad/s mechanical

    Pmsm motor(p);
    // No load, no excitation — back-EMF = ψ_PM · p · ω_m
    const Real expected = p.psi_pm * p.pole_pairs * p.omega_init;
    CHECK(motor.back_emf_peak() == Approx(expected).margin(1e-12));
}

TEST_CASE("Phase 3.5: PMSM locked-rotor i_q step matches Vq / Rs at steady state",
          "[v1][motors][phase3][pmsm][locked_rotor][gate_G4]") {
    // With ω_m forced to 0 (locked) and starting from rest, applying
    // V_q drives i_q toward V_q / R_s on the L_q time constant.
    PmsmParams p;
    p.Rs = 1.0;
    p.Ld = 5e-3;
    p.Lq = 5e-3;
    p.psi_pm = 0.05;
    p.pole_pairs = 1;
    p.J = 1e9;                          // huge inertia → effectively locked
    p.b_friction = 0.0;

    Pmsm motor(p);
    constexpr Real Vq = 5.0;
    constexpr Real dt = 1e-5;
    constexpr int N   = 50000;          // 0.5 s — many τ_e
    for (int i = 0; i < N; ++i) {
        motor.step(/*Vd*/0.0, Vq, /*tau_load*/0.0, dt);
    }
    const Real iq_ss = Vq / p.Rs;
    CHECK(motor.i_q() == Approx(iq_ss).epsilon(0.01));
    CHECK(std::abs(motor.omega_m()) < 1e-3);   // truly locked
}

// -----------------------------------------------------------------------------
// Phase 5.2: DC motor steady-state speed step
// -----------------------------------------------------------------------------

TEST_CASE("Phase 5.2: DC motor speed step matches first-order analytical (≤ 5 %)",
          "[v1][motors][phase5][dc_motor][gate_G1]") {
    DcMotorParams p;
    p.R_a = 1.0;
    p.L_a = 1e-3;
    p.K_e = 0.05;
    p.K_t = 0.05;
    p.J   = 1e-4;
    p.b   = 1e-5;

    DcMotor m(p);
    constexpr Real Va = 12.0;
    constexpr Real dt = 1e-5;
    // Run to steady state; mechanical τ_m ≈ J·R_a / (K_t·K_e)
    //   = 1e-4 · 1 / (0.05 · 0.05) = 1e-4 / 2.5e-3 = 0.04 s.
    constexpr int N = 50000;            // 0.5 s ≈ 12·τ_m
    for (int i = 0; i < N; ++i) {
        m.step(Va, /*tau_load*/0.0, dt);
    }

    const Real omega_ss_expected = m.steady_state_omega(Va, /*tau_load*/0.0);
    CHECK(m.omega() == Approx(omega_ss_expected).epsilon(0.05));
    CHECK(m.mechanical_time_constant() ==
          Approx(p.J * p.R_a / (p.K_t * p.K_e)).margin(1e-12));
}

// -----------------------------------------------------------------------------
// Phase 7: PMSM-FOC current loop tracks references
// -----------------------------------------------------------------------------

TEST_CASE("Phase 7: PMSM-FOC current loop tracks i_q_ref step within bandwidth",
          "[v1][motors][phase7][foc][gate_G3]") {
    PmsmParams p;
    p.Rs = 0.5;
    p.Ld = 2e-3;
    p.Lq = 2e-3;
    p.psi_pm = 0.05;
    p.pole_pairs = 4;
    p.J = 1e9;                          // locked rotor (no torque buildup)
    p.omega_init = 0.0;

    Pmsm motor(p);
    PmsmFocCurrentLoop foc(p, {.bandwidth_hz = 500.0,
                                 .Vd_min = -50.0, .Vd_max = 50.0,
                                 .Vq_min = -50.0, .Vq_max = 50.0});

    constexpr Real iq_ref = 5.0;
    constexpr Real id_ref = 0.0;
    constexpr Real dt     = 1e-5;       // 100 kHz control loop
    constexpr int N       = 5000;        // 50 ms — many bandwidth periods

    for (int i = 0; i < N; ++i) {
        const auto [Vd, Vq] = foc.step(id_ref, iq_ref, motor.i_d(), motor.i_q(), dt);
        motor.step(Vd, Vq, /*tau_load*/0.0, dt);
    }

    // After settling, both currents should track within 5 % of refs.
    CHECK(motor.i_q() == Approx(iq_ref).epsilon(0.05));
    CHECK(motor.i_d() == Approx(id_ref).margin(0.5));
}

TEST_CASE("Phase 7: FOC retune updates PI gains to match the new bandwidth",
          "[v1][motors][phase7][foc][retune]") {
    PmsmParams p;
    p.Rs = 0.5;
    p.Ld = 2e-3;
    p.Lq = 2e-3;
    p.psi_pm = 0.05;

    PmsmFocCurrentLoop foc(p, {.bandwidth_hz = 500.0});
    const Real kp_500 = foc.pi_d().params().kp;

    foc.retune(p, {.bandwidth_hz = 2000.0});
    const Real kp_2000 = foc.pi_d().params().kp;

    // Kp scales linearly with bandwidth: 4× faster loop → 4× Kp.
    CHECK(kp_2000 / kp_500 == Approx(4.0).epsilon(1e-9));
}
