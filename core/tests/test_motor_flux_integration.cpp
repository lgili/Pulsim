// =============================================================================
// Phase A5 of harden-component-models-vs-psim-plecs: trapezoidal integration
// on the induction-motor rotor-flux state and the PSC run-capacitor voltage.
// =============================================================================
//
// What changed:
//   * `motors::InductionMotor::advance` and
//     `InductionMotorDevice::advance_state`: rotor flux (ψ_rα, ψ_rβ) now
//     advances via one-iteration trapezoidal (Heun's predictor-corrector)
//     instead of forward Euler. Stator-current and mechanical updates
//     remain forward Euler — they have separate stability headroom and
//     the device wrapper already uses a trapezoidal MNA companion for
//     the stator inductors.
//   * `motors::SinglePhaseInductionMotor::advance` and
//     `SinglePhaseInductionMotorDevice::advance_state`: the run-cap
//     voltage V_cap now advances via the average of (i_aux_old,
//     i_aux_new), removing the half-step lag introduced by the
//     previous `V_cap += dt·i_aux/C_run` forward-Euler form.
//
// These are focused unit tests on the integration step itself. The
// system-level smoothness benefit is real but tangled up with the
// stator-current forward-Euler step, the mechanical step, and the
// nonlinear ω_e cross-coupling, so the cleanest regression test is to
// confirm the integration rule applied to a single step matches the
// trapezoidal formula bit-for-bit (and is strictly more accurate than
// forward Euler on a known closed-form rotation).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/motors/induction_motor.hpp"
#include "pulsim/v1/motors/single_phase_induction_motor.hpp"

#include <algorithm>
#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

constexpr Real kPi    = 3.14159265358979323846;
constexpr Real kTwoPi = 2.0 * kPi;

motors::InductionMotorParams kw1_im() {
    motors::InductionMotorParams p{};
    p.R_s = 1.0;
    p.R_r = 1.5;
    p.L_s = 0.15;
    p.L_r = 0.15;
    p.L_m = 0.14;
    p.pole_pairs = 2;
    p.J = 0.01;
    p.b_friction = 1e-3;
    return p;
}

motors::SinglePhaseInductionMotorParams psc_compressor() {
    motors::SinglePhaseInductionMotorParams p{};
    p.R_s_main = 10.0;
    p.L_s_main = 50e-3;
    p.R_s_aux  = 20.0;
    p.L_s_aux  = 80e-3;
    p.C_run    = 4e-6;
    p.R_r      = 8.0;
    p.L_r      = 55e-3;
    p.L_m      = 50e-3;
    p.pole_pairs = 2;
    p.J = 1e-4;
    p.b_friction = 1e-4;
    p.friction_coulomb = 0.05;
    return p;
}

}  // namespace

// -----------------------------------------------------------------------------
// PSC run-cap trapezoidal V_cap
// -----------------------------------------------------------------------------

TEST_CASE("PSC compressor: trapezoidal V_cap integration matches the "
          "0.5·dt·(i_old + i_new)/C closed form",
          "[v1][single_phase_im][regression][a5]") {
    // One-step check: stage a known (i_aux_old, i_aux_new) pair by
    // running `advance` twice with controlled line voltages, then verify
    // ΔV_cap = 0.5·dt·(i_aux_old + i_aux_new) / C_run within a tight
    // numerical tolerance.
    motors::SinglePhaseInductionMotor m(psc_compressor(), "SPIM_A5_step");

    const Real dt = 100e-6;
    // Seed with one step at zero line voltage to capture a stable
    // i_aux_old.
    m.advance(Real{0.0}, dt);
    const Real V_cap_before = m.V_cap();
    const Real i_aux_old   = m.i_aux();

    // Drive a step. The integration order is:
    //   1. Update i_main, i_aux (forward Euler) and ψ_r.
    //   2. V_cap += 0.5 · dt · (i_aux_old + i_aux_new) / C_run.
    m.advance(Real{220.0} * std::sqrt(Real{2}), dt);
    const Real V_cap_after = m.V_cap();
    const Real i_aux_new   = m.i_aux();

    const Real expected_dV =
        Real{0.5} * dt * (i_aux_old + i_aux_new) / Real{4e-6};
    const Real actual_dV = V_cap_after - V_cap_before;

    INFO("dt          = " << dt);
    INFO("i_aux_old   = " << i_aux_old);
    INFO("i_aux_new   = " << i_aux_new);
    INFO("expected ΔV = " << expected_dV);
    INFO("actual ΔV   = " << actual_dV);

    CHECK(actual_dV == Approx(expected_dV).margin(1e-12));

    // Sanity: the trapezoidal form differs from forward Euler when
    // i_aux changes within the step. At the i_aux step above, the FE
    // form would give `dt · i_aux_new / C_run` (using the END-of-step
    // current). The two formulae must differ; this protects against
    // an accidental revert of the integration rule.
    const Real fe_dV =
        dt * i_aux_new / Real{4e-6};
    CHECK(std::abs(actual_dV - fe_dV) > Real{1e-6});
}

// -----------------------------------------------------------------------------
// Trapezoidal ψ_r — closed-form rotation test
// -----------------------------------------------------------------------------

TEST_CASE("Induction motor: trapezoidal ψ_r integration tracks pure αβ "
          "rotation more accurately than forward Euler",
          "[v1][induction_motor][regression][a5]") {
    // Construct a degenerate motor with R_r = 0 and i_s = 0. The rotor-
    // flux ODE collapses to a pure rotation in the αβ plane:
    //
    //   dψ_rα/dt = -ω_e · ψ_rβ
    //   dψ_rβ/dt = +ω_e · ψ_rα
    //
    // Closed-form solution: |ψ_r(t)| = |ψ_r(0)| (exact rotation). After
    // N steps the magnitude error of any explicit method is bounded by
    // (ω_e · dt)^p · N for a p-th-order scheme:
    //   * Forward Euler   (p = 1): magnitude GROWS by (1 + (ω_e·dt)²)^(N/2)
    //   * Trapezoidal/RK2 (p = 2): magnitude error O((ω_e·dt)³ · N)
    //
    // At ω_e = 314 rad/s (50 Hz), dt = 100 µs, N = 200 steps:
    //   * FE magnitude grows by exp(N·ω_e²·dt²/2) ≈ exp(0.0987) ≈ 1.10
    //     → ~10% magnitude blow-up
    //   * Trapezoidal magnitude error ≈ N·(ω_e·dt)³ / 6 ≈ 200·3.1e-5/6
    //     ≈ 1e-3 → ~0.1%

    auto params = kw1_im();
    // Force pure rotation: zero rotor resistance + zero stator coupling.
    params.R_r = 0.0;
    params.L_m = 0.0;
    motors::InductionMotor motor(params, "IM_rotation");

    // Seed the rotor flux at unit magnitude, on the α-axis.
    motor.set_rotor_flux(Real{1.0}, Real{0.0});
    // Pin the rotor at a fixed mechanical speed so ω_e stays constant
    // throughout the simulation.
    constexpr Real omega_e_target = 314.159;
    motor.set_omega(omega_e_target / static_cast<Real>(params.pole_pairs));

    const Real dt = 100e-6;
    const std::size_t N = 200;        // 20 ms
    for (std::size_t k = 0; k < N; ++k) {
        // Zero stator voltage. With L_m = 0 the rotor branch is
        // electrically decoupled from the stator and the αβ flux
        // rotates at ω_e.
        motor.advance(Real{0.0}, Real{0.0}, dt);
        // Re-pin ω so it doesn't drift from the friction model — the
        // test is about ψ_r integration accuracy under a constant ω_e.
        motor.set_omega(omega_e_target / static_cast<Real>(params.pole_pairs));
    }

    const Real psi_a_end = motor.psi_ra();
    const Real psi_b_end = motor.psi_rb();
    const Real magnitude =
        std::sqrt(psi_a_end * psi_a_end + psi_b_end * psi_b_end);

    INFO("|ψ_r(0)| = 1.0");
    INFO("|ψ_r(N)| = " << magnitude);
    INFO("error    = " << std::abs(magnitude - Real{1.0}));

    CHECK(std::isfinite(magnitude));
    // Trapezoidal target: < 1 % magnitude error. Forward Euler at this
    // op-point produces ~10 % growth, so this is a clean discriminator.
    CHECK(std::abs(magnitude - Real{1.0}) < Real{0.01});
}
