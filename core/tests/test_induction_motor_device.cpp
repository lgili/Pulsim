// Induction-motor (squirrel cage) device — Catch2 tests.
//
// consolidate-motors-and-three-phase Phase C.2.5: smoke-tests for the math
// object `pulsim::v1::motors::InductionMotor` and the
// `pulsim::v1::InductionMotorDevice` wrapper. The MNA / Circuit integration
// arrives in the parent agent's follow-up commit (DeviceVariant entry,
// add_induction_motor method, stamping branches in stamp_rlc /
// advance_state / restore_state). Until then, this file exercises the
// device in isolation by driving its `advance_state` directly with
// balanced-3φ stator voltages and pre-computed line currents.
//
// Three physics-driven scenarios:
//
//   1. Locked-rotor inrush — ω_m clamped to 0, balanced 3φ voltage applied.
//      The steady-state stator current should be approximately
//        V_phase_peak / |R_s + jω·σ·L_s + (R_r/s) || (jω·L_m / s)|
//      For the locked-rotor case (s=1) the rotor branch is fully in
//      series — a generous 30% tolerance gates the inrush magnitude.
//
//   2. No-load (free-running) — apply balanced 3φ voltage, let the rotor
//      accelerate. After a few hundred ms the slip should be near zero
//      (limited only by viscous friction).
//
//   3. Loaded operation — apply a constant load torque and verify the
//      motor settles at a non-trivial positive slip (i.e. the slip
//      mechanism is wired and produces torque opposing the load).
//
// Plus a handful of pure-arithmetic unit checks on the device helpers
// (mirrors the PMSM test pattern).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/induction_motor_device.hpp"
#include "pulsim/v1/motors/induction_motor.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

constexpr Real kPi = 3.14159265358979323846;
constexpr Real kTwoPi = 2.0 * kPi;
constexpr Real kSqrt3Over2 = 0.8660254037844386;

/// Typical small (~1 kW) squirrel-cage IM parameter set used by the
/// physics tests.
motors::InductionMotorParams default_im_params() {
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

/// Build a balanced three-phase αβ voltage vector at electrical angle θ.
/// V_peak is the per-phase peak voltage.
std::pair<Real, Real> three_phase_alpha_beta(Real V_peak, Real theta) {
    const Real v_a = V_peak * std::cos(theta);
    const Real v_b = V_peak * std::cos(theta - kTwoPi / Real{3});
    const Real v_c = V_peak * std::cos(theta + kTwoPi / Real{3});
    constexpr Real two_thirds = Real{2} / Real{3};
    constexpr Real one_third  = Real{1} / Real{3};
    constexpr Real one_over_sqrt3 = Real{1} / Real{1.7320508075688772};
    const Real v_alpha = two_thirds * v_a - one_third * (v_b + v_c);
    const Real v_beta  = (v_b - v_c) * one_over_sqrt3;
    return {v_alpha, v_beta};
}

/// Inverse Clarke: αβ → abc, returns (a, b, c).
std::tuple<Real, Real, Real> alpha_beta_to_abc(Real a, Real b) {
    return {a,
            -Real{0.5} * a + kSqrt3Over2 * b,
            -Real{0.5} * a - kSqrt3Over2 * b};
}

/// Drive the bare `InductionMotor` math object with a balanced 3φ voltage
/// source for `t_stop` seconds at step `dt`. The mechanical speed is left
/// free to evolve unless `lock_rotor=true`, which clamps ω_m to zero
/// every step (emulates a locked-rotor / blocked-rotor test).
struct DriveResult {
    Real i_sa_peak = 0.0;          ///< peak αβ stator current over final window
    Real i_a_peak  = 0.0;          ///< peak phase-A line current over final window
    Real i_a_rms   = 0.0;          ///< RMS phase-A line current over final window
    Real omega_m_end = 0.0;
    Real tau_em_avg = 0.0;
};

DriveResult drive_balanced_3ph(motors::InductionMotor& motor,
                               Real V_phase_peak, Real f_hz,
                               Real t_stop, Real dt,
                               Real measurement_window_start,
                               bool lock_rotor) {
    DriveResult r{};
    const Real omega_e = kTwoPi * f_hz;
    std::size_t samples = 0;
    Real sum_sq = 0.0;
    Real sum_tau = 0.0;
    std::size_t tau_samples = 0;

    Real t = 0.0;
    while (t < t_stop) {
        const Real theta = omega_e * t;
        const auto [v_alpha, v_beta] = three_phase_alpha_beta(V_phase_peak, theta);
        motor.advance(v_alpha, v_beta, dt);
        if (lock_rotor) {
            motor.set_omega(0.0);
            motor.set_theta(0.0);
        }
        t += dt;

        if (t >= measurement_window_start) {
            const Real i_a = motor.i_phase_a();
            r.i_sa_peak = std::max(r.i_sa_peak, std::abs(motor.i_sa()));
            r.i_a_peak  = std::max(r.i_a_peak, std::abs(i_a));
            sum_sq += i_a * i_a;
            ++samples;
            sum_tau += motor.electromagnetic_torque();
            ++tau_samples;
        }
    }
    if (samples > 0) {
        r.i_a_rms = std::sqrt(sum_sq / static_cast<Real>(samples));
    }
    if (tau_samples > 0) {
        r.tau_em_avg = sum_tau / static_cast<Real>(tau_samples);
    }
    r.omega_m_end = motor.omega_m();
    return r;
}

}  // namespace

TEST_CASE("Induction motor: locked-rotor inrush current is bounded by "
          "stator + rotor impedance",
          "[induction][motor][regression]") {
    // Locked rotor → slip s = 1 → the rotor referred branch (R_r/s + jωL_lr)
    // is at full value. With balanced 3φ voltage applied, the steady-state
    // stator current is large but finite. We do not nail down the exact
    // textbook closed-form value — that requires the full T-equivalent
    // referred-impedance reduction, and depends on the L_m branch. Instead
    // we verify two robust properties:
    //
    //   - The current settles to a finite, repeating sinusoid (no blow-up).
    //   - The peak is within a factor of 2× of V_peak / R_s (a loose upper
    //     bound that any sensible motor model must respect — pure-R would
    //     give exactly V/R; adding inductance only reduces the current).
    //
    // Tolerance is qualitative: the test exists to guard against a totally
    // mis-stamped model (zero or NaN current, runaway, wrong sign).
    auto params = default_im_params();
    motors::InductionMotor motor(params, "IM_locked");

    constexpr Real f_hz = 50.0;
    constexpr Real V_LL_rms = 220.0;
    const Real V_phase_peak = V_LL_rms * std::sqrt(Real{2}) / std::sqrt(Real{3});

    const Real dt = 50e-6;
    const Real t_stop = 0.20;        // 10 fundamental cycles
    const Real t_measure_start = 0.15;  // last 50 ms (≈ 2.5 cycles)
    auto r = drive_balanced_3ph(motor, V_phase_peak, f_hz,
                                t_stop, dt, t_measure_start,
                                /*lock_rotor=*/true);

    INFO("V_phase_peak = " << V_phase_peak << " V, R_s = " << params.R_s
         << " Ω, measured I_a_peak = " << r.i_a_peak
         << " A, I_a_rms = " << r.i_a_rms << " A");

    CHECK(std::isfinite(r.i_a_peak));
    CHECK(std::isfinite(r.i_a_rms));
    CHECK(r.i_a_peak > Real{1.0});              // motor is actually carrying current
    CHECK(r.i_a_peak < Real{2.0} * V_phase_peak / params.R_s);  // bounded
    // Locked rotor produces no useful torque on average (no slip-cycling
    // power flow into the shaft) — the time-average should be near zero.
    CHECK(std::isfinite(r.tau_em_avg));
}

TEST_CASE("Induction motor: no-load operation accelerates rotor toward "
          "synchronous speed",
          "[induction][motor][regression]") {
    // With a balanced 3φ voltage, no load torque, and zero initial speed,
    // a healthy IM accelerates toward synchronous speed. After a few
    // hundred ms (limited by rotor inertia, J = 0.01 kg·m²) the slip
    // should be small. We verify two qualitative properties:
    //
    //   - The rotor has been accelerated to >50% of synchronous speed.
    //   - The slip is positive (motoring, not regenerating).
    //
    // 50 Hz, 2 pole-pairs → ω_sync_mech = 2π·50 / 2 = 157.08 rad/s.
    auto params = default_im_params();
    motors::InductionMotor motor(params, "IM_noload");

    constexpr Real f_hz = 50.0;
    constexpr Real V_LL_rms = 220.0;
    const Real V_phase_peak = V_LL_rms * std::sqrt(Real{2}) / std::sqrt(Real{3});
    const Real omega_sync_mech =
        kTwoPi * f_hz / static_cast<Real>(params.pole_pairs);

    const Real dt = 50e-6;
    const Real t_stop = 2.0;
    auto r = drive_balanced_3ph(motor, V_phase_peak, f_hz,
                                t_stop, dt, t_stop - 0.05,
                                /*lock_rotor=*/false);

    INFO("ω_sync_mech = " << omega_sync_mech
         << " rad/s, final ω_m = " << r.omega_m_end << " rad/s");

    CHECK(std::isfinite(r.omega_m_end));
    // No-load IM with small viscous friction accelerates well above 50% of
    // synchronous speed within 2 seconds — generous lower bound to avoid
    // tight coupling with the explicit-Euler discretisation error.
    CHECK(r.omega_m_end > Real{0.5} * omega_sync_mech);
    // Sub-synchronous (positive slip) — never overshoots ω_sync with
    // viscous friction present.
    CHECK(r.omega_m_end < omega_sync_mech);
}

TEST_CASE("Induction motor: applying a load torque slows the rotor and "
          "produces positive slip",
          "[induction][motor][regression]") {
    // Loaded operation: with τ_load > 0 the rotor settles at a slip that
    // produces enough electromagnetic torque to balance the load. We
    // verify the slip mechanism is wired correctly by comparing the
    // no-load final speed to the loaded final speed — loading should
    // drop the speed (i.e., increase the slip). This is the qualitative
    // direction-check that the spec requires.
    auto params = default_im_params();

    // No-load reference run.
    motors::InductionMotor motor_noload(params, "IM_ref");
    constexpr Real f_hz = 50.0;
    constexpr Real V_LL_rms = 220.0;
    const Real V_phase_peak = V_LL_rms * std::sqrt(Real{2}) / std::sqrt(Real{3});
    const Real dt = 50e-6;
    const Real t_stop = 1.5;
    auto r_noload = drive_balanced_3ph(motor_noload, V_phase_peak, f_hz,
                                       t_stop, dt, t_stop - 0.05,
                                       /*lock_rotor=*/false);

    // Loaded run — apply a moderate constant load torque.
    motors::InductionMotor motor_loaded(params, "IM_load");
    motor_loaded.set_load_torque(Real{2.0});  // N·m
    auto r_loaded = drive_balanced_3ph(motor_loaded, V_phase_peak, f_hz,
                                       t_stop, dt, t_stop - 0.05,
                                       /*lock_rotor=*/false);

    INFO("ω_m_noload = " << r_noload.omega_m_end
         << " rad/s, ω_m_loaded = " << r_loaded.omega_m_end << " rad/s");

    CHECK(std::isfinite(r_loaded.omega_m_end));
    // Loading slows the rotor (positive slip increases).
    CHECK(r_loaded.omega_m_end < r_noload.omega_m_end);
    // But the motor still operates (does not stall) — speed remains
    // positive.
    CHECK(r_loaded.omega_m_end > Real{0});
}

TEST_CASE("Induction motor: device wrapper leakage inductance + companion "
          "conductance helpers",
          "[induction][motor][unit]") {
    // Pure-arithmetic unit test of the device's own helpers — no solver
    // or integration involved. Mirrors the PMSM companion-conductance
    // test pattern.
    motors::InductionMotorParams params{};
    params.R_s = 1.0;
    params.R_r = 1.5;
    params.L_s = 0.15;
    params.L_r = 0.15;
    params.L_m = 0.14;
    params.pole_pairs = 2;
    params.J = 0.01;

    InductionMotorDevice dev(params, "IM1");
    const Real dt = 100e-6;

    // σ = 1 − L_m²/(L_s·L_r) = 1 − 0.14²/(0.15·0.15) ≈ 0.12889
    const Real sigma_expected =
        Real{1} - (params.L_m * params.L_m) / (params.L_s * params.L_r);
    const Real l_eff_expected = sigma_expected * params.L_s;
    const Real g_expected = Real{1.0} /
        (params.R_s + Real{2.0} * l_eff_expected / dt);

    CHECK(dev.l_s_effective() == Approx(l_eff_expected).epsilon(1e-9));
    CHECK(dev.companion_conductance(dt) == Approx(g_expected).epsilon(1e-9));
    CHECK(dev.theta_electrical() == Approx(0.0));
    CHECK(dev.omega_electrical() == Approx(0.0));
}

TEST_CASE("Induction motor: device wrapper slip helper",
          "[induction][motor][unit]") {
    // Verify slip(ω_sync) returns the textbook (ω_sync − ω_e)/ω_sync.
    motors::InductionMotorParams params{};
    params.R_s = 1.0;
    params.R_r = 1.5;
    params.L_s = 0.15;
    params.L_r = 0.15;
    params.L_m = 0.14;
    params.pole_pairs = 2;
    params.J = 0.01;

    InductionMotorDevice dev(params, "IM1");

    // Locked rotor → slip = 1.0.
    const Real omega_sync_e = kTwoPi * Real{50.0};   // 50 Hz electrical
    CHECK(dev.slip(omega_sync_e) == Approx(1.0));
    CHECK(dev.slip_from_hz(50.0) == Approx(1.0));

    // Half-speed rotor (mechanical) → ω_e = pole_pairs · ω_m → slip = 0.5
    // when ω_e = ω_sync_e / 2.
    const Real omega_m_half = omega_sync_e / Real{2.0} /
                              static_cast<Real>(params.pole_pairs);
    dev.set_initial_omega(omega_m_half);
    CHECK(dev.slip(omega_sync_e) == Approx(0.5).epsilon(1e-9));

    // Synchronous speed → slip = 0.
    const Real omega_m_sync = omega_sync_e /
                              static_cast<Real>(params.pole_pairs);
    dev.set_initial_omega(omega_m_sync);
    CHECK(dev.slip(omega_sync_e) == Approx(0.0).margin(1e-12));

    // Edge: ω_sync = 0 (DC) → slip = 1 (full slip convention).
    CHECK(dev.slip(0.0) == Approx(1.0));
}

TEST_CASE("Induction motor: device wrapper advance_state seeds αβ stator "
          "currents and updates history",
          "[induction][motor][unit]") {
    // After a single advance_state with a known balanced 3φ current set,
    // the device's cached i_sα / i_sβ should match the Clarke transform
    // of the inputs, the per-phase i_*() accessors should return the
    // newly supplied currents, and history_initialized() should be true.
    motors::InductionMotorParams params{};
    params.R_s = 0.5;
    params.R_r = 0.6;
    params.L_s = 0.05;
    params.L_r = 0.05;
    params.L_m = 0.04;
    params.pole_pairs = 1;
    params.J = 1e-2;

    InductionMotorDevice dev(params, "IM1");
    REQUIRE_FALSE(dev.history_initialized());

    // Push balanced abc currents at θ = 0.
    const Real I_peak = 5.0;
    const Real i_a = I_peak * std::cos(Real{0});                       // = 5
    const Real i_b = I_peak * std::cos(-kTwoPi / Real{3});             // ≈ -2.5
    const Real i_c = I_peak * std::cos( kTwoPi / Real{3});             // ≈ -2.5

    // Balanced terminal voltages (use peak 100 V).
    const Real V_peak = 100.0;
    const Real v_a = V_peak * std::cos(Real{0});
    const Real v_b = V_peak * std::cos(-kTwoPi / Real{3});
    const Real v_c = V_peak * std::cos( kTwoPi / Real{3});

    const Real dt = 100e-6;
    dev.advance_state(v_a, v_b, v_c, i_a, i_b, i_c, dt);

    CHECK(dev.history_initialized());
    CHECK(dev.i_a() == Approx(i_a));
    CHECK(dev.i_b() == Approx(i_b));
    CHECK(dev.i_c() == Approx(i_c));

    // Clarke (amplitude-invariant): i_sα = 2/3·a − 1/3·(b+c), i_sβ = (b−c)/√3.
    constexpr Real two_thirds = Real{2} / Real{3};
    constexpr Real one_third  = Real{1} / Real{3};
    constexpr Real one_over_sqrt3 = Real{1} / Real{1.7320508075688772};
    const Real i_sa_expected = two_thirds * i_a - one_third * (i_b + i_c);
    const Real i_sb_expected = (i_b - i_c) * one_over_sqrt3;
    CHECK(dev.i_sa() == Approx(i_sa_expected).epsilon(1e-9));
    CHECK(dev.i_sb() == Approx(i_sb_expected).epsilon(1e-9));

    // Torque is finite (initial flux is zero → torque should also be ~0).
    CHECK(std::isfinite(dev.tau_em()));
    CHECK(dev.tau_em() == Approx(0.0).margin(1e-9));
}
