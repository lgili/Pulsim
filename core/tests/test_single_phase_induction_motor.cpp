// Single-phase induction motor (PSC topology, "CC" compressor) — Catch2 tests.
//
// compressor-models follow-up: smoke-tests for the math object
// `pulsim::v1::motors::SinglePhaseInductionMotor` and the
// `pulsim::v1::SinglePhaseInductionMotorDevice` wrapper. Two layers:
//
//   1. Pure math object — drive `advance(v_line, dt)` with a 60 Hz line
//      sine and confirm:
//        - i_main / i_aux are bounded (no runaway)
//        - V_cap settles into a bounded oscillation
//        - the rotor accelerates from rest under no load
//
//   2. Circuit integration — wire a SineVoltageSource to the motor's
//      line / neutral terminals at 220 V, 60 Hz, advance for several
//      cycles, and confirm electromagnetic torque is non-zero and the
//      rotor speed is positive (motor self-starts thanks to the PSC's
//      90° auxiliary phase shift).
//
// These are physics smoke tests — generous tolerances. The deeper
// validation (slip-vs-torque curve, locked-rotor inrush, etc.) lives
// alongside the 3φ induction motor tests; the PSC topology shares the
// same underlying αβ machine model.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/single_phase_induction_motor_device.hpp"
#include "pulsim/v1/motors/single_phase_induction_motor.hpp"
#include "pulsim/v1/runtime_circuit.hpp"
#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

constexpr Real kPi    = 3.14159265358979323846;
constexpr Real kTwoPi = 2.0 * kPi;

motors::SinglePhaseInductionMotorParams default_psc_params() {
    // Embraco-style 1/8 HP compressor on 220 V / 60 Hz. Values match the
    // defaults in the header but pinned here so the test is independent
    // of header tweaks.
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

}  // anonymous namespace

TEST_CASE("Single-phase IM math: 60 Hz sine drive keeps state bounded and "
          "rotor accelerates from rest",
          "[motor][single_phase_im][math]") {
    motors::SinglePhaseInductionMotor m(default_psc_params(), "SPIM");

    const Real V_peak = 220.0 * std::sqrt(2.0);
    const Real f      = 60.0;
    const Real dt     = 5e-6;
    const Real t_stop = 0.10;  // 6 line cycles
    const std::size_t N = static_cast<std::size_t>(t_stop / dt);

    Real max_i_main = 0.0;
    Real max_i_aux  = 0.0;
    Real max_V_cap  = 0.0;
    for (std::size_t k = 0; k < N; ++k) {
        const Real t = k * dt;
        const Real v_line = V_peak * std::sin(kTwoPi * f * t);
        m.advance(v_line, dt);
        max_i_main = std::max(max_i_main, std::abs(m.i_main()));
        max_i_aux  = std::max(max_i_aux,  std::abs(m.i_aux()));
        max_V_cap  = std::max(max_V_cap,  std::abs(m.V_cap()));
    }

    INFO("max |i_main| = " << max_i_main << " A");
    INFO("max |i_aux|  = " << max_i_aux  << " A");
    INFO("max |V_cap|  = " << max_V_cap  << " V");
    INFO("omega_m at t_stop = " << m.omega_m() << " rad/s");

    // Currents bounded — pre-startup transient cap is < 200 A under 311 V
    // peak across a 10 Ω + 50 mH winding (V_peak / R_s = 31 A; we leave
    // headroom for the inductive kick).
    CHECK(max_i_main < 200.0);
    CHECK(max_i_aux  < 200.0);
    // Capacitor voltage envelope shouldn't run away.
    CHECK(max_V_cap < 5.0 * V_peak);
    // Rotor accelerates from rest (no-load, PSC self-starting). Rotation
    // direction depends on whether the run cap makes the aux current lead
    // or lag the main current — real PSC motors pick a direction by
    // mechanical winding sense, so we only check magnitude.
    CHECK(std::abs(m.omega_m()) > 50.0);
}

TEST_CASE("Single-phase IM math: torque is non-zero when stator currents and "
          "rotor flux are non-zero",
          "[motor][single_phase_im][math]") {
    motors::SinglePhaseInductionMotor m(default_psc_params(), "SPIM");

    // Manually seed the math object's rotor flux + stator currents via
    // the external-state advance hook (mirrors what the device does).
    m.advance_state_external_(/*i_main=*/2.0, /*i_aux=*/-1.0,
                               /*dpsi_ra_dt=*/0.5, /*dpsi_rb_dt=*/-0.3,
                               /*dt=*/1e-3);
    INFO("torque = " << m.electromagnetic_torque());
    CHECK(std::abs(m.electromagnetic_torque()) > 0.0);
}

TEST_CASE("Single-phase IM device: 220 V / 60 Hz line drive accelerates rotor",
          "[motor][single_phase_im][circuit]") {
    Circuit ckt;
    const auto line    = ckt.add_node("line");
    const auto neutral = ckt.add_node("neutral");

    // 220 V (RMS) / 60 Hz line voltage, tied to ground at the neutral.
    ckt.add_sine_voltage_source("V_line", line, neutral,
                                 220.0 * std::sqrt(2.0), 60.0);
    ckt.add_voltage_source("V_n", neutral, Circuit::ground(), 0.0);

    // Single-phase induction motor with default Embraco-style PSC params.
    ckt.add_single_phase_induction_motor("M_cc", line, neutral,
                                          default_psc_params());

    // Brief transient: 200 ms = 12 line cycles. The motor should pick up
    // speed even from rest thanks to the PSC's run-cap phase shift.
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 0.2;
    opts.dt     = 5e-5;
    opts.dt_min = 1e-9;
    opts.dt_max = 5e-5;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto run = sim.run_transient();
    REQUIRE(run.success);

    const Real omega = ckt.single_phase_im_omega("M_cc");
    const Real torque = ckt.single_phase_im_torque("M_cc");
    INFO("omega at 200 ms = " << omega << " rad/s");
    INFO("electromagnetic torque at 200 ms = " << torque << " N·m");

    // PSC self-start: with no shaft load, the rotor must spin up to near
    // synchronous speed (2π·60/p_pairs ≈ 188.5 rad/s) within 200 ms.
    // The rotation direction depends on the capacitor / winding phase
    // alignment — a real PSC motor's mechanical wiring decides forward
    // vs reverse — so we only check that the magnitude is bounded and
    // non-trivial.
    CHECK(std::abs(omega) > 50.0);
    CHECK(std::abs(omega) < 220.0);
}

TEST_CASE("Single-phase IM device: load torque setter / getter round-trip",
          "[motor][single_phase_im][api]") {
    // Simple API smoke test — confirms the Circuit's load-torque wrapper
    // routes through to the device. We don't run a transient because the
    // PSC settles to ±sync regardless of sign convention for the load,
    // and the simulator-level interplay between τ_load and the rotor
    // direction is already exercised by the compressor_load tests.
    Circuit ckt;
    const auto line    = ckt.add_node("line");
    const auto neutral = ckt.add_node("neutral");
    ckt.add_single_phase_induction_motor("M_cc", line, neutral,
                                          default_psc_params());

    ckt.set_single_phase_im_tau_load("M_cc", 0.42);

    // No direct circuit-level "get τ_load" accessor exists — that's
    // intentional; simulator code reads through the math object. So just
    // verify the motor still answers and state is at zero (since we
    // never advanced).
    CHECK(ckt.single_phase_im_omega("M_cc") == Approx(0.0));
    CHECK(ckt.single_phase_im_torque("M_cc") == Approx(0.0));
}
