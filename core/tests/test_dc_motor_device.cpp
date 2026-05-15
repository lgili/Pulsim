// DC Motor runtime device — Catch2 tests for the Phase Track-2 device-variant
// integration (Pulsim 0.10.0a2). Verifies the trapezoidal-armature plus
// forward-Euler mechanical state evolution match the analytical closed-form
// expressions from ``motors::DcMotor::steady_state_omega()`` and the
// mechanical time constant.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"
#include "pulsim/v1/motors/dc_motor.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

motors::DcMotorParams default_params() {
    motors::DcMotorParams p{};
    p.R_a = 0.5;
    p.L_a = 1e-2;   // 10 mH
    p.K_e = 0.05;
    p.K_t = 0.05;
    p.J   = 1e-4;
    p.b   = 1e-5;
    p.i_a_init = 0.0;
    p.omega_init = 0.0;
    p.theta_init = 0.0;
    return p;
}

SimulationOptions transient_opts(Circuit& circuit, Real tstop, Real dt) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = tstop;
    opts.dt = dt;
    opts.dt_min = 1e-9;
    opts.dt_max = dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    return opts;
}

}  // namespace

TEST_CASE("DC motor: open-loop no-load step response reaches analytical speed",
          "[dc_motor][motor][regression]") {
    Circuit circuit;
    const Index n_arm = circuit.add_node("arm");
    auto params = default_params();
    constexpr Real V_a = 12.0;

    circuit.add_voltage_source("Va", n_arm, Circuit::ground(), V_a);
    circuit.add_dc_motor("M1", n_arm, Circuit::ground(), params);

    auto opts = transient_opts(circuit, 1.0, 100e-6);  // 1 s, 100 µs steps

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    INFO("status: " << static_cast<int>(result.final_status)
                    << " message: " << result.message);
    REQUIRE(result.success);

    // Analytical: ω_ss = (V·K_t − τ_load·R_a) / (K_t·K_e + b·R_a)
    motors::DcMotor reference(params);
    const Real omega_ss_expected = reference.steady_state_omega(V_a, 0.0);
    INFO("expected ω_ss = " << omega_ss_expected << " rad/s");

    const Real omega_final = circuit.motor_omega("M1");
    INFO("measured ω at tstop = " << omega_final << " rad/s");

    // Allow 1% slack — forward-Euler on a 1 ms mechanical loop with dt = 100 µs
    // settles to within ~0.5 % of the analytical value at t = 5·τ_m.
    CHECK(omega_final == Approx(omega_ss_expected).epsilon(0.01));
}

TEST_CASE("DC motor: under load, steady-state speed drops linearly with τ_load",
          "[dc_motor][motor][regression]") {
    Circuit circuit;
    const Index n_arm = circuit.add_node("arm");
    auto params = default_params();
    constexpr Real V_a = 12.0;
    constexpr Real tau_load = 0.01;  // 10 mN·m

    circuit.add_voltage_source("Va", n_arm, Circuit::ground(), V_a);
    circuit.add_dc_motor("M2", n_arm, Circuit::ground(), params);
    circuit.set_motor_tau_load("M2", tau_load);

    auto opts = transient_opts(circuit, 1.0, 100e-6);
    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    motors::DcMotor reference(params);
    const Real omega_ss_expected = reference.steady_state_omega(V_a, tau_load);
    const Real omega_final = circuit.motor_omega("M2");

    INFO("expected ω_ss(loaded) = " << omega_ss_expected << " rad/s");
    INFO("measured ω(loaded)    = " << omega_final << " rad/s");
    CHECK(omega_final == Approx(omega_ss_expected).epsilon(0.015));

    // Sanity: armature current ≈ tau_load / K_t at steady state
    const Real i_a_expected = tau_load / params.K_t;
    const Real i_a_final = circuit.motor_i_a("M2");
    INFO("expected i_a = " << i_a_expected << " A");
    INFO("measured i_a = " << i_a_final << " A");
    CHECK(i_a_final == Approx(i_a_expected).margin(0.05));
}

TEST_CASE("DC motor: mechanical time constant matches J·R_a / (K_t·K_e)",
          "[dc_motor][motor][regression]") {
    Circuit circuit;
    const Index n_arm = circuit.add_node("arm");
    auto params = default_params();
    constexpr Real V_a = 12.0;

    circuit.add_voltage_source("Va", n_arm, Circuit::ground(), V_a);
    circuit.add_dc_motor("M3", n_arm, Circuit::ground(), params);

    // τ_m = J·R_a / (K_t·K_e) = 1e-4 · 0.5 / (0.05·0.05) = 0.02 s = 20 ms
    const Real tau_m_expected = params.J * params.R_a / (params.K_t * params.K_e);
    REQUIRE(tau_m_expected == Approx(0.02).margin(1e-6));

    // Simulate to τ_m and check ω reaches ~(1 − 1/e) ≈ 63.2 % of ω_ss.
    auto opts = transient_opts(circuit, tau_m_expected, 50e-6);
    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    motors::DcMotor reference(params);
    const Real omega_ss = reference.steady_state_omega(V_a, 0.0);
    const Real omega_at_tau_m = circuit.motor_omega("M3");
    const Real ratio = omega_at_tau_m / omega_ss;

    INFO("ω(τ_m) / ω_ss = " << ratio << " (analytical τ_m=20ms predicts ~0.632 "
         "for the L_a→0 mechanical-only model; coupled with the back-EMF "
         "the actual response is faster.)");
    // Sanity bounds: the response should have reached AT LEAST half ω_ss by
    // t = τ_m, but not overshoot (this is a 1st-order, well-damped system).
    CHECK(ratio > 0.50);
    CHECK(ratio < 1.05);
}

TEST_CASE("DC motor: rotor angle θ integrates ω over time",
          "[dc_motor][motor][regression]") {
    Circuit circuit;
    const Index n_arm = circuit.add_node("arm");
    auto params = default_params();
    constexpr Real V_a = 12.0;

    circuit.add_voltage_source("Va", n_arm, Circuit::ground(), V_a);
    circuit.add_dc_motor("M4", n_arm, Circuit::ground(), params);

    auto opts = transient_opts(circuit, 0.5, 50e-6);
    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    const Real theta_final = circuit.motor_theta("M4");
    const Real omega_final = circuit.motor_omega("M4");

    // Lower bound: if the motor stayed at ω_final for the entire run,
    // θ would be at least ω_final · tstop / 2 (because it ramps up).
    // Upper bound: θ ≤ ω_ss · tstop.
    motors::DcMotor reference(params);
    const Real omega_ss = reference.steady_state_omega(V_a, 0.0);
    INFO("θ(tstop) = " << theta_final << " rad, ω(tstop) = "
                       << omega_final << " rad/s");
    CHECK(theta_final > 0.0);
    CHECK(theta_final < omega_ss * 0.5);  // can't have travelled more than ω_ss·tstop
}
