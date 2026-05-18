// MechanicalDevice — Catch2 tests for the signal-domain mechanical primitive
// (consolidate-motors-and-three-phase, Phase B.2a).
//
// The device has no electrical pins; it advances ω and θ via forward Euler
// on each accepted timestep. These tests pin (1) inertia integration under
// pure input torque, (2) viscous-friction equilibrium, and (3) Circuit-side
// integration via `add_mechanical` + `update_history` walker.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/components/mechanical_device.hpp"
#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("MechanicalDevice: forward-Euler inertia integration matches "
          "analytical ω(t) = τ·t/J",
          "[mechanical][device][b2a]") {
    MechanicalDevice::Params params{};
    params.shaft.J = 1e-3;            // 1 mNm·s²
    params.shaft.b_friction = 0.0;    // pure inertia, no friction
    params.shaft.omega = 0.0;

    MechanicalDevice dev(params, "M1");
    dev.set_tau_input(0.5);            // constant 0.5 N·m

    const Real dt = 1e-3;
    const Real t_end = 0.1;
    const int n_steps = static_cast<int>(t_end / dt);

    for (int i = 0; i < n_steps; ++i) {
        dev.advance(dt);
    }

    // Analytical: J·dω/dt = τ → ω(t) = τ·t/J = 0.5 · 0.1 / 1e-3 = 50 rad/s
    const Real expected_omega = 0.5 * t_end / params.shaft.J;
    INFO("ω(t=" << t_end << "s) = " << dev.omega_m() << " rad/s, "
         "expected = " << expected_omega);
    CHECK(dev.omega_m() == Approx(expected_omega).margin(0.5));   // forward Euler accuracy band

    // θ ≈ (1/2)·(τ/J)·t² (rotational kinematics under constant torque)
    const Real expected_theta = 0.5 * (0.5 / params.shaft.J) * t_end * t_end;
    CHECK(dev.theta_m() == Approx(expected_theta).margin(0.5));
}

TEST_CASE("MechanicalDevice: viscous-friction equilibrium ω_ss = τ/b",
          "[mechanical][device][b2a]") {
    MechanicalDevice::Params params{};
    params.shaft.J = 1e-3;
    params.shaft.b_friction = 0.01;
    params.shaft.omega = 0.0;

    MechanicalDevice dev(params, "M2");
    dev.set_tau_input(1.0);            // constant 1.0 N·m

    // Settle for a few mechanical time constants τ_m = J/b = 0.1 s.
    const Real dt = 1e-3;
    const int n_steps = 1000;          // 1 second = 10 time constants
    for (int i = 0; i < n_steps; ++i) {
        dev.advance(dt);
    }

    // Analytical steady state: ω_ss = τ/b = 1.0 / 0.01 = 100 rad/s.
    const Real expected_omega_ss = 1.0 / params.shaft.b_friction;
    INFO("ω_ss = " << dev.omega_m() << " rad/s, expected = "
         << expected_omega_ss);
    CHECK(dev.omega_m() == Approx(expected_omega_ss).epsilon(0.01));
}

TEST_CASE("MechanicalDevice: registers in Circuit and advances on transient",
          "[mechanical][device][b2a][circuit]") {
    // Smoke test for the Circuit-side integration: add a MechanicalDevice
    // alongside a passive RC, run a brief transient, and verify the
    // mechanical state advances under the user-supplied τ_input.
    Circuit ckt;
    auto vin = ckt.add_node("vin");
    auto out = ckt.add_node("out");
    ckt.add_voltage_source("V1", vin, Circuit::ground(), 1.0);
    ckt.add_resistor("R1", vin, out, 100.0);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);

    MechanicalDevice::Params mech{};
    mech.shaft.J = 1e-4;
    mech.shaft.b_friction = 0.0;
    mech.tau_load_const = 0.0;
    ckt.add_mechanical("Shaft1", mech);
    ckt.set_mechanical_tau_input("Shaft1", 0.1);   // 0.1 N·m

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 0.01;
    opts.dt = 100e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 100e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto run = sim.run_transient();
    REQUIRE(run.success);

    // After ~10 ms under 0.1 N·m torque with J = 1e-4 kg·m², ω should be
    // around τ·t/J = 0.1 · 0.01 / 1e-4 = 10 rad/s (forward-Euler band).
    const Real omega = ckt.mechanical_omega("Shaft1");
    INFO("Mechanical ω after transient = " << omega << " rad/s");
    CHECK(omega == Approx(10.0).margin(2.0));   // generous band — forward Euler under coarse dt

    // θ should be positive and roughly 0.5 · (τ/J) · t².
    const Real theta = ckt.mechanical_theta("Shaft1");
    CHECK(theta > 0.0);
}
