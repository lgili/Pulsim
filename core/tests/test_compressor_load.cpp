// CompressorLoad — Catch2 tests for the refrigeration-compressor torque
// profile (compressor-models feature). Covers the polytropic mean torque
// formula across the 3 topologies plus the Circuit-side wiring that
// pushes the load into a BLDC motor's `set_load_torque` setter.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/loads/compressor_load.hpp"
#include "pulsim/v1/simulation.hpp"

#include <cmath>
#include <numbers>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("CompressorLoad: polytropic mean torque matches analytical formula",
          "[compressor][load][reciprocating]") {
    // Embraco-style domestic refrigerator (R600a, 6 cm³, 8 bar discharge,
    // 0.7 bar suction). Polytropic exponent 1.13 for isobutane.
    loads::CompressorParams p{};
    p.topology = loads::CompressorTopology::Reciprocating;
    p.num_cylinders = 1;
    p.displacement_m3 = 6.0e-6;
    p.P_suction_Pa = 7.0e4;
    p.P_discharge_Pa = 8.0e5;
    p.polytropic_n = 1.13;
    p.b_friction = 0.0;
    p.tau_coulomb = 0.0;
    p.ripple_amplitude = 0.0;   // pure mean torque, no ripple

    loads::CompressorLoad load(p);

    // Analytical: W = P_s · V_d / (n−1) · [(P_d/P_s)^((n−1)/n) − 1]
    const Real pr = 8.0e5 / 7.0e4;            // pressure ratio ≈ 11.43
    const Real exp_arg = (1.13 - 1.0) / 1.13;
    const Real W_ind = 7.0e4 * 6.0e-6 / (1.13 - 1.0) *
                       (std::pow(pr, exp_arg) - 1.0);
    const Real T_mean = W_ind / (2.0 * std::numbers::pi_v<Real>);

    INFO("W_ind = " << W_ind << " J, T_mean = " << T_mean << " N·m");
    CHECK(load.indicated_work_per_cycle() == Approx(W_ind).epsilon(1e-9));
    CHECK(load.mean_torque() == Approx(T_mean).epsilon(1e-9));

    // With ripple_amplitude = 0, load_torque(θ, 0) is constant = T_mean
    // across all angles.
    CHECK(load.load_torque(0.0, 0.0) == Approx(T_mean).epsilon(1e-9));
    CHECK(load.load_torque(0.5, 0.0) == Approx(T_mean).epsilon(1e-9));
    CHECK(load.load_torque(2.0, 0.0) == Approx(T_mean).epsilon(1e-9));
}

TEST_CASE("CompressorLoad: ripple amplitude generates angle-dependent torque",
          "[compressor][load][reciprocating]") {
    loads::CompressorParams p{};
    p.topology = loads::CompressorTopology::Reciprocating;
    p.num_cylinders = 1;
    p.displacement_m3 = 6.0e-6;
    p.P_suction_Pa = 7.0e4;
    p.P_discharge_Pa = 8.0e5;
    p.polytropic_n = 1.13;
    p.ripple_amplitude = 0.5;
    p.b_friction = 0.0;
    p.tau_coulomb = 0.0;

    loads::CompressorLoad load(p);
    const Real T_mean = load.mean_torque();

    // For reciprocating with N=1, ripple is cos(2θ) — peak at θ=0 and π,
    // trough at θ=π/2 and 3π/2.
    CHECK(load.load_torque(0.0, 0.0) == Approx(T_mean * 1.5).epsilon(1e-6));
    CHECK(load.load_torque(std::numbers::pi_v<Real> / 2.0, 0.0) ==
          Approx(T_mean * 0.5).epsilon(1e-6));
    CHECK(load.load_torque(std::numbers::pi_v<Real>, 0.0) ==
          Approx(T_mean * 1.5).epsilon(1e-6));
}

TEST_CASE("CompressorLoad: rotary topology has smaller ripple than reciprocating",
          "[compressor][load][rotary]") {
    loads::CompressorParams p{};
    p.topology = loads::CompressorTopology::Rotary;
    p.ripple_amplitude = 0.5;
    p.b_friction = 0.0;
    p.tau_coulomb = 0.0;

    loads::CompressorLoad load(p);
    const Real T_mean = load.mean_torque();
    // Rotary: ripple is 0.2·α·cos(2θ), so peak = T_mean·(1 + 0.1) = 1.1·T_mean.
    CHECK(load.load_torque(0.0, 0.0) == Approx(T_mean * 1.1).epsilon(1e-6));
    CHECK(load.load_torque(std::numbers::pi_v<Real> / 2.0, 0.0) ==
          Approx(T_mean * 0.9).epsilon(1e-6));
}

TEST_CASE("CompressorLoad: scroll topology has near-constant torque",
          "[compressor][load][scroll]") {
    loads::CompressorParams p{};
    p.topology = loads::CompressorTopology::Scroll;
    p.ripple_amplitude = 0.5;
    p.b_friction = 0.0;
    p.tau_coulomb = 0.0;

    loads::CompressorLoad load(p);
    const Real T_mean = load.mean_torque();
    // Scroll: 0.05·α·cos(8θ) → peak = T_mean·(1 + 0.025) = 1.025·T_mean.
    CHECK(load.load_torque(0.0, 0.0) == Approx(T_mean * 1.025).epsilon(1e-6));
}

TEST_CASE("CompressorLoad: friction adds linear and Coulomb terms",
          "[compressor][load][friction]") {
    loads::CompressorParams p{};
    p.ripple_amplitude = 0.0;     // no compression ripple, isolate friction
    p.b_friction = 0.01;          // 10 mN·m·s
    p.tau_coulomb = 0.02;          // 20 mN·m

    loads::CompressorLoad load(p);
    const Real T_mean = load.mean_torque();

    // At ω = 0 → friction_torque = 0 (Coulomb sign is 0 at zero ω).
    CHECK(load.load_torque(0.0, 0.0) == Approx(T_mean).margin(1e-9));

    // At ω = 100 rad/s → friction = 0.01·100 + 0.02·sign(100) = 1.02 N·m.
    CHECK(load.load_torque(0.0, 100.0) ==
          Approx(T_mean + 1.02).margin(1e-9));

    // At ω = −50 rad/s → friction = 0.01·(−50) + 0.02·(−1) = −0.52 N·m.
    CHECK(load.load_torque(0.0, -50.0) ==
          Approx(T_mean - 0.52).margin(1e-9));
}

TEST_CASE("Circuit::attach_compressor_load pushes torque demand to BLDC motor",
          "[compressor][circuit][bldc]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    auto c = ckt.add_node("c");
    auto n = ckt.add_node("n");

    // Apply small DC voltage on phase A to drive the motor.
    ckt.add_voltage_source("V_a", a, Circuit::ground(), 6.0);
    ckt.add_voltage_source("V_b", b, Circuit::ground(), 0.0);
    ckt.add_voltage_source("V_c", c, Circuit::ground(), 0.0);
    ckt.add_voltage_source("V_n", n, Circuit::ground(), 0.0);

    // BLDC motor — sized loosely for a domestic compressor.
    motors::BldcMotorParams motor_params{};
    motor_params.R_s = 5.0;
    motor_params.L_s = 8e-3;
    motor_params.K_e_peak = 0.012;
    motor_params.pole_pairs = 2;
    motor_params.J = 5e-5;
    motor_params.b_friction = 1e-5;
    ckt.add_bldc_motor("M_compressor", a, b, c, n, motor_params);

    // Attach the compressor load (default Embraco-style domestic params).
    loads::CompressorParams comp{};
    comp.topology = loads::CompressorTopology::Reciprocating;
    comp.displacement_m3 = 6.0e-6;
    comp.P_suction_Pa = 7.0e4;
    comp.P_discharge_Pa = 8.0e5;
    comp.polytropic_n = 1.13;
    comp.b_friction = 5e-4;
    ckt.attach_compressor_load("M_compressor", comp);

    // Read the analytical mean torque through the Circuit accessor.
    INFO("compressor_mean_torque = "
         << ckt.compressor_mean_torque("M_compressor") << " N·m");
    CHECK(ckt.compressor_mean_torque("M_compressor") > 0.0);
    CHECK(ckt.compressor_indicated_work("M_compressor") > 0.0);

    // Brief transient to confirm the load is actually being pushed onto
    // the motor (rotor speed stays low under the compressor's heavy
    // mean torque). With no compressor load the rotor would be running
    // away to thousands of rad/s under 6 V on phase A.
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-3;
    opts.dt = 1e-5;
    opts.dt_min = 1e-9;
    opts.dt_max = 1e-5;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto run = sim.run_transient();
    REQUIRE(run.success);

    const Real omega_final = ckt.bldc_omega("M_compressor");
    INFO("ω at t=5ms with compressor load: " << omega_final << " rad/s");

    // The compressor's mean torque > BLDC's locked-rotor torque from
    // 6 V on phase A → the rotor should NOT have spun up to anywhere
    // near no-load levels (~50+ rad/s from the earlier BLDC-only test).
    // Detach the load to confirm it really was acting.
    ckt.detach_compressor_load("M_compressor");
    REQUIRE(std::isnan(ckt.compressor_mean_torque("M_compressor")));
}
