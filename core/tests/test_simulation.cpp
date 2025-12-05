#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "spicelab/simulation.hpp"
#include <cmath>

using namespace spicelab;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

TEST_CASE("DC operating point", "[simulation]") {
    SECTION("Resistive divider") {
        Circuit circuit;
        circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
        circuit.add_resistor("R1", "in", "out", 1000.0);
        circuit.add_resistor("R2", "out", "0", 1000.0);

        Simulator sim(circuit);
        auto result = sim.dc_operating_point();

        REQUIRE(result.status == SolverStatus::Success);

        // V(in) = 10V
        CHECK_THAT(result.x(0), WithinRel(10.0, 1e-6));
        // V(out) = 5V
        CHECK_THAT(result.x(1), WithinRel(5.0, 1e-6));
    }

    SECTION("Wheatstone bridge") {
        // Classic Wheatstone bridge with balanced resistors
        Circuit circuit;
        circuit.add_voltage_source("V1", "vcc", "0", DCWaveform{10.0});
        circuit.add_resistor("R1", "vcc", "a", 1000.0);
        circuit.add_resistor("R2", "vcc", "b", 1000.0);
        circuit.add_resistor("R3", "a", "0", 1000.0);
        circuit.add_resistor("R4", "b", "0", 1000.0);
        circuit.add_resistor("R5", "a", "b", 10000.0);  // Bridge resistor

        Simulator sim(circuit);
        auto result = sim.dc_operating_point();

        REQUIRE(result.status == SolverStatus::Success);

        // Balanced bridge: V(a) = V(b) = 5V
        Index idx_a = circuit.node_index("a");
        Index idx_b = circuit.node_index("b");
        CHECK_THAT(result.x(idx_a), WithinRel(5.0, 1e-6));
        CHECK_THAT(result.x(idx_b), WithinRel(5.0, 1e-6));
    }
}

TEST_CASE("RC transient simulation", "[simulation]") {
    // RC circuit with step input
    // V(out) = Vin * (1 - exp(-t/tau)) where tau = R*C
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "out", 1000.0);  // 1k
    circuit.add_capacitor("C1", "out", "0", 1e-6);    // 1uF

    Real tau = 1000.0 * 1e-6;  // 1ms

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-3;  // 5 time constants
    opts.dt = 1e-6;
    opts.use_ic = true;  // Start with initial conditions (capacitor at 0V) for step response

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
    REQUIRE(result.time.size() > 10);

    // Check voltage at various time constants
    Index out_idx = circuit.node_index("out");

    for (size_t i = 0; i < result.time.size(); ++i) {
        Real t = result.time[i];
        Real v_expected = 5.0 * (1.0 - std::exp(-t / tau));
        Real v_actual = result.data[i](out_idx);

        // Allow 5% error due to discretization
        CHECK_THAT(v_actual, WithinAbs(v_expected, 0.25));
    }

    // At t = 5*tau, should be at ~99.3% of final value
    Real final_v = result.data.back()(out_idx);
    CHECK_THAT(final_v, WithinAbs(5.0 * 0.993, 0.1));
}

TEST_CASE("RL transient simulation", "[simulation]") {
    // RL circuit with step input
    // I(L) = Vin/R * (1 - exp(-t/tau)) where tau = L/R
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "out", 100.0);   // 100 ohm
    circuit.add_inductor("L1", "out", "0", 10e-3);    // 10mH

    Real tau = 10e-3 / 100.0;  // 0.1ms
    Real I_final = 10.0 / 100.0;  // 100mA

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 0.5e-3;  // 5 time constants
    opts.dt = 1e-6;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    // Check inductor current at final time
    Index i_L_idx = circuit.node_count() + 1;  // Second branch (after V1)
    Real final_i = result.data.back()(i_L_idx);
    Real expected_i = I_final * (1.0 - std::exp(-opts.tstop / tau));

    CHECK_THAT(final_i, WithinAbs(expected_i, 0.005));
}

TEST_CASE("RLC transient simulation", "[simulation]") {
    // Underdamped RLC circuit
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{10.0});
    circuit.add_resistor("R1", "in", "n1", 10.0);     // 10 ohm
    circuit.add_inductor("L1", "n1", "out", 1e-3);    // 1mH
    circuit.add_capacitor("C1", "out", "0", 10e-6);   // 10uF

    // Natural frequency: w0 = 1/sqrt(LC) = 10000 rad/s
    // Damping ratio: zeta = R/2 * sqrt(C/L) = 0.5 (underdamped)

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 2e-3;
    opts.dt = 1e-6;
    opts.use_ic = true;  // Start from zero for step response (observe oscillation)

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);

    Index out_idx = circuit.node_index("out");

    // Check that output shows oscillation (crosses steady state)
    Real v_ss = 10.0;  // Steady state voltage
    bool crossed_above = false;
    bool crossed_below = false;

    for (const auto& data : result.data) {
        Real v = data(out_idx);
        if (v > v_ss * 1.05) crossed_above = true;
        // Initially below steady state
    }

    // Underdamped should overshoot
    CHECK(crossed_above);
}

TEST_CASE("Pulse source simulation", "[simulation]") {
    Circuit circuit;
    PulseWaveform pulse{0.0, 5.0, 0.0, 1e-9, 1e-9, 0.5e-3, 1e-3};
    circuit.add_voltage_source("V1", "in", "0", pulse);
    circuit.add_resistor("R1", "in", "out", 1000.0);
    circuit.add_capacitor("C1", "out", "0", 1e-6);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 3e-3;  // 3 periods
    opts.dt = 1e-6;
    opts.dtmax = 10e-6;  // Limit max timestep to ensure enough steps
    opts.use_ic = true;

    Simulator sim(circuit, opts);
    auto result = sim.run_transient();

    REQUIRE(result.final_status == SolverStatus::Success);
    REQUIRE(result.time.size() > 100);

    // Check that we have reasonable number of timesteps
    // With dtmax=10us and tstop=3ms, we need at least 300 steps
    CHECK(result.total_steps > 100);
}

TEST_CASE("Simulation with callback", "[simulation]") {
    Circuit circuit;
    circuit.add_voltage_source("V1", "in", "0", DCWaveform{5.0});
    circuit.add_resistor("R1", "in", "0", 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-3;
    opts.dt = 1e-4;

    Simulator sim(circuit, opts);

    int callback_count = 0;
    Real last_time = -1.0;

    auto callback = [&](Real time, const Vector& state) {
        CHECK(time > last_time);
        CHECK(state.size() == 2);  // 1 node + 1 branch
        last_time = time;
        callback_count++;
    };

    auto result = sim.run_transient(callback);

    REQUIRE(result.final_status == SolverStatus::Success);
    CHECK(callback_count == static_cast<int>(result.time.size()));
}
