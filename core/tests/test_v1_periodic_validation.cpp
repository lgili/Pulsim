#include <catch2/catch_test_macros.hpp>

#include "pulsim/v1/simulation.hpp"

#include <limits>
#include <string>

using namespace pulsim::v1;

namespace {

Circuit make_periodic_test_circuit() {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    const auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 10.0);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 1e-6);
    return circuit;
}

SimulationOptions make_periodic_test_options(const Circuit& circuit) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-4;
    opts.dt = 1e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 1e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    return opts;
}

}  // namespace

TEST_CASE("v1 periodic shooting validates initial state size", "[v1][steady][validation]") {
    Circuit circuit = make_periodic_test_circuit();
    Simulator sim(circuit, make_periodic_test_options(circuit));

    PeriodicSteadyStateOptions pss;
    pss.period = 50e-6;
    Vector invalid_x0 = Vector::Zero(static_cast<Index>(circuit.system_size() + 1));

    const auto result = sim.run_periodic_shooting(invalid_x0, pss);
    REQUIRE_FALSE(result.success);
    CHECK(result.message.find("size mismatch") != std::string::npos);
}

TEST_CASE("v1 harmonic balance validates finite initial state", "[v1][steady][validation]") {
    Circuit circuit = make_periodic_test_circuit();
    Simulator sim(circuit, make_periodic_test_options(circuit));

    HarmonicBalanceOptions hb;
    hb.period = 50e-6;
    hb.num_samples = 8;

    Vector invalid_x0 = Vector::Zero(static_cast<Index>(circuit.system_size()));
    invalid_x0[0] = std::numeric_limits<Real>::quiet_NaN();

    const auto result = sim.run_harmonic_balance(invalid_x0, hb);
    REQUIRE_FALSE(result.success);
    CHECK(result.message.find("non-finite") != std::string::npos);
}

TEST_CASE("v1 harmonic balance validates finite period", "[v1][steady][validation]") {
    Circuit circuit = make_periodic_test_circuit();
    Simulator sim(circuit, make_periodic_test_options(circuit));

    HarmonicBalanceOptions hb;
    hb.period = std::numeric_limits<Real>::quiet_NaN();
    hb.num_samples = 8;

    Vector x0 = Vector::Zero(static_cast<Index>(circuit.system_size()));
    const auto result = sim.run_harmonic_balance(x0, hb);
    REQUIRE_FALSE(result.success);
    CHECK(result.message.find("positive finite period") != std::string::npos);
}
