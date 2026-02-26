#include <catch2/catch_test_macros.hpp>

#include "pulsim/v1/simulation.hpp"

#include <limits>
#include <string>

using namespace pulsim::v1;

namespace {

Circuit make_validation_circuit() {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("Rload", n_in, Circuit::ground(), 10.0);
    return circuit;
}

SimulationOptions make_fixed_options(const Circuit& circuit) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 2e-6;
    opts.dt = 2e-7;
    opts.dt_min = 1e-9;
    opts.dt_max = 2e-7;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    return opts;
}

}  // namespace

TEST_CASE("v1 transient rejects initial-state size mismatch", "[v1][safety][validation]") {
    Circuit circuit = make_validation_circuit();
    const SimulationOptions opts = make_fixed_options(circuit);
    Simulator sim(circuit, opts);

    Vector x0 = Vector::Zero(static_cast<Index>(circuit.system_size() + 1));
    const auto result = sim.run_transient(x0);

    REQUIRE_FALSE(result.success);
    CHECK(result.final_status == SolverStatus::NumericalError);
    CHECK(result.message.find("size mismatch") != std::string::npos);
    CHECK(result.backend_telemetry.failure_reason == "invalid_initial_state");
}

TEST_CASE("v1 transient rejects non-finite initial state", "[v1][safety][validation]") {
    Circuit circuit = make_validation_circuit();
    const SimulationOptions opts = make_fixed_options(circuit);
    Simulator sim(circuit, opts);

    Vector x0 = Vector::Zero(static_cast<Index>(circuit.system_size()));
    x0[0] = std::numeric_limits<Real>::quiet_NaN();

    const auto result = sim.run_transient(x0);
    REQUIRE_FALSE(result.success);
    CHECK(result.final_status == SolverStatus::NumericalError);
    CHECK(result.message.find("non-finite") != std::string::npos);
    CHECK(result.backend_telemetry.failure_reason == "invalid_initial_state");
}

TEST_CASE("v1 transient rejects invalid timestep bounds", "[v1][safety][validation]") {
    Circuit circuit = make_validation_circuit();
    SimulationOptions opts = make_fixed_options(circuit);
    opts.dt_min = 0.0;
    Simulator sim(circuit, opts);

    Vector x0 = Vector::Zero(static_cast<Index>(circuit.system_size()));

    const auto result = sim.run_transient(x0);
    REQUIRE_FALSE(result.success);
    CHECK(result.final_status == SolverStatus::NumericalError);
    CHECK(result.message.find("timestep") != std::string::npos);
    CHECK(result.backend_telemetry.failure_reason == "invalid_timestep");
}
