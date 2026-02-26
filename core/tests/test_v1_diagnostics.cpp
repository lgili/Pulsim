#include <catch2/catch_test_macros.hpp>

#include "pulsim/v1/simulation.hpp"

using namespace pulsim::v1;

namespace {

Circuit make_diag_circuit() {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("Rload", n_in, Circuit::ground(), 10.0);
    return circuit;
}

SimulationOptions make_diag_options(const Circuit& circuit) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 5e-6;
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

TEST_CASE("v1 transient sets typed diagnostic for invalid initial state", "[v1][diagnostic]") {
    Circuit circuit = make_diag_circuit();
    Simulator sim(circuit, make_diag_options(circuit));
    Vector x0 = Vector::Zero(static_cast<Index>(circuit.system_size() + 1));

    const auto result = sim.run_transient(x0);
    REQUIRE_FALSE(result.success);
    CHECK(result.diagnostic == SimulationDiagnosticCode::InvalidInitialState);
    CHECK(result.backend_telemetry.failure_reason == "invalid_initial_state");
}

TEST_CASE("v1 transient sets typed diagnostic for unsupported legacy backend", "[v1][diagnostic]") {
    Circuit circuit = make_diag_circuit();
    SimulationOptions opts = make_diag_options(circuit);
    opts.transient_backend = TransientBackendMode::Auto;
    Simulator sim(circuit, opts);
    Vector x0 = Vector::Zero(static_cast<Index>(circuit.system_size()));

    const auto result = sim.run_transient(x0);
    REQUIRE_FALSE(result.success);
    CHECK(result.diagnostic == SimulationDiagnosticCode::LegacyBackendUnsupported);
    CHECK(result.backend_telemetry.failure_reason == "legacy_backend_removed");
}

TEST_CASE("v1 periodic and HB validations set typed diagnostics", "[v1][diagnostic][steady]") {
    Circuit circuit = make_diag_circuit();
    Simulator sim(circuit, make_diag_options(circuit));
    Vector x0 = Vector::Zero(static_cast<Index>(circuit.system_size()));

    PeriodicSteadyStateOptions pss;
    pss.period = 0.0;
    const auto pss_result = sim.run_periodic_shooting(x0, pss);
    REQUIRE_FALSE(pss_result.success);
    CHECK(pss_result.diagnostic == SimulationDiagnosticCode::PeriodicInvalidPeriod);

    HarmonicBalanceOptions hb;
    hb.period = 0.0;
    const auto hb_result = sim.run_harmonic_balance(x0, hb);
    REQUIRE_FALSE(hb_result.success);
    CHECK(hb_result.diagnostic == SimulationDiagnosticCode::HarmonicInvalidPeriod);
}
