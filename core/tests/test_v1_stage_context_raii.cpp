#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

SimulationOptions make_fixed_options(const Circuit& circuit, Integrator integrator) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 40e-6;
    opts.dt = 1e-6;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.integrator = integrator;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    return opts;
}

}  // namespace

TEST_CASE("v1 stage context does not leak across runs/method switches", "[v1][raii][integration]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    const auto n_out = circuit.add_node("out");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 5.0);
    circuit.add_resistor("R1", n_in, n_out, 100.0);
    circuit.add_capacitor("C1", n_out, Circuit::ground(), 10e-6);

    const SimulationOptions trbdf2_opts = make_fixed_options(circuit, Integrator::TRBDF2);
    const SimulationOptions bdf1_opts = make_fixed_options(circuit, Integrator::BDF1);

    Simulator reused_sim(circuit, trbdf2_opts);
    const auto warmup = reused_sim.run_transient();
    REQUIRE(warmup.success);

    reused_sim.set_options(bdf1_opts);
    const auto reused_result = reused_sim.run_transient();
    REQUIRE(reused_result.success);
    REQUIRE_FALSE(reused_result.states.empty());

    Simulator fresh_sim(circuit, bdf1_opts);
    const auto fresh_result = fresh_sim.run_transient();
    REQUIRE(fresh_result.success);
    REQUIRE_FALSE(fresh_result.states.empty());

    const Real reused_final = reused_result.states.back()[n_out];
    const Real fresh_final = fresh_result.states.back()[n_out];
    REQUIRE(std::isfinite(reused_final));
    REQUIRE(std::isfinite(fresh_final));
    CHECK(reused_final == Approx(fresh_final).margin(1e-9));
}
