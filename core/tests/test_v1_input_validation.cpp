#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/parser/yaml_parser.hpp"
#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>

using namespace pulsim::v1;
using Catch::Approx;

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

TEST_CASE("v1 transient rejects invalid time window", "[v1][safety][validation]") {
    Circuit circuit = make_validation_circuit();
    SimulationOptions opts = make_fixed_options(circuit);
    opts.tstop = opts.tstart - 1e-9;
    Simulator sim(circuit, opts);

    Vector x0 = Vector::Zero(static_cast<Index>(circuit.system_size()));
    const auto result = sim.run_transient(x0);

    REQUIRE_FALSE(result.success);
    CHECK(result.diagnostic == SimulationDiagnosticCode::InvalidTimeWindow);
    CHECK(result.final_status == SolverStatus::NumericalError);
    CHECK(result.backend_telemetry.failure_reason == "invalid_time_window");
}

TEST_CASE("v1 parser flags malformed netlist topology with coded diagnostics",
          "[v1][safety][yaml][validation]") {
    const std::string yaml = R"(schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-6
  dt: 1e-7
components:
  - type: resistor
    name: Rbad
    nodes: [in]
    value: 10
)";

    parser::YamlParser parser;
    parser.load_string(yaml);

    REQUIRE_FALSE(parser.errors().empty());
    CHECK(std::any_of(parser.errors().begin(), parser.errors().end(),
                      [](const std::string& err) {
                          return err.find("PULSIM_YAML_E_PIN_COUNT") != std::string::npos;
                      }));
}

TEST_CASE("v1 transient contains hard nonlinear failure with deterministic diagnostics",
          "[v1][safety][containment]") {
    Circuit circuit;
    const auto n_in = circuit.add_node("in");
    const auto n_mid = circuit.add_node("mid");

    circuit.add_voltage_source("Vin", n_in, Circuit::ground(), 50.0);
    circuit.add_diode("D1", n_in, n_mid, 350.0, 1e-9);
    circuit.add_resistor("Rload", n_mid, Circuit::ground(), 0.5);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-6;
    opts.dt = 2e-7;
    opts.dt_min = opts.dt;
    opts.dt_max = opts.dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.max_step_retries = 0;
    opts.newton_options.max_iterations = 0;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    opts.model_regularization.enable_auto = false;
    opts.linear_solver.allow_fallback = false;

    Simulator sim(circuit, opts);
    Vector x0 = Vector::Zero(static_cast<Index>(circuit.system_size()));

    const auto result = sim.run_transient(x0);
    REQUIRE_FALSE(result.success);
    CHECK(result.final_status == SolverStatus::MaxIterationsReached);
    CHECK(result.diagnostic == SimulationDiagnosticCode::TransientStepFailure);
    CHECK(result.backend_telemetry.failure_reason == "transient_step_failure");
    REQUIRE_FALSE(result.fallback_trace.empty());
    CHECK(result.fallback_trace.back().reason == FallbackReasonCode::MaxRetriesExceeded);
}

TEST_CASE("v1 parser accepts basic control aliases and executes channels",
                    "[v1][yaml][control][aliases]") {
                const std::string yaml =
                        "schema: pulsim-v1\n"
                        "version: 1\n"
                        "simulation:\n"
                        "  tstop: 1e-4\n"
                        "  dt: 1e-5\n"
                        "components:\n"
                        "  - type: voltage_source\n"
                        "    name: VIN\n"
                        "    nodes:\n"
                        "      - in\n"
                        "      - \"0\"\n"
                        "    value: 2.0\n"
                        "  - type: voltage_source\n"
                        "    name: VREF\n"
                        "    nodes:\n"
                        "      - ref\n"
                        "      - \"0\"\n"
                        "    value: 1.0\n"
                        "  - type: gain\n"
                        "    name: G1\n"
                        "    nodes:\n"
                        "      - in\n"
                        "      - \"0\"\n"
                        "    gain: 3.0\n"
                        "  - type: sum\n"
                        "    name: SUM1\n"
                        "    nodes:\n"
                        "      - in\n"
                        "      - ref\n"
                        "  - type: subtraction\n"
                        "    name: SUB1\n"
                        "    nodes:\n"
                        "      - in\n"
                        "      - ref\n"
                        "  - type: pi\n"
                        "    name: PI1\n"
                        "    nodes:\n"
                        "      - ref\n"
                        "      - in\n"
                        "      - ctrl\n"
                        "    kp: 1.0\n"
                        "    ki: 0.0\n"
                        "    output_min: -10.0\n"
                        "    output_max: 10.0\n"
                        "  - type: pwm\n"
                        "    name: PWM1\n"
                        "    nodes:\n"
                        "      - ctrl\n"
                        "    frequency: 1000\n"
                        "    duty: 0.25\n";

        parser::YamlParser parser;
        auto [circuit, _options] = parser.load_string(yaml);

        std::ostringstream parser_errors;
        parser_errors << "YAML errors count=" << parser.errors().size();
        for (const auto& err : parser.errors()) {
            parser_errors << "\n" << err;
        }
        INFO(parser_errors.str());
        REQUIRE(parser.errors().empty());
        CHECK(circuit.num_virtual_components() >= 5);

        Vector x = circuit.initial_state();
        if (x.size() != circuit.system_size()) {
                x = Vector::Zero(static_cast<Index>(circuit.system_size()));
        }

        const auto step = circuit.execute_mixed_domain_step(x, 1e-4);
        REQUIRE(step.channel_values.contains("G1"));
        REQUIRE(step.channel_values.contains("SUM1"));
        REQUIRE(step.channel_values.contains("SUB1"));
        REQUIRE(step.channel_values.contains("PI1"));
        REQUIRE(step.channel_values.contains("PWM1"));
        REQUIRE(step.channel_values.contains("PWM1.duty"));

        CHECK(step.channel_values.at("G1") == Approx(6.0).margin(1e-12));
        CHECK(step.channel_values.at("SUM1") == Approx(3.0).margin(1e-12));
        CHECK(step.channel_values.at("SUB1") == Approx(1.0).margin(1e-12));
        CHECK(step.channel_values.at("PI1") == Approx(-1.0).margin(1e-12));
        CHECK(step.channel_values.at("PWM1.duty") == Approx(0.25).margin(1e-12));
        CHECK(step.channel_values.at("PWM1") == Approx(1.0).margin(1e-12));
}
