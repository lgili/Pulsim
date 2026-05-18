// consolidate-motors-and-three-phase Phase D follow-up — smoke that the
// `benchmarks/circuits/motor_bldc_six_step.yaml` netlist loads and runs
// against the new BldcMotorDevice. Pins the YAML pipeline against the
// real on-disk benchmark file rather than an inline string.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/parser/yaml_parser.hpp"
#include "pulsim/v1/simulation.hpp"

#include <filesystem>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("BLDC benchmark YAML loads and simulates",
          "[benchmark][bldc][yaml]") {
    // Resolve the YAML file. Ctest typically runs from build/, so search
    // a few candidate roots until we find the benchmark file.
    const std::filesystem::path candidates[] = {
        std::filesystem::current_path() / "benchmarks" / "circuits" / "motor_bldc_six_step.yaml",
        std::filesystem::current_path().parent_path() / "benchmarks" / "circuits" / "motor_bldc_six_step.yaml",
        std::filesystem::current_path().parent_path().parent_path() / "benchmarks" / "circuits" / "motor_bldc_six_step.yaml",
    };
    std::filesystem::path yaml_path;
    for (const auto& c : candidates) {
        if (std::filesystem::exists(c)) { yaml_path = c; break; }
    }
    INFO("CWD: " << std::filesystem::current_path());
    REQUIRE(!yaml_path.empty());
    REQUIRE(std::filesystem::exists(yaml_path));

    parser::YamlParser parser;
    auto [circuit, options] = parser.load(yaml_path.string());

    // The benchmark YAML carries non-component blocks (`benchmark:`, etc.)
    // that the parser doesn't strictly validate — surface only true errors.
    const auto& errs = parser.errors();
    int hard_errors = 0;
    for (const auto& e : errs) {
        // Tolerate "unknown root field" diagnostics for `benchmark` and
        // `simulation.uic` — those belong to the benchmark wrapper, not
        // to the simulation kernel.
        if (e.find("root.benchmark") == std::string::npos &&
            e.find("simulation.uic") == std::string::npos) {
            ++hard_errors;
            INFO("unexpected YAML error: " << e);
        }
    }
    REQUIRE(hard_errors == 0);

    // The BLDC device should be present in the circuit.
    REQUIRE(circuit.bldc_omega("M1") == Approx(0.0));

    // Run the transient — locked-rotor at theta=0 with 12 V on phase A
    // accelerates the rotor in the trapezoidal back-EMF direction.
    options.newton_options.num_nodes = circuit.num_nodes();
    options.newton_options.num_branches = circuit.num_branches();
    Simulator sim(circuit, options);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    // After the run, omega has departed from zero (the motor accelerated
    // under the applied voltage). Direction is implementation-defined
    // (depends on the back-EMF profile convention) — only assert
    // non-trivial spin-up.
    CHECK(std::abs(circuit.bldc_omega("M1")) > 0.1);
}
