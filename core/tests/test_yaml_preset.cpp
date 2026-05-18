// simplify-and-harden-numerical-surface — Phase 2 YAML parser tests.
//
// Verifies the `simulation.preset:` YAML key materializes the same
// profile as `SimulationOptions::from_preset(...)` and that explicit
// per-field overrides take precedence over the preset's defaults.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/parser/yaml_parser.hpp"
#include "pulsim/v1/simulation.hpp"

#include <string>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

[[nodiscard]] std::pair<Circuit, SimulationOptions>
load(const std::string& yaml, parser::YamlParser& parser_holder) {
    auto loaded = parser_holder.load_string(yaml);
    return {std::move(loaded.first), std::move(loaded.second)};
}

}  // namespace

TEST_CASE("YAML: simulation.preset = robust materializes Robust profile",
          "[yaml][preset][robust]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  preset: robust
  tstop: 1e-3
  dt: 1e-6
components:
  - { type: resistor,  name: R1, nodes: [a, 0], value: 1000.0 }
  - { type: capacitor, name: C1, nodes: [a, 0], value: 1e-6 }
)";
    parser::YamlParser parser;
    auto [ckt, opts] = load(yaml, parser);
    REQUIRE(parser.errors().empty());

    CHECK(opts.integrator == Integrator::TRBDF2);
    CHECK(opts.step_mode  == TransientStepMode::Variable);
    CHECK(opts.stiffness_config.enable);
    CHECK(opts.max_step_retries == 12);
    CHECK(opts.tstop == Approx(1e-3));
    CHECK(opts.dt    == Approx(1e-6));
}

TEST_CASE("YAML: preset = fast materializes Fast profile",
          "[yaml][preset][fast]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  preset: fast
  tstop: 1e-3
  dt: 1e-6
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
)";
    parser::YamlParser parser;
    auto [ckt, opts] = load(yaml, parser);
    REQUIRE(parser.errors().empty());

    CHECK(opts.integrator     == Integrator::Trapezoidal);
    CHECK(opts.switching_mode == SwitchingMode::Ideal);
    CHECK(opts.step_mode      == TransientStepMode::Fixed);
    CHECK_FALSE(opts.stiffness_config.enable);
}

TEST_CASE("YAML: preset = high_fidelity materializes HighFidelity profile",
          "[yaml][preset][highfidelity]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  preset: high_fidelity
  tstop: 1e-3
  dt: 1e-6
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
)";
    parser::YamlParser parser;
    auto [ckt, opts] = load(yaml, parser);
    REQUIRE(parser.errors().empty());

    CHECK(opts.integrator     == Integrator::TRBDF2);
    CHECK(opts.max_step_retries == 24);
    CHECK(opts.timestep_config.error_tolerance == Approx(5e-4));
    CHECK(opts.lte_config.voltage_tolerance    == Approx(5e-4));
}

TEST_CASE("YAML: preset is case-insensitive (Auto / AUTO / auto)",
          "[yaml][preset][parse]") {
    for (const char* spelling : {"Auto", "AUTO", "auto"}) {
        const std::string yaml = std::string{R"(
schema: pulsim-v1
version: 1
simulation:
  preset: )"} + spelling + R"(
  tstop: 1e-3
  dt: 1e-6
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
)";
        parser::YamlParser parser;
        auto [ckt, opts] = load(yaml, parser);
        INFO("preset spelling: " << spelling);
        REQUIRE(parser.errors().empty());
        // Auto == Robust profile today.
        CHECK(opts.integrator == Integrator::TRBDF2);
        CHECK(opts.stiffness_config.enable);
    }
}

TEST_CASE("YAML: explicit field override wins over preset",
          "[yaml][preset][override]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  preset: robust
  integrator: bdf1
  tstop: 1e-3
  dt: 1e-6
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
)";
    parser::YamlParser parser;
    auto [ckt, opts] = load(yaml, parser);
    REQUIRE(parser.errors().empty());

    // Override wins over Robust's TRBDF2.
    CHECK(opts.integrator == Integrator::BDF1);
    // Rest of the Robust profile still applies.
    CHECK(opts.stiffness_config.enable);
    CHECK(opts.max_step_retries == 12);
}

TEST_CASE("YAML: no preset preserves legacy raw defaults",
          "[yaml][preset][backcompat]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-6
components:
  - { type: resistor, name: R1, nodes: [a, 0], value: 1000.0 }
)";
    parser::YamlParser parser;
    auto [ckt, opts] = load(yaml, parser);
    REQUIRE(parser.errors().empty());

    // Raw default integrator is Trapezoidal (not TRBDF2 — that's Robust's
    // job). max_step_retries stays at 6 (Robust bumps it to 12).
    CHECK(opts.integrator       == Integrator::Trapezoidal);
    CHECK(opts.max_step_retries == 6);
    CHECK_FALSE(opts.enable_bdf_order_control);
}
