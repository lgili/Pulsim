// compressor-models follow-up: YAML parser smoke tests for the
// `single_phase_induction_motor` and `compressor_load` component types.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/parser/yaml_parser.hpp"
#include "pulsim/v1/simulation.hpp"

#include <string>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

[[nodiscard]] Circuit load_circuit(const std::string& yaml,
                                   parser::YamlParser& parser_holder) {
    auto loaded = parser_holder.load_string(yaml);
    return std::move(loaded.first);
}

}  // namespace

TEST_CASE("YAML: single_phase_induction_motor parses into Circuit",
          "[yaml][single_phase_im][compressor]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: voltage_source, name: V_line, nodes: [line, 0], value: 311.0 }
  - { type: single_phase_induction_motor, name: M_cc,
      nodes: [line, 0],
      R_s_main: 10.0, L_s_main: 50e-3,
      R_s_aux: 20.0,  L_s_aux: 80e-3,
      C_run: 4e-6,
      R_r: 8.0, L_r: 55e-3, L_m: 50e-3,
      pole_pairs: 2, J: 1e-4, b_friction: 1e-4,
      friction_coulomb: 0.05 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());

    CHECK(circuit.single_phase_im_omega("M_cc") == Approx(0.0));
    CHECK(circuit.single_phase_im_V_cap("M_cc") == Approx(0.0));
    CHECK(circuit.single_phase_im_i_main("M_cc") == Approx(0.0));
}

TEST_CASE("YAML: compressor_load attaches to a motor by name",
          "[yaml][compressor_load]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: voltage_source, name: V_a, nodes: [a, 0], value: 6.0 }
  - { type: voltage_source, name: V_b, nodes: [b, 0], value: 0.0 }
  - { type: voltage_source, name: V_c, nodes: [c, 0], value: 0.0 }
  - { type: voltage_source, name: V_n, nodes: [n, 0], value: 0.0 }
  - { type: bldc_motor, name: M_compressor,
      nodes: [a, b, c, n],
      R_s: 5.0, L_s: 8e-3, K_e_peak: 0.012,
      pole_pairs: 2, J: 5e-5, b_friction: 1e-5 }
  - { type: compressor_load, name: COMP1,
      motor: M_compressor,
      refrigerant: R600a,
      topology: Reciprocating,
      displacement_m3: 6.0e-6,
      P_suction_Pa: 7.0e4,
      P_discharge_Pa: 8.0e5,
      polytropic_n: 1.13 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());

    // The compressor_load got attached to "M_compressor".
    const Real tau_mean = circuit.compressor_mean_torque("M_compressor");
    INFO("compressor mean torque = " << tau_mean << " N·m");
    CHECK(tau_mean > 0.0);
    CHECK(circuit.compressor_indicated_work("M_compressor") > 0.0);
}

TEST_CASE("YAML: compressor_load defaults to R600a when no refrigerant given",
          "[yaml][compressor_load][refrigerant]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: voltage_source, name: V_a, nodes: [a, 0], value: 6.0 }
  - { type: voltage_source, name: V_b, nodes: [b, 0], value: 0.0 }
  - { type: voltage_source, name: V_c, nodes: [c, 0], value: 0.0 }
  - { type: voltage_source, name: V_n, nodes: [n, 0], value: 0.0 }
  - { type: bldc_motor, name: M1,
      nodes: [a, b, c, n],
      R_s: 5.0, L_s: 8e-3, K_e_peak: 0.012,
      pole_pairs: 2, J: 5e-5, b_friction: 1e-5 }
  - { type: compressor_load, name: COMP1,
      motor: M1,
      displacement_m3: 6.0e-6 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());

    // R600a defaults: P_suction = 0.59 bar, P_discharge = 5.30 bar,
    // polytropic_n = 1.13. Just check the result is a sane positive
    // mean torque — the refrigerant table tests assert exact values.
    CHECK(circuit.compressor_mean_torque("M1") > 0.0);
}

TEST_CASE("YAML: compressor_load rejects missing motor: field",
          "[yaml][compressor_load][error]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: compressor_load, name: COMP1,
      topology: Reciprocating,
      displacement_m3: 6.0e-6 }
)";
    parser::YamlParser parser;
    auto loaded = parser.load_string(yaml);
    REQUIRE_FALSE(parser.errors().empty());
}

TEST_CASE("YAML: compressor_load with refrigerant alias swap",
          "[yaml][compressor_load][refrigerant]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: voltage_source, name: V_a, nodes: [a, 0], value: 6.0 }
  - { type: voltage_source, name: V_b, nodes: [b, 0], value: 0.0 }
  - { type: voltage_source, name: V_c, nodes: [c, 0], value: 0.0 }
  - { type: voltage_source, name: V_n, nodes: [n, 0], value: 0.0 }
  - { type: bldc_motor, name: M_R290,
      nodes: [a, b, c, n],
      R_s: 5.0, L_s: 8e-3, K_e_peak: 0.012,
      pole_pairs: 2, J: 5e-5, b_friction: 1e-5 }
  - { type: refrigeration_compressor, name: COMP1,
      motor: M_R290,
      refrigerant: R290,
      topology: rotary,
      displacement_m3: 12.0e-6 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());

    // R290 (propane) gives higher discharge pressure than R600a, so
    // mean torque should be larger than the R600a default case for the
    // same displacement.
    CHECK(circuit.compressor_mean_torque("M_R290") > 0.0);
}
