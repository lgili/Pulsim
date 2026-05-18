// consolidate-motors-and-three-phase Phase D — YAML parser smoke for the
// four motor device types (`dc_motor`, `pmsm`, `bldc_motor`,
// `induction_motor`). Confirms the YAML `type:` aliases land on the right
// `Circuit::add_*` builder and that named parameters propagate.

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

TEST_CASE("Phase D: YAML dc_motor parses into DcMotorDevice",
          "[consolidation][yaml][dc_motor]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: voltage_source, name: V1, nodes: [vin, 0], value: 12.0 }
  - { type: resistor, name: R_load, nodes: [vneg, 0], value: 1.0 }
  - { type: dc_motor, name: M1, nodes: [vin, vneg], R_a: 0.5,
      L_a: 1e-3, K_e: 0.05, K_t: 0.05, J: 1e-4, b: 1e-5,
      omega_init: 0.0 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());
    // Confirm initial state through the Circuit accessor (DC motor uses
    // `motor_omega` historically; the other three motors use the per-type
    // `<type>_omega` pattern).
    CHECK(circuit.motor_omega("M1") == Approx(0.0));
}

TEST_CASE("Phase D: YAML pmsm parses into PmsmDevice",
          "[consolidation][yaml][pmsm]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: voltage_source, name: Va, nodes: [a, 0], value: 0.0 }
  - { type: voltage_source, name: Vb, nodes: [b, 0], value: 0.0 }
  - { type: voltage_source, name: Vc, nodes: [c, 0], value: 0.0 }
  - { type: voltage_source, name: Vn, nodes: [n, 0], value: 0.0 }
  - { type: pmsm, name: M1, nodes: [a, b, c, n], Rs: 0.5,
      Ld: 1e-3, Lq: 1e-3, psi_pm: 0.05, pole_pairs: 4,
      J: 1e-3, b_friction: 1e-4, omega_init: 0.0 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());
    CHECK(circuit.pmsm_omega("M1") == Approx(0.0));
}

TEST_CASE("Phase D: YAML bldc_motor parses into BldcMotorDevice",
          "[consolidation][yaml][bldc_motor]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: voltage_source, name: Va, nodes: [a, 0], value: 0.0 }
  - { type: voltage_source, name: Vb, nodes: [b, 0], value: 0.0 }
  - { type: voltage_source, name: Vc, nodes: [c, 0], value: 0.0 }
  - { type: voltage_source, name: Vn, nodes: [n, 0], value: 0.0 }
  - { type: bldc_motor, name: M1, nodes: [a, b, c, n], R_s: 0.5,
      L_s: 1e-3, K_e_peak: 0.05, pole_pairs: 4,
      J: 1e-4, b_friction: 1e-5 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());
    CHECK(circuit.bldc_omega("M1") == Approx(0.0));
}

TEST_CASE("Phase D: YAML induction_motor parses into InductionMotorDevice",
          "[consolidation][yaml][induction_motor]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: voltage_source, name: Va, nodes: [a, 0], value: 0.0 }
  - { type: voltage_source, name: Vb, nodes: [b, 0], value: 0.0 }
  - { type: voltage_source, name: Vc, nodes: [c, 0], value: 0.0 }
  - { type: voltage_source, name: Vn, nodes: [n, 0], value: 0.0 }
  - { type: induction_motor, name: M1, nodes: [a, b, c, n],
      R_s: 1.0, R_r: 1.5, L_s: 0.15, L_r: 0.15, L_m: 0.14,
      pole_pairs: 2, J: 0.01, b_friction: 1e-3 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());
    CHECK(circuit.induction_omega("M1") == Approx(0.0));
}

// Item 3 of the deferred-items follow-up: zero-pin signal-domain devices.
// `mechanical` and `pmsm_foc` load from YAML without a `nodes:` array —
// they have no electrical pins and no MNA contribution.

TEST_CASE("Item 3: YAML mechanical parses into MechanicalDevice (no nodes)",
          "[consolidation][yaml][mechanical]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: mechanical, name: Shaft1,
      J: 1e-3, b_friction: 0.01, omega_init: 0.0,
      tau_load_const: 0.0 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());
    CHECK(circuit.mechanical_omega("Shaft1") == Approx(0.0));
}

TEST_CASE("Item 3: YAML pmsm_foc parses into PmsmFocDevice (no nodes)",
          "[consolidation][yaml][pmsm_foc]") {
    const std::string yaml = R"(
schema: pulsim-v1
version: 1
simulation:
  tstop: 1e-3
  dt: 1e-5
components:
  - { type: pmsm_foc, name: Ctrl1,
      Rs: 0.5, Ld: 1.5e-3, Lq: 1.5e-3,
      psi_pm: 0.05, pole_pairs: 4, J: 1e-3,
      bandwidth_hz: 1000.0, Vd_min: -50.0, Vd_max: 50.0,
      Vq_min: -50.0, Vq_max: 50.0 }
)";
    parser::YamlParser parser;
    const auto circuit = load_circuit(yaml, parser);
    REQUIRE(parser.errors().empty());
    CHECK(circuit.pmsm_foc_vd_ref("Ctrl1") == Approx(0.0));
    CHECK(circuit.pmsm_foc_vq_ref("Ctrl1") == Approx(0.0));
}
