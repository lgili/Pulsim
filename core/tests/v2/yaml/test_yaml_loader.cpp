// =============================================================================
// Layer 8 — YAML circuit loader unit + integration tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/yaml/loader.hpp"

#include <stdexcept>
#include <string>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

// -----------------------------------------------------------------------------
// Round-trip tests: one per device type
// -----------------------------------------------------------------------------

TEST_CASE("YAML loader: voltage_source round-trip",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: voltage_source
      name: Vin
      from: n0
      to: gnd
      V: 5.0
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 1);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::VoltageSource);
    REQUIRE(loaded.builder.pool()
                .voltage_source_params(0)
                .V == Approx(5.0));
}

TEST_CASE("YAML loader: current_source round-trip",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: current_source
      name: Ibias
      from: n0
      to: gnd
      I: 0.01
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 1);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::CurrentSource);
    REQUIRE(loaded.builder.pool()
                .current_source_params(0)
                .I == Approx(0.01));
}

TEST_CASE("YAML loader: resistor round-trip (R in ohms → G = 1/R)",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: resistor
      name: R1
      from: a
      to: b
      R: 100.0
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 1);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::Resistor);
    REQUIRE(loaded.builder.pool().resistor_params(0).G ==
              Approx(1.0 / 100.0));
}

TEST_CASE("YAML loader: capacitor round-trip",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: capacitor
      name: C1
      from: a
      to: gnd
      C: 1e-6
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 1);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::Capacitor);
    REQUIRE(loaded.builder.pool().capacitor_params(0).C ==
              Approx(1e-6));
}

TEST_CASE("YAML loader: inductor round-trip",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: inductor
      name: L1
      from: a
      to: b
      L: 100e-6
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 1);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::Inductor);
    REQUIRE(loaded.builder.pool().inductor_params(0).L ==
              Approx(100e-6));
}

TEST_CASE("YAML loader: diode round-trip",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: diode
      name: D1
      anode: a
      cathode: c
      g_on: 1000.0
      g_off: 1e-9
      V_th: 0.7
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 1);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::Diode);
}

TEST_CASE("YAML loader: nonlinear_diode round-trip",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: nonlinear_diode
      name: D2
      anode: a
      cathode: c
      V_F0: 0.7
      R_d: 0.01
      G_off: 1e-9
      kappa: 20.0
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 1);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::NonlinearDiode);
}

TEST_CASE("YAML loader: mosfet round-trip",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: mosfet
      name: Q1
      drain: d
      source: s
      R_on: 2e-3
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 1);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::Switch);
    REQUIRE(loaded.builder.pool().switch_g_on(0) ==
              Approx(1.0 / 2e-3));
}

TEST_CASE("YAML loader: mosfet_with_body_diode adds 2 branches",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: mosfet_with_body_diode
      name: Q1
      drain: d
      source: s
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 2);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::Switch);
    REQUIRE(loaded.builder.pool().kind_of(1) ==
              DevicePool::StoredKind::Diode);
}

TEST_CASE("YAML loader: igbt round-trip",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: igbt
      name: T1
      collector: c
      emitter: e
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 1);
    REQUIRE(loaded.builder.pool().kind_of(0) ==
              DevicePool::StoredKind::Switch);
}

TEST_CASE("YAML loader: transformer round-trip",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: transformer
      name: T1
      p_from: p+
      p_to: p-
      s_from: s+
      s_to: s-
      L_p: 1e-3
      L_s: 4e-3
      k: 0.95
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 2);
    REQUIRE(loaded.builder.pool().transformer_couplings()
                .size() == 1);
    REQUIRE(loaded.builder.pool()
                .transformer_couplings()[0]
                .params.k == Approx(0.95));
}

// -----------------------------------------------------------------------------
// simulation: block round-trip
// -----------------------------------------------------------------------------

TEST_CASE("YAML loader: simulation: block populates SimulationOptions",
          "[v2][layer8][yaml]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: voltage_source
      name: Vin
      from: a
      to: gnd
      V: 5.0
simulation:
  t_start: 0.0
  t_end: 1e-3
  dt: 1e-7
  enable_newton_line_search: true
  enable_newton_lm: false
  max_newton_iterations: 100
)YAML";
    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.options.t_start == Approx(0.0));
    REQUIRE(loaded.options.t_end == Approx(1e-3));
    REQUIRE(loaded.options.dt == Approx(1e-7));
    REQUIRE(loaded.options.enable_newton_line_search == true);
    REQUIRE(loaded.options.enable_newton_lm == false);
    REQUIRE(loaded.options.max_newton_iterations ==
              Size{100});
}

// -----------------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------------

TEST_CASE("YAML loader: missing R on resistor throws with name",
          "[v2][layer8][yaml][errors]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: resistor
      name: R_load
      from: a
      to: b
)YAML";
    try {
        (void)yaml::load_string(yaml);
        FAIL("Expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        const std::string what{e.what()};
        REQUIRE(what.find("R_load") != std::string::npos);
        REQUIRE(what.find("'R'") != std::string::npos);
    }
}

TEST_CASE("YAML loader: unknown device type throws with the type name",
          "[v2][layer8][yaml][errors]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: thyristor_v999
      name: T1
      from: a
      to: b
)YAML";
    try {
        (void)yaml::load_string(yaml);
        FAIL("Expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        const std::string what{e.what()};
        REQUIRE(what.find("thyristor_v999") !=
                  std::string::npos);
    }
}

TEST_CASE("YAML loader: missing top-level circuit: throws",
          "[v2][layer8][yaml][errors]") {
    const std::string yaml = R"YAML(
simulation:
  t_end: 1e-3
)YAML";
    REQUIRE_THROWS_AS(yaml::load_string(yaml),
                       std::runtime_error);
}

// -----------------------------------------------------------------------------
// Integration: YAML-built circuit matches direct builder
// -----------------------------------------------------------------------------

TEST_CASE("YAML-built half-wave rectifier matches direct builder",
          "[v2][layer8][yaml][integration]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: voltage_source
      name: Vin
      from: n0
      to: gnd
      V: 0.0
    - type: diode
      name: D1
      anode: n0
      cathode: n1
      g_on: 1000.0
      g_off: 1e-9
      V_th: 0.0
    - type: resistor
      name: R_L
      from: n1
      to: gnd
      R: 10.0
)YAML";

    auto loaded = yaml::load_string(yaml);
    REQUIRE(loaded.builder.num_branches() == 3);

    // Build a direct version for comparison.
    builder::CircuitBuilder direct;
    direct.add_voltage_source("Vin", "n0", "gnd", 0.0)
          .add_diode("D1", "n0", "n1", 1000.0, 1e-9, 0.0)
          .add_resistor("R_L", "n1", "gnd", 10.0);

    REQUIRE(direct.num_branches() ==
              loaded.builder.num_branches());

    // Topology equivalence: same nodes, same per-branch
    // kinds.
    REQUIRE(direct.graph().num_nodes() ==
              loaded.builder.graph().num_nodes());
    for (Index b = 0;
         b < direct.graph().num_branches(); ++b) {
        REQUIRE(direct.pool().kind_of(b) ==
                  loaded.builder.pool().kind_of(b));
    }
}

TEST_CASE("YAML-built V_dc + R: DC solve gives v_node = V_dc",
          "[v2][layer8][yaml][integration]") {
    const std::string yaml = R"YAML(
circuit:
  devices:
    - type: voltage_source
      name: Vin
      from: n0
      to: gnd
      V: 5.0
    - type: resistor
      name: R1
      from: n0
      to: gnd
      R: 1.0
)YAML";
    auto loaded = yaml::load_string(yaml);

    PwlStateSpaceCache cache(
        loaded.builder.graph(), loaded.builder.pool());
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    Vector x = Vector::Zero(seg.state_size);
    Vector b_extra = Vector::Zero(seg.state_size);
    cache.solve(SwitchStateMask(0), b_extra, x);

    REQUIRE(x[0] == Approx(5.0).margin(1e-9));
}

TEST_CASE("YAML loader: malformed YAML throws cleanly",
          "[v2][layer8][yaml][errors]") {
    const std::string yaml = "this is :: not :: valid::: yaml";
    REQUIRE_THROWS_AS(yaml::load_string(yaml),
                       std::runtime_error);
}
