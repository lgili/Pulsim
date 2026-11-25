// add-immutable-circuit-ir group 4: Circuit YAML round trip (C++ side;
// the per-kind, example and harness coverage is test_circuit_yaml.py).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/yaml/emitter.hpp"

#include <stdexcept>
#include <string>

using pulsim::builder::CircuitBuilder;
using pulsim::ir::from_yaml;
using pulsim::ir::to_yaml;

TEST_CASE("Circuit YAML: same hashes, fixed point, pinned values", "[ir][yaml]") {
    CircuitBuilder b;
    b.add_voltage_source("V1", "in", "gnd", 10.0);
    b.add_resistor("R1", "in", "out", 1e3);
    b.add_capacitor("C1", "out", "gnd", 1e-6);
    const auto c = b.build();
    // FNV-1a over names and IEEE bits: equal on every platform (pinned
    // in test_circuit_yaml.py too).
    CHECK(c.structure_hash() == 0x57BF94FE7FA50967ULL);
    CHECK(c.value_hash() == 0x746D6010F3DC01E0ULL);

    const auto y = to_yaml(c);
    const auto c2 = from_yaml(y);
    CHECK(c2.structure_hash() == c.structure_hash());
    CHECK(c2.value_hash() == c.value_hash());
    CHECK(to_yaml(c2) == y);
    CHECK_THAT(y, Catch::Matchers::ContainsSubstring(
        R"({kind: resistor, name: "R1", pins: ["in", "out"], G: 0.001})"));
}

TEST_CASE("Circuit YAML: a branch-form diode, a table and an IC", "[ir][yaml]") {
    CircuitBuilder b;
    b.add_current_source("I", "gnd", "a", 1.0);
    b.add_saturable_inductor_table("L", "a", "gnd", {0.0, 1.0, 2.0}, {0.0, 1e-3, 1.5e-3});
    b.add_capacitor("C", "a", "gnd", 1e-6, 0.25);
    b.add_diode_branch("D", "a", "gnd", 0.0, 0.0, 0.7, 0.7);
    const auto c = b.build();
    const auto c2 = from_yaml(to_yaml(c));
    CHECK(c2.value_hash() == c.value_hash());
    CHECK(to_yaml(c2) == to_yaml(c));
    CHECK(c2.pool().is_branch_form(3));
}

TEST_CASE("Circuit YAML: a malformed document names device and key", "[ir][yaml]") {
    CHECK_THROWS_WITH(from_yaml("pulsim_circuit: 1\nnodes: [\"a\"]\ndevices:\n"
                                "  - {kind: resistor, name: \"R1\", pins: [\"a\", \"gnd\"], R: 5}\n"),
                      Catch::Matchers::ContainsSubstring("R1: unknown key \"R\""));
    CHECK_THROWS_AS(from_yaml("pulsim_circuit: [1"), std::invalid_argument);
}

// ---- group 8: scopes; the leftovers of slice 2 (bounds, one schema) ----

#include "pulsim/yaml/loader.hpp"

TEST_CASE("Scopes: lookup from the open scope outward; the tree is structure",
          "[ir][yaml][scope]") {
    CircuitBuilder b;
    b.param("g", 1.0);                       // shadowed below
    b.add_voltage_source("V", "a", "gnd", 1.0);
    const auto plain = b.build();
    CHECK(plain.scopes().empty());
    (void)b.open_scope("arm", "Arm");
    b.param("g", 500.0);
    (void)b.open_scope("sm0", "Cell");
    const auto q = b.param_expr("G2", "2*g");   // the arm's g
    CHECK(q.name == "arm/sm0/G2");
    CHECK(q.value == 1000.0);
    CHECK(b.scoped_name("R") == "arm/sm0/R");
    b.add_resistor(b.scoped_name("R"), "a", "gnd", 1.0);
    b.bind_param("arm/sm0/R.G", q.name);
    b.close_scope();
    b.close_scope();
    CHECK_THROWS_AS(b.close_scope(), std::logic_error);
    CHECK_THROWS_WITH(b.open_scope("arm", "Other"),
                      Catch::Matchers::ContainsSubstring("already used by an instance"));
    const auto c = b.build();
    REQUIRE(c.scopes().size() == 2);
    CHECK(c.scopes()[1].path == "arm/sm0");
    CHECK(c.scopes()[1].parent == 1);
    CHECK(c.scope_of("arm/sm0/R") == 2);
    CHECK(c.scope_of("arm/x") == 1);
    CHECK(c.scope_of("a") == 0);
    CHECK(c.lowered().get_value("arm/sm0/R.G") == 1000.0);
    // with_params re-evaluates the scoped expression; the root g drives nothing
    const std::pair<std::string_view, pulsim::Real> arm{"arm/g", 250.0};
    CHECK(c.with_params(std::span{&arm, 1}).lowered().get_value("arm/sm0/R.G") == 500.0);
    const std::pair<std::string_view, pulsim::Real> root{"g", 7.0};
    CHECK(c.with_params(std::span{&root, 1}).value_hash() != c.value_hash());
    CHECK(c.with_params(std::span{&root, 1}).lowered().get_value("arm/sm0/R.G") == 1000.0);
    // round trip: same tree, same hashes, expressions resolve again in scope
    const auto c2 = from_yaml(to_yaml(c));
    CHECK(c2.structure_hash() == c.structure_hash());
    CHECK(c2.value_hash() == c.value_hash());
    CHECK(to_yaml(c2) == to_yaml(c));
    CHECK(c2.with_params(std::span{&arm, 1}).lowered().get_value("arm/sm0/R.G") == 500.0);
}

TEST_CASE("from_yaml applies the authoring bounds, naming device and field",
          "[ir][yaml]") {
    auto doc = [](std::string_view dev) {
        return std::string{"pulsim_circuit: 1\nnodes: [\"a\"]\ndevices:\n  - "} +
               std::string{dev} + "\n";
    };
    CHECK_THROWS_WITH(from_yaml(doc(R"({kind: resistor, name: "R1", pins: ["a", "gnd"], G: -1})")),
                      Catch::Matchers::ContainsSubstring("R1: G must be a finite number > 0"));
    CHECK_THROWS_WITH(from_yaml(doc(R"({kind: inductor, name: "L1", pins: ["a", "gnd"], L: 0})")),
                      Catch::Matchers::ContainsSubstring("L1: L must be a finite number > 0"));
    CHECK_THROWS_WITH(
        from_yaml(doc(R"({kind: pwm_voltage_source, name: "P", pins: ["a", "gnd"], duty: 1.5})")),
        Catch::Matchers::ContainsSubstring("P: duty must be a number in [0, 1]"));
    // a branch-form switch keeps its own rule: g_off = 0 is the ideal open switch
    CircuitBuilder b;
    b.add_voltage_source("V", "a", "gnd", 1.0);
    b.add_diode_branch("D", "a", "gnd", 0.0, 0.0, 0.7, 0.7);
    CHECK(from_yaml(to_yaml(b.build())).value_hash() == b.build().value_hash());
}

TEST_CASE("yaml::load reads the emitter's documents and every registry kind",
          "[ir][yaml]") {
    CircuitBuilder b;
    b.add_voltage_source("V", "a", "gnd", 1.0);
    b.add_shockley_diode("D", "a", "gnd");
    const auto c = b.build();
    const auto loaded = pulsim::yaml::load_string(to_yaml(c));
    CHECK(loaded.builder.build().value_hash() == c.value_hash());
    // an authoring netlist naming a kind with no authoring type of its own
    const auto l2 = pulsim::yaml::load_string(
        "circuit:\n  devices:\n"
        "    - {type: voltage_source, name: V, from: a, to: gnd, V: 1.0}\n"
        "    - {type: shockley_diode, name: D, pins: [a, gnd]}\n");
    CHECK(l2.builder.build().value_hash() == c.value_hash());
    CHECK_THROWS_WITH(pulsim::yaml::load_string(
        "circuit:\n  devices:\n"
        "    - {type: lauritzen_diode, name: D, pins: [a, gnd], tau: -1}\n"),
        Catch::Matchers::ContainsSubstring("device 'D': LauritzenDiode: tau (carrier lifetime) must be"));
}

TEST_CASE("YAML subsystems: ports bind outer nodes, names are scoped",
          "[ir][yaml][scope]") {
    const auto l = pulsim::yaml::load_string(R"(
circuit:
  devices:
    - {type: voltage_source, name: V, from: in, to: gnd, V: 1.0}
  subsystems:
    - name: x1
      kind: RC
      ports: {p: in}
      parameters: [{name: R0, value: 2.0}]
      devices:
        - {type: resistor, name: R, from: p, to: m, R: R0}
        - {type: capacitor, name: C, from: m, to: gnd, C: 1.0e-6}
)");
    const auto c = l.builder.build();
    CHECK(c.lowered().branch_id_of("x1/R") == 1);
    CHECK(c.lowered().node_id_of("x1/m") >= 0);
    CHECK(c.lowered().get_value("x1/R.R") == 2.0);
    REQUIRE(c.scopes().size() == 1);
    CHECK(c.scopes()[0].kind == "RC");
}

TEST_CASE("Circuit YAML: a recorded composite and a hysteretic permeance",
          "[ir][yaml][composite][permeance]") {
    // add-mna-native-machines D23, add-permeance-magnetic-domain 4.1
    CircuitBuilder b;
    b.add_voltage_source("V", "a", "gnd", 10.0);
    b.add_inductor("M.la", "a", "M.emf", 1e-3);
    b.add_ideal_transformer("M.em", "w", "gnd", "M.emf", "gnd", 0.1);
    b.add_capacitor("M.j", "w", "gnd", 1e-5);
    b.record_composite("dc_machine", "M", "M.em", {"M.la", "M.em", "M.j"});
    pulsim::models::JilesAthertonParams ja;
    ja.Ms = 4.0e5; ja.a = 50.0; ja.alpha = 5e-5; ja.c = 0.2; ja.k = 30.0;
    b.add_winding("W", "a", "gnd", "m1", "gnd", 10.0);
    b.add_hysteretic_permeance("H", "m1", "gnd", 1e-4, 0.05, ja);
    CHECK(b.branch_id_of("H.w") >= 0);
    CHECK(b.pool().is_flux_node(b.node_id_of("m1")));

    const auto c = b.build();
    const auto y = to_yaml(c);
    CHECK_THAT(y, Catch::Matchers::ContainsSubstring(
        R"({kind: "dc_machine", name: "M", anchor: "M.em", members: ["M.la", "M.em", "M.j"]})"));
    const auto c2 = from_yaml(y);
    CHECK(c2.value_hash() == c.value_hash());
    CHECK(to_yaml(c2) == y);

    CHECK_THROWS_WITH(b.record_composite("dc_machine", "M", "M.em", {"M.em"}),
                      Catch::Matchers::ContainsSubstring("already recorded"));
    CHECK_THROWS_WITH(b.add_hysteretic_permeance("X", "m1", "gnd", 0.0, 0.05, ja),
                      Catch::Matchers::ContainsSubstring("area"));
}
