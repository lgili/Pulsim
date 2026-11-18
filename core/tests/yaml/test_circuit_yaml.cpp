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
