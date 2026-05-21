// Catch2 unit tests for Circuit::set_position / position_hint /
// position_hints — the kernel-side storage added in Phase 2 of
// `add-python-schematic-renderer`.
//
// Hints have no effect on simulation; the kernel just round-trips them
// through the parser → renderer pipeline. Tests cover:
//   - set/get round-trip for both `(layer, slot)` and `(x, y)` forms
//   - snapshot accessor is detached from later mutation
//   - missing-component query returns nullopt
//   - empty-hint rejection at the kernel boundary
//   - determinism across runs (byte-identical state from identical
//     construction sequences)

#include <catch2/catch_test_macros.hpp>

#include "pulsim/v1/runtime_circuit.hpp"

using namespace pulsim::v1;

TEST_CASE("position_hints: empty circuit has no hints",
          "[position_hints]") {
    Circuit ckt;
    REQUIRE(ckt.num_position_hints() == 0);
    REQUIRE(ckt.position_hints().empty());
}

TEST_CASE("position_hints: set/get a (layer, slot) hint",
          "[position_hints]") {
    Circuit ckt;
    ckt.set_position("Q1", /*layer=*/2, /*slot=*/1);

    REQUIRE(ckt.num_position_hints() == 1);
    auto hint = ckt.position_hint("Q1");
    REQUIRE(hint.has_value());
    REQUIRE(hint->layer.has_value());
    REQUIRE(*hint->layer == 2);
    REQUIRE(hint->slot.has_value());
    REQUIRE(*hint->slot == 1);
    REQUIRE_FALSE(hint->x.has_value());
    REQUIRE_FALSE(hint->y.has_value());
}

TEST_CASE("position_hints: set/get an absolute (x, y) hint",
          "[position_hints]") {
    Circuit ckt;
    ckt.set_position("Cout", /*layer=*/std::nullopt, /*slot=*/std::nullopt,
                     /*x=*/200.0, /*y=*/80.0);

    auto hint = ckt.position_hint("Cout");
    REQUIRE(hint.has_value());
    REQUIRE(hint->x.has_value());
    REQUIRE(*hint->x == 200.0);
    REQUIRE(hint->y.has_value());
    REQUIRE(*hint->y == 80.0);
    REQUIRE_FALSE(hint->layer.has_value());
    REQUIRE_FALSE(hint->slot.has_value());
}

TEST_CASE("position_hints: both (layer, slot) AND (x, y) can coexist",
          "[position_hints]") {
    // The kernel persists every set field; the renderer (downstream)
    // decides priority. We just confirm we don't silently drop one.
    Circuit ckt;
    ckt.set_position("M1", 3, 4, 120.0, 60.0);
    auto hint = ckt.position_hint("M1");
    REQUIRE(hint.has_value());
    REQUIRE(*hint->layer == 3);
    REQUIRE(*hint->slot == 4);
    REQUIRE(*hint->x == 120.0);
    REQUIRE(*hint->y == 60.0);
}

TEST_CASE("position_hints: querying an unhinted component returns nullopt",
          "[position_hints]") {
    Circuit ckt;
    ckt.set_position("R1", 0, 0);
    REQUIRE_FALSE(ckt.position_hint("never_set").has_value());
}

TEST_CASE("position_hints: re-setting replaces the previous hint wholesale",
          "[position_hints]") {
    Circuit ckt;
    ckt.set_position("R1", /*layer=*/0, /*slot=*/0);
    ckt.set_position("R1", /*layer=*/std::nullopt, /*slot=*/std::nullopt,
                     /*x=*/50.0, /*y=*/25.0);

    auto hint = ckt.position_hint("R1");
    REQUIRE(hint.has_value());
    // First call's (layer, slot) is gone — replaced, not merged.
    REQUIRE_FALSE(hint->layer.has_value());
    REQUIRE_FALSE(hint->slot.has_value());
    REQUIRE(*hint->x == 50.0);
    REQUIRE(*hint->y == 25.0);
}

TEST_CASE("position_hints: snapshot is detached from later mutation",
          "[position_hints]") {
    Circuit ckt;
    ckt.set_position("R1", 0, 0);
    auto snap = ckt.position_hints();
    REQUIRE(snap.size() == 1);
    REQUIRE(*snap.at("R1").layer == 0);

    ckt.set_position("R1", 5, 5);

    // Snapshot still shows the old values.
    REQUIRE(*snap.at("R1").layer == 0);
    // But the live circuit reflects the new ones.
    REQUIRE(*ckt.position_hint("R1")->layer == 5);
}

TEST_CASE("position_hints: empty hint is rejected",
          "[position_hints]") {
    Circuit ckt;
    REQUIRE_THROWS_AS(
        ckt.set_position("R1"),  // all four optionals default to nullopt
        std::invalid_argument);
    REQUIRE(ckt.num_position_hints() == 0);
}

TEST_CASE("position_hints: hints survive irrelevant Circuit mutations",
          "[position_hints]") {
    // Setting hints before adding devices is allowed (YAML parser
    // pattern). After adding devices the hints must still be queryable.
    Circuit ckt;
    auto a = ckt.add_node("a");
    ckt.set_position("R1", 0, 0);
    ckt.set_position("V1", /*layer=*/std::nullopt, /*slot=*/std::nullopt,
                     /*x=*/10.0, /*y=*/20.0);
    ckt.add_resistor("R1", a, ckt.ground(), 1000.0);
    ckt.add_voltage_source("V1", a, ckt.ground(), 5.0);

    REQUIRE(ckt.position_hint("R1").has_value());
    REQUIRE(*ckt.position_hint("R1")->layer == 0);
    REQUIRE(ckt.position_hint("V1").has_value());
    REQUIRE(*ckt.position_hint("V1")->x == 10.0);
}

TEST_CASE("position_hints: hints have no effect on num_components or topology",
          "[position_hints]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    ckt.add_resistor("R1", a, ckt.ground(), 100.0);
    const auto before = ckt.num_components();
    ckt.set_position("R1", 0, 0);
    REQUIRE(ckt.num_components() == before);  // hint isn't a device
    // The descriptor list is unchanged by setting a hint.
    auto descs = ckt.components();
    REQUIRE(descs.size() == 1);
    REQUIRE(descs[0].name == "R1");
}

TEST_CASE("position_hints: determinism — identical builds yield identical hints",
          "[position_hints]") {
    auto build = []() {
        Circuit c;
        c.set_position("A", 0, 0);
        c.set_position("B", 1, 0, std::nullopt, std::nullopt);
        c.set_position("C", std::nullopt, std::nullopt, 50.0, 75.0);
        return c;
    };
    auto c1 = build();
    auto c2 = build();
    auto s1 = c1.position_hints();
    auto s2 = c2.position_hints();
    REQUIRE(s1.size() == s2.size());
    for (const auto& [name, hint] : s1) {
        auto it = s2.find(name);
        REQUIRE(it != s2.end());
        REQUIRE(hint.layer == it->second.layer);
        REQUIRE(hint.slot == it->second.slot);
        REQUIRE(hint.x == it->second.x);
        REQUIRE(hint.y == it->second.y);
    }
}
