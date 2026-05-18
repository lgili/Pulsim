// =============================================================================
// Test: Circuit component introspection (add-schematic-rendering Phase 1)
// =============================================================================
//
// Validates `Circuit::components()`, `Circuit::num_components()`, and
// `Circuit::node_position_hint()`. These accessors are the data layer for
// the schematic-rendering capability — they expose what was wired without
// requiring callers to scan the parallel `devices_` / `connections_`
// storage themselves. The contract is:
//
//   - One descriptor per `add_*` call, in insertion order.
//   - Stable canonical kind strings matching the YAML `type:` field.
//   - Best-effort param map: R/L/C/V/I populated for primary device types.
//   - Deterministic — same construction sequence -> same descriptors.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/runtime_circuit.hpp"

using namespace pulsim::v1;
using Catch::Approx;

// -----------------------------------------------------------------------------
// components(): canonical kinds + params for primary types
// -----------------------------------------------------------------------------

TEST_CASE("components: empty circuit returns empty list",
          "[component_introspection]") {
    Circuit ckt;
    REQUIRE(ckt.num_components() == 0);
    REQUIRE(ckt.components().empty());
}

TEST_CASE("components: passive RC + V source produces 3 descriptors in order",
          "[component_introspection]") {
    Circuit ckt;
    const Index n_in  = ckt.add_node("in");
    const Index n_out = ckt.add_node("out");
    const Index gnd   = ckt.ground();

    ckt.add_voltage_source("V1", n_in, gnd, 12.0);
    ckt.add_resistor("R1", n_in, n_out, 1000.0);
    ckt.add_capacitor("C1", n_out, gnd, 1e-6, 0.0);

    REQUIRE(ckt.num_components() == 3);
    const auto comps = ckt.components();
    REQUIRE(comps.size() == 3);

    // Insertion order
    REQUIRE(comps[0].name == "V1");
    REQUIRE(comps[1].name == "R1");
    REQUIRE(comps[2].name == "C1");

    // Canonical kinds
    REQUIRE(comps[0].kind == "voltage_source");
    REQUIRE(comps[1].kind == "resistor");
    REQUIRE(comps[2].kind == "capacitor");

    // Pin order preserved
    REQUIRE(comps[0].nodes == std::vector<Index>{n_in, gnd});
    REQUIRE(comps[1].nodes == std::vector<Index>{n_in, n_out});
    REQUIRE(comps[2].nodes == std::vector<Index>{n_out, gnd});

    // Params populated for primary types
    REQUIRE(comps[0].params.at("V") == Approx(12.0));
    REQUIRE(comps[1].params.at("R") == Approx(1000.0));
    REQUIRE(comps[2].params.at("C") == Approx(1e-6));
}

TEST_CASE("components: inductor + current source populate L and I params",
          "[component_introspection]") {
    Circuit ckt;
    const Index n1  = ckt.add_node("n1");
    const Index n2  = ckt.add_node("n2");
    const Index gnd = ckt.ground();

    ckt.add_inductor("L1", n1, n2, 100e-6, 0.0);
    ckt.add_current_source("I1", n1, gnd, 0.5);

    const auto comps = ckt.components();
    REQUIRE(comps.size() == 2);
    REQUIRE(comps[0].kind == "inductor");
    REQUIRE(comps[0].params.at("L") == Approx(100e-6));
    REQUIRE(comps[1].kind == "current_source");
    REQUIRE(comps[1].params.at("I") == Approx(0.5));
}

TEST_CASE("components: switching devices report canonical kind, params empty",
          "[component_introspection]") {
    Circuit ckt;
    const Index a    = ckt.add_node("a");
    const Index b    = ckt.add_node("b");
    const Index ctrl = ckt.add_node("ctrl");
    const Index gnd  = ckt.ground();

    ckt.add_diode("D1", a, b);
    ckt.add_switch("S1", a, b, false);
    ckt.add_vcswitch("S2", ctrl, a, b);

    const auto comps = ckt.components();
    REQUIRE(comps.size() == 3);
    REQUIRE(comps[0].kind == "diode");
    REQUIRE(comps[1].kind == "switch");
    REQUIRE(comps[2].kind == "vcswitch");

    // 2-terminal vs 3-terminal pin counts
    REQUIRE(comps[0].nodes.size() == 2);
    REQUIRE(comps[1].nodes.size() == 2);
    REQUIRE(comps[2].nodes.size() == 3);
    REQUIRE(comps[2].nodes[0] == ctrl);

    (void)gnd;
}

// -----------------------------------------------------------------------------
// Determinism — same construction -> byte-identical descriptor list
// -----------------------------------------------------------------------------

TEST_CASE("components: same construction sequence yields identical list",
          "[component_introspection][determinism]") {
    auto build = [] {
        Circuit ckt;
        const Index a = ckt.add_node("a");
        const Index b = ckt.add_node("b");
        ckt.add_voltage_source("V1", a, ckt.ground(), 5.0);
        ckt.add_resistor("R1", a, b, 100.0);
        ckt.add_capacitor("C1", b, ckt.ground(), 1e-9, 0.0);
        return ckt;
    };

    const auto a = build().components();
    const auto b = build().components();

    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        REQUIRE(a[i].name == b[i].name);
        REQUIRE(a[i].kind == b[i].kind);
        REQUIRE(a[i].nodes == b[i].nodes);
        REQUIRE(a[i].params == b[i].params);
    }
}

// -----------------------------------------------------------------------------
// Snapshot semantics — retained reference is not mutated by later add_*
// -----------------------------------------------------------------------------

TEST_CASE("components: retained snapshot is independent from later add_* calls",
          "[component_introspection]") {
    Circuit ckt;
    const Index a = ckt.add_node("a");
    ckt.add_resistor("R1", a, ckt.ground(), 100.0);

    const auto snapshot = ckt.components();
    REQUIRE(snapshot.size() == 1);

    ckt.add_capacitor("C1", a, ckt.ground(), 1e-9, 0.0);

    // Original snapshot unchanged
    REQUIRE(snapshot.size() == 1);
    REQUIRE(snapshot[0].name == "R1");

    // New snapshot reflects the additional add_*
    REQUIRE(ckt.num_components() == 2);
    REQUIRE(ckt.components().size() == 2);
}

// -----------------------------------------------------------------------------
// node_position_hint
// -----------------------------------------------------------------------------

TEST_CASE("node_position_hint: ground returns Ground",
          "[component_introspection][position_hint]") {
    Circuit ckt;
    const auto role = ckt.node_position_hint(ckt.ground());
    REQUIRE(role.has_value());
    REQUIRE(*role == NodeRole::Ground);
}

TEST_CASE("node_position_hint: voltage source positive terminal is SourcePos",
          "[component_introspection][position_hint]") {
    Circuit ckt;
    const Index vin = ckt.add_node("vin");
    ckt.add_voltage_source("V1", vin, ckt.ground(), 12.0);

    const auto role_pos = ckt.node_position_hint(vin);
    REQUIRE(role_pos.has_value());
    REQUIRE(*role_pos == NodeRole::SourcePos);
}

TEST_CASE("node_position_hint: resistor-to-ground node classified as Load",
          "[component_introspection][position_hint]") {
    Circuit ckt;
    const Index vin  = ckt.add_node("vin");
    const Index vout = ckt.add_node("vout");
    ckt.add_voltage_source("V1", vin, ckt.ground(), 12.0);
    ckt.add_resistor("R1", vin, vout, 100.0);   // not a load (no gnd)
    ckt.add_resistor("Rload", vout, ckt.ground(), 1000.0);

    const auto role = ckt.node_position_hint(vout);
    REQUIRE(role.has_value());
    REQUIRE(*role == NodeRole::Load);
}

TEST_CASE("node_position_hint: out-of-range node id returns nullopt",
          "[component_introspection][position_hint]") {
    Circuit ckt;
    ckt.add_node("a");
    REQUIRE_FALSE(ckt.node_position_hint(99).has_value());
}

TEST_CASE("node_position_hint: unclassified internal node returns Internal",
          "[component_introspection][position_hint]") {
    Circuit ckt;
    const Index mid = ckt.add_node("mid");
    const Index hi  = ckt.add_node("hi");
    // mid is connected only by a non-grounded resistor + an inductor: no role.
    ckt.add_resistor("R1", hi, mid, 100.0);
    ckt.add_inductor("L1", mid, hi, 1e-6, 0.0);

    const auto role = ckt.node_position_hint(mid);
    REQUIRE(role.has_value());
    REQUIRE(*role == NodeRole::Internal);
}
