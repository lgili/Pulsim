// =============================================================================
// Layer 1 — NodeEquivalence (union-find under switch state)
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/node_equivalence.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>

using namespace pulsim;
using namespace pulsim::topology;

namespace {

// Build a tiny Buck-like topology:
//   vin ─[PassiveLinear: R_pull]─ sw ─[Switch: M1]─ vin
//   sw ─[Switch: D1]─ gnd
//   sw ─[PassiveLinear: L]─ out
//   out ─[PassiveLinear: R_load]─ gnd
// (5 nodes: vin, sw, out, vload-equivalent, gnd; 2 switches.)
struct BuckTopology {
    Graph g;
    Index n_vin{}, n_sw{}, n_out{}, n_vload{};
    Size sw_M1_bit{}, sw_D1_bit{};

    BuckTopology() {
        n_vin   = g.add_node("vin");
        n_sw    = g.add_node("sw");
        n_out   = g.add_node("out");
        n_vload = g.add_node("vload");
        // PassiveLinear branches
        g.add_branch(n_vin, n_sw, BranchKind::PassiveLinear);   // R_pull
        g.add_branch(n_sw, n_out, BranchKind::PassiveLinear);   // L
        g.add_branch(n_out, n_vload, BranchKind::PassiveLinear); // R_load
        g.add_branch(n_vload, g.ground(), BranchKind::PassiveLinear);
        // Switches — bit indices follow the order of addition
        g.add_branch(n_vin, n_sw, BranchKind::Switch);  sw_M1_bit = 0;
        g.add_branch(g.ground(), n_sw, BranchKind::Switch); sw_D1_bit = 1;
    }
};

}  // namespace

TEST_CASE("All switches open → every node is its own class",
          "[v2][layer1][node_equivalence]") {
    BuckTopology t;
    SwitchStateMask mask(2);   // both open
    NodeEquivalence eq(t.g, mask);
    REQUIRE(eq.num_classes() == 4);
    for (Index i = 0; i < t.g.num_nodes(); ++i) {
        REQUIRE(eq.representative_of(i) == i);
    }
}

TEST_CASE("Closing one switch merges its endpoints",
          "[v2][layer1][node_equivalence]") {
    BuckTopology t;
    SwitchStateMask mask(2);
    mask.set(t.sw_M1_bit, true);   // M1 ON, D1 OFF
    NodeEquivalence eq(t.g, mask);

    // vin and sw now share a class.
    REQUIRE(eq.are_equivalent(t.n_vin, t.n_sw));
    // out and vload remain isolated from vin/sw.
    REQUIRE_FALSE(eq.are_equivalent(t.n_vin, t.n_out));
    REQUIRE_FALSE(eq.are_equivalent(t.n_sw, t.n_vload));
    // 3 classes: {vin, sw}, {out}, {vload}.
    REQUIRE(eq.num_classes() == 3);
}

TEST_CASE("Closed switch touching ground promotes the other endpoint to ground",
          "[v2][layer1][node_equivalence]") {
    BuckTopology t;
    SwitchStateMask mask(2);
    mask.set(t.sw_D1_bit, true);   // D1 ON: ground ↔ sw
    NodeEquivalence eq(t.g, mask);

    REQUIRE(eq.representative_of(t.n_sw) == kGround);
    REQUIRE(eq.are_equivalent(t.n_sw, t.g.ground()));
}

TEST_CASE("Both buck switches closed: sw, vin, ground all merge",
          "[v2][layer1][node_equivalence]") {
    BuckTopology t;
    SwitchStateMask mask(2);
    mask.set(t.sw_M1_bit, true);
    mask.set(t.sw_D1_bit, true);
    NodeEquivalence eq(t.g, mask);

    // Transitive: vin↔sw via M1, sw↔gnd via D1 → vin↔gnd via transitivity.
    REQUIRE(eq.representative_of(t.n_vin) == kGround);
    REQUIRE(eq.representative_of(t.n_sw) == kGround);
    REQUIRE(eq.are_equivalent(t.n_vin, t.n_sw));
    // Other nodes untouched.
    REQUIRE(eq.representative_of(t.n_out) == t.n_out);
}

TEST_CASE("Chain of closed switches merges every node into ground",
          "[v2][layer1][node_equivalence]") {
    // n1 -sw1- n2 -sw2- n3 -sw3- gnd
    Graph g;
    Index n1 = g.add_node("n1");
    Index n2 = g.add_node("n2");
    Index n3 = g.add_node("n3");
    g.add_branch(n1, n2, BranchKind::Switch);
    g.add_branch(n2, n3, BranchKind::Switch);
    g.add_branch(n3, g.ground(), BranchKind::Switch);

    SwitchStateMask mask(3);
    mask.set(0, true);
    mask.set(1, true);
    mask.set(2, true);

    NodeEquivalence eq(g, mask);
    REQUIRE(eq.are_equivalent(n1, g.ground()));
    REQUIRE(eq.are_equivalent(n2, g.ground()));
    REQUIRE(eq.are_equivalent(n3, g.ground()));
    REQUIRE(eq.representative_of(n1) == kGround);
    REQUIRE(eq.num_classes() == 1);
}

TEST_CASE("class_members returns sorted node ids",
          "[v2][layer1][node_equivalence]") {
    BuckTopology t;
    SwitchStateMask mask(2);
    mask.set(t.sw_M1_bit, true);
    NodeEquivalence eq(t.g, mask);

    const Index rep = eq.representative_of(t.n_vin);
    auto members = eq.class_members(rep);
    REQUIRE(members.size() == 2);
    REQUIRE(std::is_sorted(members.begin(), members.end()));
    // The class contains exactly {vin, sw}.
    REQUIRE(std::find(members.begin(), members.end(), t.n_vin) != members.end());
    REQUIRE(std::find(members.begin(), members.end(), t.n_sw)  != members.end());
}
