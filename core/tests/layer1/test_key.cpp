// =============================================================================
// Layer 1 — TopologyKey (graph_id × switch_mask)
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/key.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <unordered_map>

using namespace pulsim;
using namespace pulsim::topology;

namespace {

Graph make_buck_graph() {
    Graph g;
    g.add_node("vin");
    g.add_node("sw");
    g.add_node("out");
    g.add_branch(0, 1, BranchKind::PassiveLinear);
    g.add_branch(1, 2, BranchKind::PassiveLinear);
    g.add_branch(0, 1, BranchKind::Switch);
    g.add_branch(g.ground(), 1, BranchKind::Switch);
    return g;
}

}  // namespace

TEST_CASE("Same graph + same state → same TopologyKey",
          "[v2][layer1][key]") {
    Graph g1 = make_buck_graph();
    SwitchStateMask state1(2); state1.set(0, true);
    SwitchStateMask state2(2); state2.set(0, true);
    TopologyKey k1{g1.id(), state1};
    TopologyKey k2{g1.id(), state2};
    REQUIRE(k1 == k2);
    REQUIRE(k1.hash() == k2.hash());
}

TEST_CASE("Same graph + different state → different TopologyKey",
          "[v2][layer1][key]") {
    Graph g1 = make_buck_graph();
    SwitchStateMask state_off(2);
    SwitchStateMask state_M1(2); state_M1.set(0, true);
    TopologyKey k_off{g1.id(), state_off};
    TopologyKey k_M1{g1.id(), state_M1};
    REQUIRE(k_off != k_M1);
    REQUIRE(k_off.hash() != k_M1.hash());
}

TEST_CASE("Different graph + same state → different TopologyKey",
          "[v2][layer1][key]") {
    Graph g1 = make_buck_graph();
    Graph g2 = make_buck_graph();
    g2.add_branch(0, 2, BranchKind::PassiveLinear);  // extra branch
    REQUIRE(g1.id() != g2.id());

    SwitchStateMask state(2); state.set(0, true);
    TopologyKey k1{g1.id(), state};
    TopologyKey k2{g2.id(), state};
    REQUIRE(k1 != k2);
}

TEST_CASE("TopologyKey round-trips through std::unordered_map",
          "[v2][layer1][key]") {
    Graph g = make_buck_graph();
    SwitchStateMask state(2); state.set(1, true);
    TopologyKey key{g.id(), state};

    std::unordered_map<TopologyKey, int> cache;
    cache.emplace(key, 42);
    auto it = cache.find(key);
    REQUIRE(it != cache.end());
    REQUIRE(it->second == 42);
}
