// =============================================================================
// Layer 1 — Graph (nodes, branches, adjacency, structural id)
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/topology/graph.hpp"

#include <type_traits>

using namespace pulsim;
using namespace pulsim::topology;

TEST_CASE("Graph default-constructs empty with ground sentinel",
          "[v2][layer1][graph]") {
    Graph g;
    REQUIRE(g.num_nodes() == 0);
    REQUIRE(g.num_branches() == 0);
    REQUIRE(g.ground() == kGround);
    REQUIRE(g.ground() == Index{-1});
}

TEST_CASE("add_node returns monotonically increasing indices",
          "[v2][layer1][graph]") {
    Graph g;
    REQUIRE(g.add_node("n0") == 0);
    REQUIRE(g.add_node("n1") == 1);
    REQUIRE(g.add_node("n2") == 2);
    REQUIRE(g.num_nodes() == 3);
    REQUIRE(g.node(0).name == "n0");
    REQUIRE(g.node(2).name == "n2");
    REQUIRE(g.node(0).id == 0);
    REQUIRE(g.node(2).id == 2);
}

TEST_CASE("add_branch records endpoints + kind",
          "[v2][layer1][graph]") {
    Graph g;
    const Index a = g.add_node("a");
    const Index b = g.add_node("b");
    const Index id = g.add_branch(a, b, BranchKind::PassiveLinear);
    REQUIRE(id == 0);
    REQUIRE(g.num_branches() == 1);
    REQUIRE(g.branch(0).from == a);
    REQUIRE(g.branch(0).to == b);
    REQUIRE(g.branch(0).kind == BranchKind::PassiveLinear);
}

TEST_CASE("branches_of returns adjacency for the node's incident branches",
          "[v2][layer1][graph]") {
    Graph g;
    const Index a = g.add_node("a");
    const Index b = g.add_node("b");
    const Index c = g.add_node("c");
    const Index ab = g.add_branch(a, b, BranchKind::PassiveLinear);
    const Index bc = g.add_branch(b, c, BranchKind::PassiveLinear);
    const Index ac = g.add_branch(a, c, BranchKind::Switch);

    auto adj_a = g.branches_of(a);
    REQUIRE(adj_a.size() == 2);    // ab + ac touch a
    REQUIRE(adj_a[0] == ab);
    REQUIRE(adj_a[1] == ac);

    auto adj_b = g.branches_of(b);
    REQUIRE(adj_b.size() == 2);    // ab + bc touch b

    auto adj_c = g.branches_of(c);
    REQUIRE(adj_c.size() == 2);    // bc + ac touch c
}

TEST_CASE("Branches touching ground are not added to adjacency",
          "[v2][layer1][graph]") {
    Graph g;
    const Index a = g.add_node("a");
    g.add_branch(a, g.ground(), BranchKind::PassiveLinear);
    // The single real node `a` sees the branch; ground has no
    // adjacency-list slot (it's a sentinel, not a real node).
    REQUIRE(g.branches_of(a).size() == 1);
    REQUIRE(g.branches_of(g.ground()).size() == 0);
}

TEST_CASE("num_switches counts only Switch-kind branches",
          "[v2][layer1][graph]") {
    Graph g;
    const Index a = g.add_node("a");
    const Index b = g.add_node("b");
    g.add_branch(a, b, BranchKind::PassiveLinear);
    g.add_branch(a, b, BranchKind::Switch);
    g.add_branch(a, b, BranchKind::Source);
    g.add_branch(a, b, BranchKind::Switch);
    g.add_branch(a, b, BranchKind::Nonlinear);
    REQUIRE(g.num_branches() == 5);
    REQUIRE(g.num_switches() == 2);
}

TEST_CASE("Graph is move-only", "[v2][layer1][graph]") {
    STATIC_REQUIRE(std::is_move_constructible_v<Graph>);
    STATIC_REQUIRE(std::is_move_assignable_v<Graph>);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<Graph>);
    STATIC_REQUIRE_FALSE(std::is_copy_assignable_v<Graph>);
}

TEST_CASE("Graph::id is stable for a structurally identical graph",
          "[v2][layer1][graph]") {
    auto build = []() {
        Graph g;
        g.add_node("a");
        g.add_node("b");
        g.add_branch(0, 1, BranchKind::PassiveLinear);
        g.add_branch(0, 1, BranchKind::Switch);
        return g;
    };
    Graph g1 = build();
    Graph g2 = build();
    REQUIRE(g1.id() == g2.id());
}

TEST_CASE("Graph::id differs for graphs with different structure",
          "[v2][layer1][graph]") {
    Graph g1;
    g1.add_node("a"); g1.add_node("b");
    g1.add_branch(0, 1, BranchKind::PassiveLinear);

    Graph g2;
    g2.add_node("a"); g2.add_node("b");
    g2.add_branch(0, 1, BranchKind::PassiveLinear);
    g2.add_branch(0, 1, BranchKind::Switch);   // extra branch

    REQUIRE(g1.id() != g2.id());
}

TEST_CASE("Graph::id is cached after first call",
          "[v2][layer1][graph]") {
    Graph g;
    g.add_node("a");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);
    const auto id1 = g.id();
    const auto id2 = g.id();   // cached
    REQUIRE(id1 == id2);
}
