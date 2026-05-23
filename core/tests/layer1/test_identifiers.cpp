// =============================================================================
// Layer 1 — strong identifier types (NodeId, BranchId, StateIdx)
// =============================================================================
//
// Proves the safety contract: each strong type stays distinct from
// `Index` and from the other strong types at compile time. Some of
// the contract is enforced by `static_assert` inside the header
// itself (those run at build time); the rest is exercised here as
// regular runtime tests so a `ctest` run can confirm the API works.

#include <catch2/catch_test_macros.hpp>

#include "pulsim/topology/identifiers.hpp"

#include <type_traits>
#include <unordered_map>

using namespace pulsim;
using namespace pulsim::topology;

TEST_CASE("Strong IDs round-trip through get()",
          "[v2][layer1][identifiers]") {
    NodeId   n{42};
    BranchId b{7};
    StateIdx s{15};

    REQUIRE(n.get() == 42);
    REQUIRE(b.get() == 7);
    REQUIRE(s.get() == 15);
}

TEST_CASE("Default-constructed strong IDs are invalid",
          "[v2][layer1][identifiers]") {
    NodeId   n;
    BranchId b;
    StateIdx s;

    REQUIRE_FALSE(n.is_valid());
    REQUIRE_FALSE(b.is_valid());
    REQUIRE_FALSE(s.is_valid());
    REQUIRE(n.get() == kInvalidIndex);

    NodeId   n2{0};
    REQUIRE(n2.is_valid());          // 0 is a valid node ID
}

TEST_CASE("Strong IDs support ground sentinel",
          "[v2][layer1][identifiers]") {
    NodeId gnd{kGround};
    REQUIRE_FALSE(gnd.is_valid());   // kGround == kInvalidIndex == -1
    REQUIRE(gnd.get() == kGround);
}

TEST_CASE("Equality only holds within the same strong-type family",
          "[v2][layer1][identifiers]") {
    NodeId a{5};
    NodeId b{5};
    NodeId c{6};

    REQUIRE(a == b);
    REQUIRE(a != c);

    // The whole point of the strong-types refactor — these lines
    // would NOT compile because there is no `operator==(NodeId,
    // BranchId)`. We verify the absence via type traits:
    static_assert(!std::equality_comparable_with<NodeId, BranchId>);
    static_assert(!std::equality_comparable_with<NodeId, StateIdx>);
    static_assert(!std::equality_comparable_with<BranchId, StateIdx>);

    // Same family is ordered: `std::set<NodeId>` etc. works.
    NodeId smaller{3};
    REQUIRE(smaller < a);
}

TEST_CASE("Strong IDs are usable as unordered_map keys",
          "[v2][layer1][identifiers]") {
    std::unordered_map<NodeId, std::string> name_of;
    name_of[NodeId{0}] = "in";
    name_of[NodeId{1}] = "out";
    REQUIRE(name_of.at(NodeId{0}) == "in");
    REQUIRE(name_of.at(NodeId{1}) == "out");
    REQUIRE(name_of.find(NodeId{2}) == name_of.end());
}

TEST_CASE("Strong IDs are 4-byte trivially copyable POD",
          "[v2][layer1][identifiers]") {
    static_assert(sizeof(NodeId)   == 4);
    static_assert(sizeof(BranchId) == 4);
    static_assert(sizeof(StateIdx) == 4);
    static_assert(std::is_trivially_copyable_v<NodeId>);
    static_assert(std::is_trivially_copyable_v<BranchId>);
    static_assert(std::is_trivially_copyable_v<StateIdx>);
    REQUIRE(sizeof(NodeId) == sizeof(Index));
}

TEST_CASE("Index → strong-type conversion requires explicit cast",
          "[v2][layer1][identifiers]") {
    // The whole safety story: this would FAIL TO COMPILE:
    //     NodeId implicit = Index{5};
    // because `NodeId(Index)` is `explicit`. Verify via traits.
    static_assert(!std::is_convertible_v<Index, NodeId>);
    static_assert(!std::is_convertible_v<Index, BranchId>);
    static_assert(!std::is_convertible_v<Index, StateIdx>);

    // The explicit form is always available.
    NodeId   n_ok{Index{5}};
    BranchId b_ok{Index{5}};
    REQUIRE(n_ok.get() == 5);
    REQUIRE(b_ok.get() == 5);
}
