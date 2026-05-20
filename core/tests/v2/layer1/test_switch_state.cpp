// =============================================================================
// Layer 1 — SwitchStateMask (bitmask bit ops, equality, hash)
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/v2/topology/switch_state.hpp"

#include <set>
#include <stdexcept>
#include <unordered_map>

using namespace pulsim::v2;
using namespace pulsim::v2::topology;

TEST_CASE("Default-constructed mask of size N is all zero",
          "[v2][layer1][switch_state]") {
    SwitchStateMask m(8);
    REQUIRE(m.size() == 8);
    REQUIRE(m.count() == 0);
    for (Size i = 0; i < 8; ++i) {
        REQUIRE_FALSE(m.get(i));
    }
}

TEST_CASE("set / get round-trips bits in any order",
          "[v2][layer1][switch_state]") {
    SwitchStateMask m(16);
    m.set(0, true);
    m.set(5, true);
    m.set(15, true);
    REQUIRE(m.get(0));
    REQUIRE(m.get(5));
    REQUIRE(m.get(15));
    REQUIRE_FALSE(m.get(1));
    REQUIRE_FALSE(m.get(14));
    REQUIRE(m.count() == 3);

    m.set(5, false);
    REQUIRE_FALSE(m.get(5));
    REQUIRE(m.count() == 2);
}

TEST_CASE("flip toggles the addressed bit",
          "[v2][layer1][switch_state]") {
    SwitchStateMask m(4);
    m.flip(2); REQUIRE(m.get(2));
    m.flip(2); REQUIRE_FALSE(m.get(2));
    m.flip(2); REQUIRE(m.get(2));
}

TEST_CASE("Equality requires same size AND same bits",
          "[v2][layer1][switch_state]") {
    SwitchStateMask a(8); a.set(3, true);
    SwitchStateMask b(8); b.set(3, true);
    SwitchStateMask c(16); c.set(3, true);     // different size
    SwitchStateMask d(8); d.set(2, true);      // different bits
    REQUIRE(a == b);
    REQUIRE_FALSE(a == c);
    REQUIRE_FALSE(a == d);
    REQUIRE(a != c);
    REQUIRE(a != d);
}

TEST_CASE("Equal masks hash to the same value",
          "[v2][layer1][switch_state]") {
    SwitchStateMask a(8); a.set(3, true); a.set(7, true);
    SwitchStateMask b(8); b.set(3, true); b.set(7, true);
    REQUIRE(a.hash() == b.hash());
    REQUIRE(std::hash<SwitchStateMask>{}(a) ==
            std::hash<SwitchStateMask>{}(b));
}

TEST_CASE("Different-size masks with same bits hash differently",
          "[v2][layer1][switch_state]") {
    SwitchStateMask a(8); a.set(3, true);
    SwitchStateMask b(16); b.set(3, true);
    // The size enters the hash so they should (with overwhelming
    // probability) differ. Birthday-bound makes this almost certain
    // for splitmix64-mixed inputs.
    REQUIRE(a.hash() != b.hash());
}

TEST_CASE("operator< yields a stable total order usable by std::set",
          "[v2][layer1][switch_state]") {
    std::set<SwitchStateMask> seen;
    SwitchStateMask a(4); a.set(0, true);
    SwitchStateMask b(4); b.set(1, true);
    SwitchStateMask c(4); c.set(2, true);
    seen.insert(a); seen.insert(b); seen.insert(c);
    seen.insert(a);                            // duplicate insert
    REQUIRE(seen.size() == 3);                 // dedup'd
}

TEST_CASE("Constructor rejects N > 64",
          "[v2][layer1][switch_state]") {
    REQUIRE_NOTHROW(SwitchStateMask{64});
    REQUIRE_THROWS_AS(SwitchStateMask{65}, std::invalid_argument);
}

TEST_CASE("Usable as unordered_map key",
          "[v2][layer1][switch_state]") {
    std::unordered_map<SwitchStateMask, int> cache;
    SwitchStateMask k1(8); k1.set(3, true);
    cache.emplace(k1, 42);
    REQUIRE(cache.find(k1) != cache.end());
    REQUIRE(cache.at(k1) == 42);
}

TEST_CASE("to_string emits a binary representation",
          "[v2][layer1][switch_state]") {
    SwitchStateMask m(4);
    m.set(0, true); m.set(3, true);
    const auto s = m.to_string();
    REQUIRE(s.find("0b") != std::string::npos);
    REQUIRE(s.find("N=4") != std::string::npos);
}
