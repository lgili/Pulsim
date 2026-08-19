// =============================================================================
// Layer 1 — SwitchStateMask (bitmask bit ops, equality, hash)
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/topology/switch_state.hpp"

#include <set>
#include <stdexcept>
#include <unordered_map>

using namespace pulsim;
using namespace pulsim::topology;

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

TEST_CASE("Dynamic width: masks beyond 64 switches (Phase 1)",
          "[switch_state][wide]") {
    // The v1.x class threw std::invalid_argument for N > 64 — the hard
    // ceiling that made 120-switch MMC phases unrepresentable (audit
    // finding switch-mask-64-cap). Phase 1 replaces the single-word
    // backing with a small-buffer word array: <= 128 switches inline,
    // beyond that heap-backed. Same bit-indexed API.
    SECTION("65..128 switches: inline two-word storage") {
        SwitchStateMask m{100};
        REQUIRE(m.size() == 100);
        REQUIRE(m.num_words() == 2);
        REQUIRE(m.count() == 0);
        m.set(0, true);
        m.set(63, true);
        m.set(64, true);   // first bit of word 1
        m.set(99, true);
        REQUIRE(m.count() == 4);
        REQUIRE(m.get(63));
        REQUIRE(m.get(64));
        REQUIRE(m.get(99));
        REQUIRE_FALSE(m.get(65));
        m.flip(64);
        REQUIRE_FALSE(m.get(64));
        REQUIRE(m.count() == 3);
    }
    SECTION("beyond 128 switches: heap spill, same semantics") {
        SwitchStateMask m{400};   // e.g. 200-SM MMC arm, HB cells
        REQUIRE(m.num_words() == 7);
        m.set(399, true);
        m.set(128, true);
        REQUIRE(m.get(399));
        REQUIRE(m.get(128));
        REQUIRE(m.count() == 2);
        SwitchStateMask copy = m;      // deep copy
        copy.set(399, false);
        REQUIRE(m.get(399));           // original untouched
        REQUIRE(copy.count() == 1);
    }
    SECTION("equality + hash are width-safe") {
        SwitchStateMask a{100};
        SwitchStateMask b{100};
        a.set(97, true);
        b.set(97, true);
        REQUIRE(a == b);
        REQUIRE(std::hash<SwitchStateMask>{}(a) ==
                std::hash<SwitchStateMask>{}(b));
        b.flip(3);
        REQUIRE(a != b);
        REQUIRE(a.hamming_distance(b) == 1);
    }
    SECTION("overlay merge is word-wise") {
        SwitchStateMask user{100};
        SwitchStateMask overlay{100};
        SwitchStateMask owned{100};
        user.set(2, true);
        user.set(70, true);      // will be overridden (owned)
        overlay.set(70, false);
        overlay.set(90, true);
        owned.set(70, true);
        owned.set(90, true);
        const auto out = user.overlay(overlay, owned);
        REQUIRE(out.get(2));          // user bit kept
        REQUIRE_FALSE(out.get(70));   // overlay wins where owned
        REQUIRE(out.get(90));
    }
    SECTION("raw-word shortcuts throw loudly on wide masks") {
        SwitchStateMask m{100};
        REQUIRE_THROWS_AS(m.bits(), std::logic_error);
        REQUIRE_THROWS_AS(m.set_bits(1ULL), std::logic_error);
        SwitchStateMask small{8};
        small.set_bits(0b1010ULL);      // still fine at <= 64
        REQUIRE(small.bits() == 0b1010ULL);
    }
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
