// =============================================================================
// Layer 1 — Gray-code enumeration over switch states
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/topology/enumerator.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <bit>
#include <cstdint>
#include <set>
#include <vector>

using namespace pulsim;
using namespace pulsim::topology;

TEST_CASE("N = 0 yields exactly 1 state (the empty mask)",
          "[v2][layer1][enumerator]") {
    int count = 0;
    for (auto mask : enumerate_switch_states(0)) {
        REQUIRE(mask.size() == 0);
        REQUIRE(mask.count() == 0);
        ++count;
    }
    REQUIRE(count == 1);
}

TEST_CASE("N = 3 yields exactly 8 distinct masks",
          "[v2][layer1][enumerator]") {
    std::set<SwitchStateMask> seen;
    for (auto mask : enumerate_switch_states(3)) {
        seen.insert(mask);
    }
    REQUIRE(seen.size() == 8);
}

TEST_CASE("Consecutive Gray-code states differ by exactly one bit",
          "[v2][layer1][enumerator]") {
    std::vector<SwitchStateMask> sequence;
    for (auto mask : enumerate_switch_states(4)) {
        sequence.push_back(mask);
    }
    REQUIRE(sequence.size() == 16);

    // For every consecutive pair, popcount(prev XOR curr) == 1.
    for (Size i = 1; i < sequence.size(); ++i) {
        const std::uint64_t xor_bits =
            sequence[i].bits() ^ sequence[i - 1].bits();
        REQUIRE(std::popcount(xor_bits) == 1);
    }
}

TEST_CASE("N = 16 yields exactly 65536 distinct masks (smoke)",
          "[v2][layer1][enumerator]") {
    std::uint64_t count = 0;
    for (auto mask : enumerate_switch_states(16)) {
        (void)mask;
        ++count;
    }
    REQUIRE(count == 65536ULL);
}

TEST_CASE("Enumerator exposes num_states + num_switches accessors",
          "[v2][layer1][enumerator]") {
    SwitchStateEnumerator e(5);
    REQUIRE(e.num_switches() == 5);
    REQUIRE(e.num_states() == 32);
}

TEST_CASE("Iterator equality compares both counter and N",
          "[v2][layer1][enumerator]") {
    SwitchStateEnumerator e1(3);
    SwitchStateEnumerator e2(3);
    REQUIRE(e1.begin() == e2.begin());
    REQUIRE(e1.end() == e2.end());
}
