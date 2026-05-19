// simplify-and-harden-numerical-surface — Phase 12.4 tests.
//
// Verifies the round-robin cap-balancing helper for MMC submodules:
//   - num_inserted = 0 → all bypassed
//   - num_inserted = N → all inserted
//   - arm_current > 0 → inserts the N lowest-voltage caps
//   - arm_current < 0 → inserts the N highest-voltage caps
//   - submodule_id preserved through the decision

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/templates/mmc.hpp"

#include <algorithm>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

std::vector<templates::MmcSubmoduleState>
make_states(std::initializer_list<Real> caps) {
    std::vector<templates::MmcSubmoduleState> out;
    int id = 0;
    for (Real v : caps) {
        out.push_back({id++, v});
    }
    return out;
}

}  // namespace

TEST_CASE("mmc_balance_submodules: num_inserted=0 bypasses all",
          "[mmc][balancer][bypass_all]") {
    auto states = make_states({100.0, 110.0, 95.0, 105.0});
    auto cmds = templates::mmc_balance_submodules(
        states, /*arm_current=*/+1.0, /*num_inserted=*/0);
    REQUIRE(cmds.size() == 4);
    for (const auto& c : cmds) {
        CHECK_FALSE(c.insert);
    }
}

TEST_CASE("mmc_balance_submodules: num_inserted=N inserts all",
          "[mmc][balancer][insert_all]") {
    auto states = make_states({100.0, 110.0, 95.0, 105.0});
    auto cmds = templates::mmc_balance_submodules(
        states, /*arm_current=*/+1.0, /*num_inserted=*/4);
    REQUIRE(cmds.size() == 4);
    for (const auto& c : cmds) {
        CHECK(c.insert);
    }
}

TEST_CASE("mmc_balance_submodules: charging inserts the lowest-voltage caps",
          "[mmc][balancer][charging]") {
    // Cap voltages: 100, 110, 95, 105 → sorted = [95 (id=2), 100 (id=0),
    //   105 (id=3), 110 (id=1)]. Charging (arm_current > 0) → insert 2
    //   lowest = ids 2 and 0.
    auto states = make_states({100.0, 110.0, 95.0, 105.0});
    auto cmds = templates::mmc_balance_submodules(
        states, /*arm_current=*/+1.0, /*num_inserted=*/2);
    REQUIRE(cmds.size() == 4);

    auto find_id = [&](int id) {
        return std::find_if(cmds.begin(), cmds.end(),
                            [id](const auto& c) {
                                return c.submodule_id == id;
                            });
    };
    CHECK(find_id(2)->insert);    // 95 V (lowest)
    CHECK(find_id(0)->insert);    // 100 V (2nd lowest)
    CHECK_FALSE(find_id(3)->insert);   // 105 V
    CHECK_FALSE(find_id(1)->insert);   // 110 V (highest)
}

TEST_CASE("mmc_balance_submodules: discharging inserts the highest-voltage caps",
          "[mmc][balancer][discharging]") {
    auto states = make_states({100.0, 110.0, 95.0, 105.0});
    auto cmds = templates::mmc_balance_submodules(
        states, /*arm_current=*/-1.0, /*num_inserted=*/2);
    REQUIRE(cmds.size() == 4);

    auto find_id = [&](int id) {
        return std::find_if(cmds.begin(), cmds.end(),
                            [id](const auto& c) {
                                return c.submodule_id == id;
                            });
    };
    CHECK(find_id(1)->insert);    // 110 V (highest)
    CHECK(find_id(3)->insert);    // 105 V (2nd highest)
    CHECK_FALSE(find_id(0)->insert);   // 100 V
    CHECK_FALSE(find_id(2)->insert);   // 95 V (lowest)
}

TEST_CASE("mmc_balance_submodules: num_inserted clamped to [0, N]",
          "[mmc][balancer][clamp]") {
    auto states = make_states({100.0, 110.0, 95.0, 105.0});

    auto over = templates::mmc_balance_submodules(states, 1.0, 100);
    REQUIRE(over.size() == 4);
    int over_count = 0;
    for (const auto& c : over) over_count += (c.insert ? 1 : 0);
    CHECK(over_count == 4);   // clamped to N

    auto under = templates::mmc_balance_submodules(states, 1.0, -5);
    REQUIRE(under.size() == 4);
    int under_count = 0;
    for (const auto& c : under) under_count += (c.insert ? 1 : 0);
    CHECK(under_count == 0);   // clamped to 0
}

TEST_CASE("mmc_balance_submodules: decisions in original input order",
          "[mmc][balancer][order]") {
    auto states = make_states({100.0, 110.0, 95.0, 105.0});
    auto cmds = templates::mmc_balance_submodules(states, +1.0, 2);
    REQUIRE(cmds.size() == 4);
    // Ids should be 0, 1, 2, 3 in that order (= input order).
    CHECK(cmds[0].submodule_id == 0);
    CHECK(cmds[1].submodule_id == 1);
    CHECK(cmds[2].submodule_id == 2);
    CHECK(cmds[3].submodule_id == 3);
}
