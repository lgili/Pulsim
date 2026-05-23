// =============================================================================
// Layer 4 — PwlStateSpaceCache (build, lookup, solve)
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <stdexcept>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("Cache with 0 switches builds exactly 1 segment",
          "[v2][layer4][cache]") {
    // V_dc + R + GND, no switches.
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {Real{12.0}});
    pool.add_resistor(1, {Real{1.0}});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    REQUIRE(cache.num_segments() == 1);
}

TEST_CASE("Cache with 1 switch builds exactly 2 segments",
          "[v2][layer4][cache]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);

    DevicePool pool;
    pool.add_switch(0, Real{1e3}, Real{1e-9});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    REQUIRE(cache.num_segments() == 2);
}

TEST_CASE("Cache with 4 switches builds exactly 16 segments",
          "[v2][layer4][cache]") {
    Graph g;
    g.add_node("n0");
    for (int i = 0; i < 4; ++i) {
        g.add_branch(0, g.ground(), BranchKind::Switch);
    }
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);  // anchor

    DevicePool pool;
    for (Index i = 0; i < 4; ++i) {
        pool.add_switch(i, Real{1e3}, Real{1e-9});
    }
    pool.add_resistor(4, {Real{1.0}});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    REQUIRE(cache.num_segments() == 16);
}

TEST_CASE("lookup for a never-built mask throws",
          "[v2][layer4][cache]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_resistor(0, {Real{1.0}});

    PwlStateSpaceCache cache(g, pool);
    cache.build();

    // A mask of mismatched size (the cache was built for N=0
    // switches, so only the empty mask is in there).
    SwitchStateMask wrong_size(2);
    REQUIRE_THROWS_AS(cache.lookup(wrong_size), std::out_of_range);
}

TEST_CASE("solve on V-R-GND returns the analytical answer",
          "[v2][layer4][cache]") {
    // V_dc(0, gnd) = 12, R(0, gnd) = 0.1 S (= 10 Ω). At convergence:
    //   v_n0 = V_dc = 12
    //   i_branch = -V_dc · G = -1.2 (sign per Layer 3's source
    //     convention — current LEAVES n0 via the source-branch row)
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {Real{12.0}});
    pool.add_resistor(1, {Real{0.1}});

    PwlStateSpaceCache cache(g, pool);
    cache.build();

    SwitchStateMask mask(0);
    Vector b_extra = Vector::Zero(2);
    Vector x;
    cache.solve(mask, b_extra, x);

    REQUIRE(x[0] == Approx(Real{12}).margin(1e-12));
    REQUIRE(x[1] == Approx(Real{-1.2}).margin(1e-12));
}

// =============================================================================
// V4 ergonomics — non-throwing try_lookup / try_build_one_segment
// =============================================================================

TEST_CASE("try_lookup returns segment pointer on hit",
          "[v2][layer4][cache][expected]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {Real{12.0}});
    pool.add_resistor(1, {Real{1.0}});

    PwlStateSpaceCache cache(g, pool);
    cache.build();

    auto r = cache.try_lookup(SwitchStateMask(0));
    REQUIRE(r.has_value());
    REQUIRE(*r != nullptr);
    REQUIRE((*r)->state_size == 2);
}

TEST_CASE("try_lookup returns MaskNotBuilt on miss (eager mode)",
          "[v2][layer4][cache][expected]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);

    DevicePool pool;
    pool.add_switch(0, Real{1e3}, Real{1e-9});

    PwlStateSpaceCache cache(g, pool);
    cache.build();

    // size != graph.num_switches → never built
    SwitchStateMask wrong(2);
    auto r = cache.try_lookup(wrong);
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().kind == CacheError::Kind::MaskNotBuilt);
    REQUIRE(r.error().mask == wrong);
}

TEST_CASE("try_lookup auto-builds in lazy mode and reports "
          "structured singularity",
          "[v2][layer4][cache][expected]") {
    // Floating R between two non-grounded nodes → assemble gives
    // a row of all-zero entries for whichever node is dangling.
    Graph g;
    g.add_node("a");
    g.add_node("b");
    g.add_branch(0, 1, BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_resistor(0, {Real{1.0}});

    PwlStateSpaceCache cache(g, pool);
    cache.build_lazy(Real{0});

    auto r = cache.try_lookup(SwitchStateMask(0));
    // Either the resistor anchors enough rows (success) or
    // we get a structured singular-matrix error. The contract
    // we test here is: the API NEVER throws.
    if (!r) {
        REQUIRE((r.error().kind ==
                    CacheError::Kind::StructurallySingular ||
                 r.error().kind ==
                    CacheError::Kind::NumericallySingular));
    }
}

