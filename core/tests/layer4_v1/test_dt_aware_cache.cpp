// =============================================================================
// Layer 4 V1 — dt-aware cache.build(dt)
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("Cache.build() with no dynamic devices behaves like V0",
          "[v2][layer4_v1][cache][regression]") {
    Graph g;
    g.add_node("vin");
    g.add_node("vout");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1, BranchKind::Switch);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_switch(1, 1e3, 1e-9);
    pool.add_resistor(2, {.G = 0.1});

    PwlStateSpaceCache cache(g, pool);
    cache.build();   // V0 path
    REQUIRE(cache.dt() == Real{0});
    REQUIRE(cache.num_segments() == 2);
}

TEST_CASE("Cache.build(dt) stamps the cap's g_eq",
          "[v2][layer4_v1][cache]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_capacitor(0, {.C = 1e-6});

    PwlStateSpaceCache cache(g, pool);
    cache.build(/*dt=*/1e-6);
    REQUIRE(cache.dt() == 1e-6);
    REQUIRE(cache.num_segments() == 1);  // no switches

    // The segment's matrix should have J(0, 0) == g_eq = 2.
    SwitchStateMask empty(0);
    const auto& seg = cache.lookup(empty);
    REQUIRE(seg.J.coeff(0, 0) == Approx(2.0).margin(1e-12));
}

TEST_CASE("Cache.build(dt) rebuilds when dt changes",
          "[v2][layer4_v1][cache]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_capacitor(0, {.C = 1e-6});

    PwlStateSpaceCache cache(g, pool);
    cache.build(1e-6);
    REQUIRE(cache.dt() == 1e-6);

    cache.build(2e-6);
    REQUIRE(cache.dt() == 2e-6);

    SwitchStateMask empty(0);
    const auto& seg = cache.lookup(empty);
    // g_eq = 2·1e-6 / 2e-6 = 1.0
    REQUIRE(seg.J.coeff(0, 0) == Approx(1.0).margin(1e-12));
}

TEST_CASE("Cache.build() on graph with capacitor SILENTLY skips cap",
          "[v2][layer4_v1][cache]") {
    // V0 backwards-compat path: dt=0 means dynamic devices don't
    // stamp. The resulting matrix is structurally singular for a
    // 1-node graph with only a (skipped) cap → build() should
    // throw.
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_capacitor(0, {.C = 1e-6});

    PwlStateSpaceCache cache(g, pool);
    REQUIRE_THROWS(cache.build());   // singular
}

TEST_CASE("Cache.build(dt) with inductor stamps the constraint row",
          "[v2][layer4_v1][cache]") {
    // V_dc → L → GND: state size = 1 node + 1 src + 1 L = 3.
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_inductor(1, {.L = 1e-6});

    PwlStateSpaceCache cache(g, pool);
    cache.build(/*dt=*/1e-7);
    REQUIRE(cache.dt() == 1e-7);
    REQUIRE(cache.num_segments() == 1);

    // g_eq_inv = 1e-7 / 2e-6 = 0.05. 2L/dt = 20.
    SwitchStateMask empty(0);
    const auto& seg = cache.lookup(empty);
    // Inductor branch_var_id = num_nodes + num_sources = 2.
    REQUIRE(seg.J.coeff(2, 2) == Approx(-20.0).margin(1e-9));
    // KCL constraint: J(2, 0) = +1.
    REQUIRE(seg.J.coeff(2, 0) == Approx(+1.0).margin(1e-12));
}
