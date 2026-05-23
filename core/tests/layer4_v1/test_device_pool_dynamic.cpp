// =============================================================================
// Layer 4 V1 — DevicePool extended with Capacitor + Inductor
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <stdexcept>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;

TEST_CASE("DevicePool: 1 capacitor → state_size == num_nodes",
          "[v2][layer4_v1][device_pool]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_capacitor(0, {.C = 1e-6});

    REQUIRE(pool.state_size(g) == 1);   // node only, no extra unknown
    REQUIRE(pool.num_voltage_sources() == 0);
    REQUIRE(pool.num_inductors() == 0);
    REQUIRE(pool.num_dynamic_branches() == 1);
}

TEST_CASE("DevicePool: 1 inductor → state_size == num_nodes + 1",
          "[v2][layer4_v1][device_pool]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, 1, BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_inductor(0, {.L = 1e-3});

    REQUIRE(pool.state_size(g) == 3);   // 2 nodes + 1 branch unknown
    REQUIRE(pool.num_inductors() == 1);
    REQUIRE(pool.num_dynamic_branches() == 1);
    REQUIRE(pool.branch_var_id_for_inductor(0, g) == 2);
}

TEST_CASE("DevicePool: source + inductor → branch_var ordering",
          "[v2][layer4_v1][device_pool]") {
    Graph g;
    g.add_node("a");
    g.add_node("b");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1, BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_inductor(1, {.L = 100e-6});

    REQUIRE(pool.state_size(g) == 4);   // 2 nodes + 1 src + 1 L
    REQUIRE(pool.branch_var_id_for_source(0, g) == 2);
    REQUIRE(pool.branch_var_id_for_inductor(1, g) == 3);
}

TEST_CASE("DevicePool: 2 inductors → sequential branch_var ids",
          "[v2][layer4_v1][device_pool]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_node("n2");
    g.add_branch(0, 1, BranchKind::PassiveLinear);
    g.add_branch(1, 2, BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_inductor(0, {.L = 1e-3});
    pool.add_inductor(1, {.L = 2e-3});

    REQUIRE(pool.state_size(g) == 5);
    REQUIRE(pool.branch_var_id_for_inductor(0, g) == 3);
    REQUIRE(pool.branch_var_id_for_inductor(1, g) == 4);
}

TEST_CASE("DevicePool: capacitor params round-trip",
          "[v2][layer4_v1][device_pool]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_capacitor(0, {.C = 4.7e-6});
    REQUIRE(pool.capacitor_params(0).C == 4.7e-6);
    REQUIRE(pool.kind_of(0) == DevicePool::StoredKind::Capacitor);
}

TEST_CASE("DevicePool: inductor params round-trip + kind_of",
          "[v2][layer4_v1][device_pool]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_inductor(0, {.L = 220e-6});
    REQUIRE(pool.inductor_params(0).L == 220e-6);
    REQUIRE(pool.kind_of(0) == DevicePool::StoredKind::Inductor);
}

TEST_CASE("DevicePool: wrong-kind capacitor lookup throws",
          "[v2][layer4_v1][device_pool]") {
    DevicePool pool;
    pool.add_resistor(0, {.G = 1.0});
    REQUIRE_THROWS_AS(pool.capacitor_params(0), std::out_of_range);
    REQUIRE_THROWS_AS(pool.inductor_params(0), std::out_of_range);
}

TEST_CASE("DevicePool: branch_var_id_for_inductor on non-L throws",
          "[v2][layer4_v1][device_pool]") {
    Graph g;
    g.add_node("n0");
    DevicePool pool;
    pool.add_capacitor(0, {.C = 1e-6});
    REQUIRE_THROWS_AS(pool.branch_var_id_for_inductor(0, g),
                      std::out_of_range);
}

TEST_CASE("DevicePool V0 path unchanged with no dynamic devices",
          "[v2][layer4_v1][device_pool][regression]") {
    // Sanity: the V0 chopper layout still reports the same
    // sizes (this guards against accidental state-size bumps).
    Graph g;
    g.add_node("vin");
    g.add_node("vout");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1, BranchKind::Switch);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_switch(1, /*g_on=*/1e3, /*g_off=*/1e-9);
    pool.add_resistor(2, {.G = 0.1});

    REQUIRE(pool.state_size(g) == 3);   // 2 nodes + 1 src
    REQUIRE(pool.num_voltage_sources() == 1);
    REQUIRE(pool.num_inductors() == 0);
    REQUIRE(pool.num_dynamic_branches() == 0);
}
