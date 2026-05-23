// =============================================================================
// Layer 4 — DevicePool (branch_id → params registry)
// =============================================================================

#include <catch2/catch_test_macros.hpp>

#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

#include <stdexcept>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;

TEST_CASE("Empty pool reports zero state_size",
          "[v2][layer4][device_pool]") {
    DevicePool pool;
    Graph g;
    REQUIRE(pool.state_size(g) == 0);
    REQUIRE(pool.num_voltage_sources() == 0);
}

TEST_CASE("Resistor adds no branch-current unknown",
          "[v2][layer4][device_pool]") {
    DevicePool pool;
    Graph g;
    g.add_node("n0"); g.add_node("n1");
    pool.add_resistor(0, {Real{1.0}});

    REQUIRE(pool.state_size(g) == 2);          // just the 2 nodes
    REQUIRE(pool.num_voltage_sources() == 0);
    REQUIRE(pool.kind_of(0) == DevicePool::StoredKind::Resistor);
}

TEST_CASE("Voltage sources add sequential branch-current unknowns",
          "[v2][layer4][device_pool]") {
    DevicePool pool;
    Graph g;
    g.add_node("n0"); g.add_node("n1");
    pool.add_voltage_source(10, {Real{5.0}});
    pool.add_voltage_source(11, {Real{12.0}});
    pool.add_resistor(12, {Real{0.5}});

    REQUIRE(pool.state_size(g) == 4);          // 2 nodes + 2 sources
    REQUIRE(pool.num_voltage_sources() == 2);

    // branch_var_id = num_nodes + (insertion order).
    REQUIRE(pool.branch_var_id_for_source(10, g) == 2);
    REQUIRE(pool.branch_var_id_for_source(11, g) == 3);
}

TEST_CASE("kind_of distinguishes the three V0 device categories",
          "[v2][layer4][device_pool]") {
    DevicePool pool;
    pool.add_resistor(0, {Real{1.0}});
    pool.add_voltage_source(1, {Real{12.0}});
    pool.add_switch(2, Real{1e3}, Real{1e-9});

    REQUIRE(pool.kind_of(0) == DevicePool::StoredKind::Resistor);
    REQUIRE(pool.kind_of(1) == DevicePool::StoredKind::VoltageSource);
    REQUIRE(pool.kind_of(2) == DevicePool::StoredKind::Switch);
}

TEST_CASE("Wrong-kind lookup throws std::out_of_range",
          "[v2][layer4][device_pool]") {
    DevicePool pool;
    pool.add_resistor(0, {Real{1.0}});

    REQUIRE_THROWS_AS(pool.voltage_source_params(0), std::out_of_range);
    REQUIRE_THROWS_AS(pool.switch_g_on(0), std::out_of_range);
}

TEST_CASE("Missing branch_id lookup throws",
          "[v2][layer4][device_pool]") {
    DevicePool pool;
    REQUIRE_THROWS_AS(pool.kind_of(99), std::out_of_range);
    REQUIRE_THROWS_AS(pool.resistor_params(99), std::out_of_range);
}

TEST_CASE("Switch params round-trip through g_on/g_off accessors",
          "[v2][layer4][device_pool]") {
    DevicePool pool;
    pool.add_switch(0, Real{2e3}, Real{5e-9});
    REQUIRE(pool.switch_g_on(0) == Real{2e3});
    REQUIRE(pool.switch_g_off(0) == Real{5e-9});
}
