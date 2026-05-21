// =============================================================================
// Layer 4 V2 — DC operating-point assembly tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/dc_assemble.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/topology/graph.hpp"

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("DC OP: V-R-GND gives v_node = V_dc",
          "[v2][layer4_v2][dc_op]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0});

    SwitchStateMask mask(0);
    const auto x = compute_dc_op(g, pool, mask);
    REQUIRE(x[0] == Approx(5.0).margin(1e-9));
}

TEST_CASE("DC OP: V-R-C-GND has v_C = V_dc (cap fully charged)",
          "[v2][layer4_v2][dc_op]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);    // V_dc
    g.add_branch(0, 1,          BranchKind::PassiveLinear);  // R
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);  // C

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0});
    pool.add_capacitor(2, {.C = 1e-6});

    SwitchStateMask mask(0);
    const auto x = compute_dc_op(g, pool, mask);
    // v_C node 1: at DC, no current flows through R into cap →
    // v_n1 = v_n0 (no drop across R).
    REQUIRE(x[1] == Approx(5.0).margin(1e-9));
}

TEST_CASE("DC OP: V-R-L-GND gives i_L = V/R",
          "[v2][layer4_v2][dc_op]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::PassiveLinear);  // R
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);  // L

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_resistor(1, {.G = 1.0});           // R = 1 Ω
    pool.add_inductor(2, {.L = 10e-6});

    SwitchStateMask mask(0);
    const auto x = compute_dc_op(g, pool, mask);
    const Index i_L_idx = pool.branch_var_id_for_inductor(2, g);
    INFO("dc_x = [" << x[0] << ", " << x[1] << ", "
         << x[2] << ", " << x[3] << "]");
    REQUIRE(x[0] == Approx(12.0).margin(1e-9));
    REQUIRE(x[1] == Approx(0.0).margin(1e-6));   // L = short
    REQUIRE(x[i_L_idx] == Approx(12.0).margin(1e-6));
}

TEST_CASE("DC OP: chopper with switch ON pulls v_out to V_dc",
          "[v2][layer4_v2][dc_op]") {
    Graph g;
    g.add_node("vin");
    g.add_node("vout");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Switch);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_switch(1, /*g_on=*/1e3, /*g_off=*/1e-9);
    pool.add_resistor(2, {.G = 0.1});

    SwitchStateMask mask_on(1);
    mask_on.set(0, true);
    const auto x_on = compute_dc_op(g, pool, mask_on);
    REQUIRE(x_on[0] == Approx(12.0).margin(1e-9));
    REQUIRE(x_on[1] == Approx(11.998801).epsilon(1e-3));
}
