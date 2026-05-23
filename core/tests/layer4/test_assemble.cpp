// =============================================================================
// Layer 4 — assemble_segment (build one per-state matrix)
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("Empty graph assembles to empty (J, b)",
          "[v2][layer4][assemble]") {
    Graph g;
    DevicePool pool;
    SwitchStateMask mask(0);
    sparse::Matrix J;
    Vector b;
    assemble_segment(g, pool, mask, J, b);
    REQUIRE(J.rows() == 0);
    REQUIRE(b.size() == 0);
}

TEST_CASE("Single resistor to ground stamps the right diagonal",
          "[v2][layer4][assemble]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);
    DevicePool pool;
    pool.add_resistor(0, {Real{2.0}});

    SwitchStateMask mask(0);
    sparse::Matrix J;
    Vector b;
    assemble_segment(g, pool, mask, J, b);

    REQUIRE(J.rows() == 1);
    REQUIRE(J.cols() == 1);
    REQUIRE(J.coeff(0, 0) == Approx(Real{2}));
    REQUIRE(b[0] == Approx(Real{0}));
}

TEST_CASE("Single voltage source to ground adds the constraint row",
          "[v2][layer4][assemble]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    DevicePool pool;
    pool.add_voltage_source(0, {Real{12.0}});

    SwitchStateMask mask(0);
    sparse::Matrix J;
    Vector b;
    assemble_segment(g, pool, mask, J, b);

    REQUIRE(J.rows() == 2);                       // n0 + branch current
    REQUIRE(J.cols() == 2);
    REQUIRE(J.coeff(0, 1) == Approx(Real{1}));    // KCL at n0: +i_branch
    REQUIRE(J.coeff(1, 0) == Approx(Real{1}));    // constraint: +v[0]
    REQUIRE(b[1] == Approx(Real{-12}));           // constraint RHS
}

TEST_CASE("Switch in OFF state stamps g_off",
          "[v2][layer4][assemble]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);
    DevicePool pool;
    pool.add_switch(0, /*g_on=*/Real{1e3}, /*g_off=*/Real{1e-9});

    SwitchStateMask mask(1);                       // bit 0 = 0 (open)
    sparse::Matrix J;
    Vector b;
    assemble_segment(g, pool, mask, J, b);

    REQUIRE(J.coeff(0, 0) == Approx(Real{1e-9}));
}

TEST_CASE("Switch in ON state stamps g_on",
          "[v2][layer4][assemble]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Switch);
    DevicePool pool;
    pool.add_switch(0, Real{1e3}, Real{1e-9});

    SwitchStateMask mask(1);
    mask.set(0, true);                              // closed
    sparse::Matrix J;
    Vector b;
    assemble_segment(g, pool, mask, J, b);

    REQUIRE(J.coeff(0, 0) == Approx(Real{1e3}));
}

TEST_CASE("Chopper assembly stamps every branch correctly",
          "[v2][layer4][assemble]") {
    // Chopper: V_dc(branch 0) → Switch(branch 1) → R(branch 2) → GND
    Graph g;
    Index n_in  = g.add_node("vin");
    Index n_out = g.add_node("vout");
    g.add_branch(n_in, g.ground(), BranchKind::Source);      // V_dc
    g.add_branch(n_in, n_out, BranchKind::Switch);            // switch
    g.add_branch(n_out, g.ground(), BranchKind::PassiveLinear); // R

    DevicePool pool;
    pool.add_voltage_source(0, {Real{12.0}});
    pool.add_switch(1, Real{1e3}, Real{1e-9});
    pool.add_resistor(2, {Real{0.1}});

    SwitchStateMask mask(1);
    mask.set(0, true);                              // switch closed

    sparse::Matrix J;
    Vector b;
    assemble_segment(g, pool, mask, J, b);

    // State size = 2 nodes + 1 source-branch unknown = 3.
    REQUIRE(J.rows() == 3);
    REQUIRE(b.size() == 3);

    // Voltage-source constraint row (branch_var_id = 2):
    //   J(2, 0) = +1 (constraint reads v_in)
    REQUIRE(J.coeff(2, 0) == Approx(Real{1}));
    REQUIRE(b[2] == Approx(Real{-12}));

    // KCL at vin: +i_branch_source contribution (KCL row 0,
    // branch_var col 2).
    REQUIRE(J.coeff(0, 2) == Approx(Real{1}));

    // R between vout and ground: J(1, 1) gets G_R = 0.1
    // PLUS the switch's g_on contribution (switch between vin and
    // vout, so its J(1, 1) entry is +g_on = +1e3).
    REQUIRE(J.coeff(1, 1) == Approx(Real{1e3 + 0.1}));

    // Closed switch contributes -g_on between (0,1) and (1,0):
    REQUIRE(J.coeff(0, 1) == Approx(Real{-1e3}));
    REQUIRE(J.coeff(1, 0) == Approx(Real{-1e3}));
}
