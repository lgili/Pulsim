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

// =============================================================================
// v2.0 Phase 1 — (G, C, b) split assembly
// =============================================================================
//
// J(mask, dt) = G(mask) + (1/dt)·C must hold EXACTLY (same
// floating-point arithmetic — assemble_segment itself recombines
// the split, so this locks the two public entry points together),
// with C mask-invariant and b dt-independent.

TEST_CASE("assemble_segment_split: J == G + (1/dt)*C exactly",
          "[v2][layer4][assemble][split]") {
    // RLC + switch + transformer — every stamp class with a
    // dt-dependence plus every static one.
    topology::Graph g;
    DevicePool pool;
    g.add_node("a");
    g.add_node("b");
    g.add_node("c");
    g.add_branch(0, g.ground(), topology::BranchKind::Source);
    g.add_branch(0, 1, topology::BranchKind::PassiveLinear);   // R
    g.add_branch(1, 2, topology::BranchKind::PassiveLinear);   // L1
    g.add_branch(2, g.ground(), topology::BranchKind::PassiveLinear); // C
    g.add_branch(2, g.ground(), topology::BranchKind::PassiveLinear); // L2
    g.add_branch(1, g.ground(), topology::BranchKind::Switch);

    pool.add_voltage_source(0, {.V = 48.0});
    pool.add_resistor(1, {.G = 2.0});
    pool.add_inductor(2, {.L = 2e-3});
    pool.add_capacitor(3, {.C = 4.7e-6});
    pool.add_inductor(4, {.L = 1e-3});
    pool.add_switch(5, /*g_on=*/1e3, /*g_off=*/1e-9);
    pool.add_transformer_coupling(2, 4,
        {.L_p = 2e-3, .L_s = 1e-3, .k = 0.95});

    topology::SwitchStateMask mask(1);
    mask.set(0, true);

    sparse::Matrix G, C;
    Vector b_split;
    assemble_segment_split(g, pool, mask, G, C, b_split);

    for (Real dt : {1e-6, 3.7e-7, 1e-3}) {
        sparse::Matrix J;
        Vector b;
        assemble_segment(g, pool, mask, dt, J, b);

        sparse::Matrix J_combined = G + (Real{1} / dt) * C;
        DenseMatrix diff =
            DenseMatrix(J) - DenseMatrix(J_combined);
        REQUIRE(diff.cwiseAbs().maxCoeff() == Real{0});  // EXACT
        REQUIRE((b - b_split).cwiseAbs().maxCoeff() == Real{0});
    }
}

TEST_CASE("assemble_segment_split: C is mask-invariant, G is not",
          "[v2][layer4][assemble][split]") {
    topology::Graph g;
    DevicePool pool;
    g.add_node("a");
    g.add_branch(0, g.ground(), topology::BranchKind::Source);
    g.add_branch(0, g.ground(), topology::BranchKind::PassiveLinear); // C
    g.add_branch(0, g.ground(), topology::BranchKind::Switch);
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_capacitor(1, {.C = 1e-6});
    pool.add_switch(2, 1e3, 1e-9);

    topology::SwitchStateMask off(1), on(1);
    on.set(0, true);

    sparse::Matrix G_off, C_off, G_on, C_on;
    Vector b_off, b_on;
    assemble_segment_split(g, pool, off, G_off, C_off, b_off);
    assemble_segment_split(g, pool, on,  G_on,  C_on,  b_on);

    REQUIRE((DenseMatrix(C_on) - DenseMatrix(C_off))
                .cwiseAbs().maxCoeff() == Real{0});
    REQUIRE((DenseMatrix(G_on) - DenseMatrix(G_off))
                .cwiseAbs().maxCoeff() > Real{1});  // g_on vs g_off
    // C's only entry: the 2·C_cap block at (0,0) (grounded cap).
    REQUIRE(DenseMatrix(C_on)(0, 0) == Real{2} * Real{1e-6});
}

TEST_CASE("assemble_segment_split: hand-stamped companion values",
          "[v2][layer4][assemble][split]") {
    // NON-tautological check (adversarial-review finding
    // TEST-TAUTOLOGY): assemble_segment recombines the split, so
    // J == G + (1/dt)·C cannot catch a wrong stamp. Here every
    // dt-coefficient is asserted against its HAND-DERIVED value:
    //   cap block         ±2·C_cap        in C
    //   inductor diagonal −2·L            in C (branch-var row)
    //   inductor incidence ±1             in G (NOT in C)
    //   transformer cross −2·M            in C (both off-diagonals)
    topology::Graph g;
    DevicePool pool;
    g.add_node("a");   // 0
    g.add_node("b");   // 1
    g.add_branch(0, g.ground(), topology::BranchKind::Source);
    g.add_branch(0, 1, topology::BranchKind::PassiveLinear);          // L1 a→b
    g.add_branch(1, g.ground(), topology::BranchKind::PassiveLinear); // C1 b→gnd
    g.add_branch(1, g.ground(), topology::BranchKind::PassiveLinear); // L2 b→gnd

    const Real Lp = 2e-3, Ls = 1e-3, Cc = 4.7e-6, kk = 0.9;
    pool.add_voltage_source(0, {.V = 1.0});
    pool.add_inductor(1, {.L = Lp});
    pool.add_capacitor(2, {.C = Cc});
    pool.add_inductor(3, {.L = Ls});
    pool.add_transformer_coupling(1, 3,
        {.L_p = Lp, .L_s = Ls, .k = kk});
    const Real M = kk * std::sqrt(Lp * Ls);

    topology::SwitchStateMask m(0);
    sparse::Matrix G, C;
    Vector b;
    assemble_segment_split(g, pool, m, G, C, b);
    DenseMatrix Gd = DenseMatrix(G);
    DenseMatrix Cd = DenseMatrix(C);

    const Index bv1 = pool.branch_var_id_for_inductor(1, g);
    const Index bv2 = pool.branch_var_id_for_inductor(3, g);

    // Capacitor block at node b (index 1), grounded → only (1,1).
    REQUIRE(Cd(1, 1) == Real{2} * Cc);

    // Inductor constraint diagonals: −2L.
    REQUIRE(Cd(bv1, bv1) == -Real{2} * Lp);
    REQUIRE(Cd(bv2, bv2) == -Real{2} * Ls);

    // Transformer cross terms: −2M, symmetric.
    REQUIRE(Cd(bv1, bv2) == Approx(-Real{2} * M).epsilon(1e-15));
    REQUIRE(Cd(bv2, bv1) == Approx(-Real{2} * M).epsilon(1e-15));

    // Inductor incidence lives in G (dt-independent), NOT in C.
    REQUIRE(Gd(0, bv1)  == Real{1});    // KCL of i_L1 at node a
    REQUIRE(Gd(1, bv1)  == Real{-1});   // KCL of i_L1 at node b
    REQUIRE(Gd(bv1, 0)  == Real{1});    // constraint row v_a
    REQUIRE(Gd(bv1, 1)  == Real{-1});   // constraint row v_b
    REQUIRE(Gd(1, bv2)  == Real{1});    // L2: from = node b
    REQUIRE(Gd(bv2, 1)  == Real{1});
    REQUIRE(Cd(0, bv1)  == Real{0});
    REQUIRE(Cd(bv1, 0)  == Real{0});
    REQUIRE(Gd(bv1, bv1) == Real{0});   // diagonal ONLY in C
    REQUIRE(Gd(1, 1)     == Real{0});   // cap block ONLY in C

    // And the recombined matrix carries the classic forms:
    // 2C/dt on the cap diagonal, −2L/dt on the constraint row.
    const Real dt = 1e-6;
    sparse::Matrix J;
    Vector b2;
    assemble_segment(g, pool, m, dt, J, b2);
    DenseMatrix Jd = DenseMatrix(J);
    REQUIRE(Jd(1, 1) ==
            Approx(Real{2} * Cc / dt).epsilon(1e-15));
    REQUIRE(Jd(bv1, bv1) ==
            Approx(-Real{2} * Lp / dt).epsilon(1e-15));
    REQUIRE(Jd(bv1, bv2) ==
            Approx(-Real{2} * M / dt).epsilon(1e-15));
}
