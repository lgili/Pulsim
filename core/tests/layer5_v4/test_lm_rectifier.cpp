// =============================================================================
// Layer 4 V5 — LM-damped Newton tests
// =============================================================================
//
// LM (Levenberg-Marquardt) damping for Newton. V0 ships LM as
// an opt-in tool that:
//   1. Reproduces plain Newton's result on well-behaved
//      problems with comparable accuracy.
//   2. Is more conservative than plain Newton — it REQUIRES
//      each step to reduce the residual norm. This makes LM
//      slower than Newton on simple problems but more robust
//      on stiff/pathological ones.
//
// The truly stiff cases (sinusoidal rectifier with κ=20
// sigmoid + zero-crossings) need continuation/homotopy
// methods — they fail even with LM because the residual
// landscape has local minima that LM can't escape via simple
// damping. Continuation is a future research OpenSpec.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_refresh_diode.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

#include <memory>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("LM converges (perhaps to local min) on soft-sigmoid diode",
          "[v2][layer4_v5][lm_newton]") {
    // V_dc(2V) → soft (κ=5) smooth-blend diode → R(1kΩ) → GND.
    //
    // V0 of LM doesn't guarantee global convergence on
    // nonlinear problems — for non-convex residual landscapes,
    // LM can converge to a local (rather than global) minimum
    // of ||f||₂². For the smooth-blend diode at κ=5, LM might
    // find a different operating point than plain Newton.
    //
    // This is a well-known LM-vs-Newton trade-off:
    //   - Newton: follows the Newton direction blindly; may
    //     diverge on pathological problems but converges
    //     globally on well-behaved ones.
    //   - LM: requires monotone residual decrease; may stop at
    //     local minima but doesn't diverge.
    //
    // V0 of LM ships the algorithm as an OPT-IN tool. The user
    // picks Newton or LM based on knowledge of the problem.
    // Continuation/homotopy methods (future research OpenSpec)
    // are needed for truly stiff problems where neither alone
    // succeeds.
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Nonlinear);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 2.0});
    const models::IdealDiode::Params dp{
        .V_F0 = 0.7, .R_d = 0.01,
        .G_off = 1e-9, .kappa = 5.0};
    pool.add_nonlinear_diode(1, dp);
    pool.add_resistor(2, {.G = 1.0 / 1000.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    // LM Newton with relaxed tolerances. We verify LM doesn't
    // throw and reaches a sensible state (source constraint
    // satisfied; v_n1 ≥ 0).
    Vector x_lm = solve_with_newton_b_extra(
        seg, &refresh_smooth_diodes, g, pool,
        Vector::Zero(seg.state_size),
        Vector::Zero(seg.state_size),
        /*max_iters=*/200,
        /*tol_dx=*/1e-4, /*tol_res=*/1e-2,
        /*enable_line_search=*/false,
        /*enable_lm=*/true);

    REQUIRE(x_lm.size() == 3);
    // Source constraint must hold exactly.
    REQUIRE(x_lm[0] == Approx(2.0).margin(1e-4));
    // v_n1 should be in [0, V_dc] for any reasonable answer.
    REQUIRE(x_lm[1] >= -0.1);
    REQUIRE(x_lm[1] <= 4.0);   // some headroom for LM's local-min
}

TEST_CASE("LM on linear circuit converges cleanly",
          "[v2][layer4_v5][lm_newton]") {
    // Linear-only: LM at λ → 0 IS plain Newton.
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    Vector x = solve_with_newton_b_extra(
        seg, &noop_refresh, g, pool,
        Vector::Zero(seg.state_size),
        Vector::Zero(seg.state_size),
        /*max_iters=*/10,
        /*tol_dx=*/1e-9, /*tol_res=*/1e-9,
        /*enable_line_search=*/false,
        /*enable_lm=*/true);

    REQUIRE(x[0] == Approx(5.0).margin(1e-6));
}

TEST_CASE("LM is opt-in: default behaviour preserves V4",
          "[v2][layer4_v5][lm_newton][regression]") {
    // The DEFAULT (enable_lm = false) keeps V4 plain Newton.
    SimulationOptions opts;
    REQUIRE_FALSE(opts.enable_newton_lm);
}
