// =============================================================================
// Layer 4 V9 — V_F0 + combined κ/V_F0 override helpers
// =============================================================================
//
// V8 shipped `make_kappa_override_refresh`. V9 adds two
// companion factories:
//   * `make_vf0_override_refresh(V_F0)` — overrides just V_F0,
//     leaves R_d/G_off/κ from the pool.
//   * `make_kappa_vf0_override_refresh(κ, V_F0)` — overrides
//     both at once, for combined homotopies or parameter
//     sweeps.
//
// Use cases:
//   * Parameter sweeps (study how a circuit responds to
//     varying V_F0 or κ without rebuilding the DevicePool).
//   * Combined κ + V_F0 continuation chains for moderately-
//     stiff problems (DC operating points where direct
//     Newton from x = 0 misses the right basin).
//
// HONEST CAVEAT about the κ=20 stiff sinusoidal rectifier
// (the V8 motivating problem):
//   * Neither V_F0 alone, nor combined κ+V_F0 continuation,
//     succeeds in solving the κ=20 sinusoidal rectifier
//     from x = 0 at every time step. The sigmoid at κ=20
//     is too sharp — Newton's local-linear model breaks
//     down around the diode's commutation, and even the
//     finest continuation chain triggers matrix singularity
//     in Newton's inner iterations.
//   * For κ=20 sinusoidal rectifiers, the V8 LOAD-LINE
//     WARM-START remains the recommended path (see
//     test_continuation_rectifier.cpp).
//   * For LESS stiff problems (DC operating points with
//     reverse-biased diodes, κ ≤ ~8), continuation in V_F0
//     IS useful — demonstrated below.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/continuation.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/nonlinear_refresh_diode.hpp"
#include "pulsim/v2/pwl/nonlinear_solve.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::topology;
using Catch::Approx;

TEST_CASE("make_vf0_override_refresh uses override V_F0, "
          "not pool V_F0",
          "[v2][layer4_v9][vf0_override]") {
    // Pool has V_F0 = 0.7. At v_diode = +0.7 (exactly the
    // pool threshold), the default refresh has alpha = 0.5 →
    // small current. An override refresh at V_F0 = -5 shifts
    // the sigmoid centre way left → diode strongly forward-
    // biased at the same v_diode = +0.7, large current.
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
        .G_off = 1e-9, .kappa = 20.0};
    pool.add_nonlinear_diode(1, dp);
    pool.add_resistor(2, {.G = 1.0 / 1000.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    Vector x = Vector::Zero(seg.state_size);
    x[0] = 0.7;
    x[1] = 0.0;

    sparse::Matrix J_default(seg.state_size, seg.state_size);
    Vector f_default = Vector::Zero(seg.state_size);
    const Real r_default =
        refresh_smooth_diodes(x, J_default, f_default, g, pool);

    sparse::Matrix J_override(seg.state_size, seg.state_size);
    Vector f_override = Vector::Zero(seg.state_size);
    auto override_fn = make_vf0_override_refresh(-5.0);
    const Real r_override =
        override_fn(x, J_override, f_override, g, pool);

    INFO("default V_F0=0.7 residual: " << r_default
         << "  override V_F0=-5 residual: " << r_override);
    REQUIRE(r_override > r_default);
    REQUIRE(r_override > 1.0);
}

TEST_CASE("make_kappa_vf0_override_refresh overrides both",
          "[v2][layer4_v9][kappa_vf0_override]") {
    // Sanity check: the combined override differs from
    // both single-overrides at a generic x.
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Nonlinear);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 1.0});
    const models::IdealDiode::Params dp{
        .V_F0 = 0.7, .R_d = 0.01,
        .G_off = 1e-9, .kappa = 20.0};
    pool.add_nonlinear_diode(1, dp);
    pool.add_resistor(2, {.G = 1.0 / 1000.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    Vector x = Vector::Zero(seg.state_size);
    x[0] = 1.0;
    x[1] = 0.5;   // v_diode = 0.5

    sparse::Matrix J(seg.state_size, seg.state_size);
    Vector f = Vector::Zero(seg.state_size);

    auto k_only  = make_kappa_override_refresh(5.0);
    auto vf0_only = make_vf0_override_refresh(0.0);
    auto both    = make_kappa_vf0_override_refresh(5.0, 0.0);

    J.setZero(); f.setZero();
    const Real r_k = k_only(x, J, f, g, pool);
    J.setZero(); f.setZero();
    const Real r_vf0 = vf0_only(x, J, f, g, pool);
    J.setZero(); f.setZero();
    const Real r_both = both(x, J, f, g, pool);

    INFO("kappa_only=" << r_k << "  vf0_only=" << r_vf0
         << "  both=" << r_both);
    // Combined override differs from each single override.
    REQUIRE(r_both != Approx(r_k));
    REQUIRE(r_both != Approx(r_vf0));
}

TEST_CASE("V_F0 continuation: single-element sequence == direct Newton",
          "[v2][layer4_v9][vf0_continuation]") {
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
        .G_off = 1e-9, .kappa = 20.0};
    pool.add_nonlinear_diode(1, dp);
    pool.add_resistor(2, {.G = 1.0 / 1000.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    auto override_fn = make_vf0_override_refresh(0.3);

    Vector x_direct = solve_with_newton_b_extra(
        seg, override_fn, g, pool,
        Vector::Zero(seg.state_size),
        Vector::Zero(seg.state_size));

    std::vector<NonlinearRefreshFn> seq{override_fn};
    Vector x_cont = continuation_solve(
        seg, seq, g, pool,
        Vector::Zero(seg.state_size),
        Vector::Zero(seg.state_size));

    REQUIRE(x_direct.size() == x_cont.size());
    for (Eigen::Index i = 0; i < x_direct.size(); ++i) {
        REQUIRE(x_direct[i] == Approx(x_cont[i]).margin(1e-9));
    }
}

// -----------------------------------------------------------------------------
// V_F0 sweep: a parameter-sweep usage pattern that does NOT
// rely on the chain solving a stiff problem from x=0. The
// circuit is the DC diode load-line from layer4_v3 (V_dc=2V,
// R_load=1kΩ, κ=20). Plain Newton from x=0 already solves
// this. We use the V_F0-override factory to confirm: when
// V_F0 is overridden, the output v_n1 tracks the analytical
// load-line v_n1 ≈ V_dc − V_F0 for V_F0 ∈ {0.3, 0.5, 0.7}.
//
// This validates the override factories as composable
// building blocks (e.g. parameter sweeps in a Python-binding
// wrapper) — not as a magic-bullet for stiff problems.
// -----------------------------------------------------------------------------
TEST_CASE("V_F0 override sweep matches analytical load-line",
          "[v2][layer4_v9][vf0_sweep][dc]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Nonlinear);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 2.0});
    const models::IdealDiode::Params dp{
        .V_F0 = 0.7,          // pool default (will be overridden)
        .R_d = 0.01,
        .G_off = 1e-9, .kappa = 20.0};
    pool.add_nonlinear_diode(1, dp);
    pool.add_resistor(2, {.G = 1.0 / 1000.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();
    const auto& seg = cache.lookup(SwitchStateMask(0));

    // For each override V_F0, the converged v_n1 should be
    // ~ V_dc − V_F0 (with small ~R_d/R_load correction).
    for (Real vf0_override : {0.3, 0.5, 0.7}) {
        auto refresh = make_vf0_override_refresh(vf0_override);
        Vector x = solve_with_newton_b_extra(
            seg, refresh, g, pool,
            Vector::Zero(seg.state_size),
            Vector::Zero(seg.state_size),
            /*max_iters=*/100);
        const Real v_n1 = x[1];
        const Real expected = 2.0 - vf0_override;
        INFO("V_F0=" << vf0_override
             << "  v_n1=" << v_n1
             << "  expected~" << expected);
        REQUIRE(v_n1 == Approx(expected).margin(0.05));
    }
}
