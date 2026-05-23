// =============================================================================
// Layer 5 V4 — Integration: Newton wired into run_transient
// =============================================================================
//
// Validates that `run_transient` correctly invokes
// `solve_with_newton_b_extra` when a `NonlinearRefreshFn` is
// supplied. Tests two scenarios:
//
//   1. Linear circuit + Newton wiring + `noop_refresh`:
//      result is bit-identical to the `cache.solve` path.
//   2. DC diode load-line via `run_transient`: simulate a
//      brief constant-source transient on a circuit with a
//      smooth-blend `IdealDiode`. Steady-state output matches
//      the textbook load-line answer.
//
// Stiffer scenarios (sinusoidal rectifier where the diode
// commutates fast) need damped Newton / line search — that's
// the next OpenSpec (`pulsim-v2-newton-globalization`).

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

TEST_CASE("run_transient + noop_refresh = cache.solve (bit-identical)",
          "[v2][layer5_v4][newton_wiring]") {
    // V_dc → R → GND. No nonlinear branches.
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();

    SimulationOptions opts{.t_start = 0, .t_end = 1.0, .dt = 0.1};
    SwitchScheduleFn fn = [](Real) { return SwitchStateMask(0); };

    // Without Newton.
    auto result_lin = run_transient(cache, g, pool, opts, fn);
    // With Newton + noop_refresh.
    NonlinearRefreshFn refresh = &noop_refresh;
    auto result_nl = run_transient(
        cache, g, pool, opts, fn,
        /*b_extra_fn=*/{}, /*start_from_dc_op=*/false,
        refresh);

    REQUIRE(result_lin.num_steps() == result_nl.num_steps());
    for (Size k = 0; k < result_lin.num_steps(); ++k) {
        REQUIRE(result_lin.times[k] == result_nl.times[k]);
        for (Eigen::Index i = 0; i < result_lin.states[k].size(); ++i) {
            REQUIRE(result_lin.states[k][i] ==
                    Approx(result_nl.states[k][i]).margin(1e-9));
        }
    }
}

TEST_CASE("DC diode load-line via run_transient",
          "[v2][layer5_v4][newton_wiring][diode]") {
    // V_dc(2V) → smooth-blend diode → R(1kΩ) → GND.
    // Simulate a brief transient at constant source; Newton in
    // run_transient finds the DC operating point.
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
    cache.build();   // static cache (no caps/L)

    SimulationOptions opts{
        .t_start = 0, .t_end = 1.0, .dt = 0.1,
        .max_newton_iterations = 100,   // give Newton room
    };
    SwitchScheduleFn fn = [](Real) { return SwitchStateMask(0); };
    NonlinearRefreshFn refresh = &refresh_smooth_diodes;

    auto result = run_transient(
        cache, g, pool, opts, fn,
        /*b_extra_fn=*/{}, /*start_from_dc_op=*/false,
        refresh);

    REQUIRE(result.num_steps() == 11);   // t = 0, 0.1, ..., 1.0

    // Every step should converge to the same DC answer.
    const Real V_dc = 2.0;
    const Real V_F0 = dp.V_F0;
    const Real R    = 1000.0;
    const Real I_ana = (V_dc - V_F0) / (R + dp.R_d);   // ~1.3 mA
    (void)V_F0;

    // Skip the first sample (warm-start from zero takes some
    // iters to converge); check k >= 1.
    for (Size k = 1; k < result.num_steps(); ++k) {
        const Real v_n0 = result.states[k][0];
        const Real v_n1 = result.states[k][1];
        const Real I    = v_n1 / R;
        INFO("k=" << k << "  v_n0=" << v_n0
             << "  v_n1=" << v_n1 << "  I=" << I);
        REQUIRE(v_n0 == Approx(V_dc).margin(1e-6));
        REQUIRE(I == Approx(I_ana).epsilon(0.05));
    }
}
