// =============================================================================
// Layer 5 V2.2 — Commutation event diagnostics (sub-step bisection V0)
// =============================================================================
//
// Verifies that run_transient populates
// `SimulationResult.commutation_events` with linear-interpolated
// commutation times when a diode flips.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

#include <memory>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("Diode-free circuit produces no commutation events",
          "[v2][layer5_v2][substep_bisection]") {
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

    auto result = run_transient(cache, g, pool, opts, fn);
    REQUIRE(result.commutation_events.empty());
}

TEST_CASE("Diode + step-function source records the turn-on event",
          "[v2][layer5_v2][substep_bisection]") {
    // V_dc(t < t_step) = -2 V (diode reverse-biased, OFF)
    // V_dc(t ≥ t_step) = +2 V (diode forward-biased, turns ON)
    //
    // We use a source with V = 0 and modulate via b_extra_fn to
    // create a step at t_step = 0.5.
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Switch);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 0.0});
    pool.add_diode(1, /*g_on=*/1e3, /*g_off=*/1e-9, /*V_th=*/0.0);
    pool.add_resistor(2, {.G = 0.1});

    PwlStateSpaceCache cache(g, pool);
    cache.build();

    const Index src_var = pool.branch_var_id_for_source(0, g);
    const Eigen::Index state_size =
        static_cast<Eigen::Index>(pool.state_size(g));
    constexpr Real t_step = 0.5;
    constexpr Real V_dc = 2.0;
    auto b_extra_fn = [src_var, state_size](Real t) -> Vector {
        Vector b = Vector::Zero(state_size);
        const Real V = (t >= t_step) ? V_dc : -V_dc;
        b[src_var] = -V;
        return b;
    };

    SimulationOptions opts{
        .t_start = 0, .t_end = 1.0, .dt = 0.01};
    SwitchScheduleFn fn = [](Real) { return SwitchStateMask(1); };

    auto result = run_transient(
        cache, g, pool, opts, fn, b_extra_fn);

    INFO("commutation_events.size() = "
         << result.commutation_events.size());
    REQUIRE(result.commutation_events.size() >= 1);

    // The first event should be the OFF → ON transition right
    // around t_step.
    const auto& first = result.commutation_events[0];
    INFO("first event t=" << first.t_estimated
         << " new_state=" << first.new_state);
    REQUIRE(first.new_state == true);   // OFF → ON
    REQUIRE(first.t_estimated == Approx(t_step).margin(opts.dt * 2));
}
