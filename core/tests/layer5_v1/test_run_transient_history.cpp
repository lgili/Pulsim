// =============================================================================
// Layer 5 V1 — run_transient with HistoryState wiring
// =============================================================================
//
// Covers the 6-arg V1 signature in isolation from the RC/RL/RLC
// integration tests:
//   * Static circuit produces V0-identical output (backwards
//     compat).
//   * cache.dt() mismatch with opts.dt throws.

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

TEST_CASE("run_transient V1: cache.dt() ≠ opts.dt throws",
          "[v2][layer5_v1][run_transient]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_capacitor(0, {.C = 1e-6});

    PwlStateSpaceCache cache(g, pool);
    cache.build(/*dt=*/1e-6);

    SimulationOptions opts{.t_start = 0, .t_end = 1e-3, .dt = 2e-6};
    SwitchScheduleFn fn = [](Real) { return SwitchStateMask(0); };

    REQUIRE_THROWS_AS(run_transient(cache, g, pool, opts, fn),
                      std::invalid_argument);
}

TEST_CASE("run_transient V1: static circuit matches V0 output",
          "[v2][layer5_v1][run_transient][regression]") {
    // V-R-GND static circuit. Build with build() (no dt) so the
    // V1 path takes the no-history branch.
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();   // dt = 0 path

    SimulationOptions opts{.t_start = 0, .t_end = 1.0, .dt = 0.1};
    SwitchScheduleFn fn = [](Real) { return SwitchStateMask(0); };

    auto v0 = run_transient(cache, pool.state_size(g), opts, fn);
    auto v1 = run_transient(cache, g, pool, opts, fn);

    REQUIRE(v0.num_steps() == v1.num_steps());
    for (Size k = 0; k < v0.num_steps(); ++k) {
        REQUIRE(v0.times[k] == v1.times[k]);
        REQUIRE(v0.states[k][0] == Approx(v1.states[k][0]).margin(1e-15));
    }
}

TEST_CASE("run_transient V1: empty switch_fn throws",
          "[v2][layer5_v1][run_transient]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);
    DevicePool pool;
    pool.add_resistor(0, {.G = 1.0});

    PwlStateSpaceCache cache(g, pool);
    cache.build();

    SimulationOptions opts{.t_start = 0, .t_end = 1.0, .dt = 0.1};
    SwitchScheduleFn empty_fn;

    REQUIRE_THROWS_AS(run_transient(cache, g, pool, opts, empty_fn),
                      std::invalid_argument);
}
