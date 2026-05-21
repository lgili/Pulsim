// =============================================================================
// Layer 4 V2 — Integration: RC starting from DC operating point
// =============================================================================
//
// With start_from_dc_op=true, the cap starts at the steady-state
// voltage (V_dc) instead of zero. The trap-rule simulation
// then produces a CONSTANT v_C across all samples — no charge
// transient, because we're already at steady state.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <memory>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::topology;
using Catch::Approx;

namespace {

struct RC {
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n0 = -1, n1 = -1;
    Real V_dc = 5.0;
    Real R    = 1.0;
    Real C    = 1e-6;

    explicit RC(Real dt) {
        n0 = g.add_node("n0");
        n1 = g.add_node("n1");
        g.add_branch(n0, g.ground(), BranchKind::Source);
        g.add_branch(n0, n1,         BranchKind::PassiveLinear);
        g.add_branch(n1, g.ground(), BranchKind::PassiveLinear);

        pool.add_voltage_source(0, {.V = V_dc});
        pool.add_resistor(1, {.G = 1.0 / R});
        pool.add_capacitor(2, {.C = C});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build(dt);
    }
};

}  // namespace

TEST_CASE("RC from DC OP: v_C stays at V_dc throughout",
          "[v2][layer4_v2][integration][rc_dc_op]") {
    const Real tau = 1e-6;
    const Real dt  = tau / 100;
    RC rc(dt);

    SimulationOptions opts{
        .t_start = 0,
        .t_end   = 5.0 * tau,
        .dt      = dt,
    };
    SwitchScheduleFn fn = [](Real) { return SwitchStateMask(0); };

    auto result = run_transient(
        *rc.cache, rc.g, rc.pool, opts, fn,
        /*b_extra_fn=*/{}, /*start_from_dc_op=*/true);

    REQUIRE(result.num_steps() == 501);
    // v_C(0) = V_dc (DC steady state).
    REQUIRE(result.states[0][rc.n1] ==
            Approx(rc.V_dc).margin(1e-6));
    // v_C(t) stays at V_dc for all t (already at steady state).
    for (Size k = 0; k < result.num_steps(); ++k) {
        REQUIRE(result.states[k][rc.n1] ==
                Approx(rc.V_dc).epsilon(0.01));
    }
}

TEST_CASE("RC from-zero IC (default) reproduces V0 behaviour",
          "[v2][layer4_v2][integration][rc_dc_op][regression]") {
    // Regression: when start_from_dc_op = false (default), the
    // RC simulation produces the V1 / V2.1 transient.
    const Real tau = 1e-6;
    const Real dt  = tau / 100;
    RC rc(dt);

    SimulationOptions opts{
        .t_start = 0,
        .t_end   = 5.0 * tau,
        .dt      = dt,
    };
    SwitchScheduleFn fn = [](Real) { return SwitchStateMask(0); };

    auto result = run_transient(
        *rc.cache, rc.g, rc.pool, opts, fn);

    // First sample is the IC = 0.
    REQUIRE(result.states[0][rc.n1] == Approx(0.0).margin(1e-9));
    // Last sample is close to V_dc · (1 - e^{-5}) ≈ 4.97 V.
    const Real expected_final = rc.V_dc * (1.0 - std::exp(-5.0));
    REQUIRE(result.states.back()[rc.n1] ==
            Approx(expected_final).epsilon(0.05));
}
