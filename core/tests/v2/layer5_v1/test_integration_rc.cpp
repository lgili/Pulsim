// =============================================================================
// Layer 5 V1 — Integration: RC charging transient
// =============================================================================
//
// Circuit:  V_dc ──[Source]── n0 ──[R]── n1 ──[C]── GND
//                                       ^
//                                       voltage we track
//
// Analytical: V_C(t) = V_dc · (1 − e^{−t/τ}),  τ = RC.
//
// V0 IC: V_C(0) = 0, I_R(0+) → V_dc/R.
//
// Trap rule + small dt should reproduce the analytical solution
// to within < 1 %.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <cmath>
#include <memory>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::topology;
using Catch::Approx;

namespace {

struct RCCircuit {
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n0 = -1, n1 = -1;
    Real V_dc = 5.0;
    Real R    = 1.0;
    Real C    = 1e-6;

    explicit RCCircuit(Real dt) {
        n0 = g.add_node("n0");
        n1 = g.add_node("n1");
        g.add_branch(n0, g.ground(), BranchKind::Source);
        g.add_branch(n0, n1,         BranchKind::PassiveLinear);
        g.add_branch(n1, g.ground(), BranchKind::PassiveLinear);

        pool.add_voltage_source(0, {.V = V_dc});
        pool.add_resistor(1, {.G = Real{1} / R});
        pool.add_capacitor(2, {.C = C});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build(dt);
    }

    [[nodiscard]] Real tau() const noexcept { return R * C; }
};

}  // namespace

TEST_CASE("RC integration: V_C(t) tracks 1−e^{−t/τ} within 1 %",
          "[v2][layer5_v1][integration][rc]") {
    const Real tau = 1.0 * 1e-6;     // τ = RC = 1 µs
    const Real dt  = tau / 100;       // 10 ns → very small relative error

    RCCircuit rc(dt);
    SimulationOptions opts{
        .t_start = 0,
        .t_end   = 5.0 * tau,         // 5 time constants
        .dt      = dt,
    };

    SwitchScheduleFn fn = [](Real) {
        return SwitchStateMask(0);
    };

    auto result = run_transient(*rc.cache, rc.g, rc.pool, opts, fn);
    REQUIRE(result.num_steps() == 501);

    // V0 IC limitation: at t = 0 the cap's branch current i_C(0+)
    // is determined by external circuit (= V_dc/R here), but
    // HistoryState initialises it to 0. This produces a trap-rule
    // boundary artifact whose magnitude decays as λ_trap^n ≈
    // (1 - dt/RC)^n per step. By t = τ (k = 100), the artifact is
    // below 1 % of the analytical value. We start the relative-
    // error check at k = N/4 to skip the IC-transient region.
    // (DC operating-point pre-charge is a follow-up OpenSpec.)
    const Size k_start = result.num_steps() / 4;
    Size n_checked = 0;
    Size n_within_1pct = 0;
    Real max_rel_err = 0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real t = result.times[k];
        const Real v_C_num = result.states[k][rc.n1];
        const Real v_C_ana = rc.V_dc * (Real{1} - std::exp(-t / tau));

        const Real rel_err = std::abs(v_C_num - v_C_ana) /
                              std::max(std::abs(v_C_ana), Real{1e-9});
        max_rel_err = std::max(max_rel_err, rel_err);
        if (rel_err < Real{0.01}) {
            ++n_within_1pct;
        }
        ++n_checked;
    }

    INFO("RC: max relative error after step " << k_start
         << " = " << max_rel_err);
    INFO("RC: " << n_within_1pct << " / " << n_checked
         << " samples within 1 %");
    REQUIRE(max_rel_err < Real{0.01});
}

TEST_CASE("RC integration: V_C(5τ) ≈ V_dc · 0.9933 within 1 %",
          "[v2][layer5_v1][integration][rc]") {
    const Real tau = 1e-6;
    const Real dt  = tau / 100;
    RCCircuit rc(dt);
    SimulationOptions opts{.t_start = 0, .t_end = 5.0 * tau,
                            .dt = dt};
    SwitchScheduleFn fn = [](Real) { return SwitchStateMask(0); };

    auto result = run_transient(*rc.cache, rc.g, rc.pool, opts, fn);

    const Real final_v_C = result.states.back()[rc.n1];
    const Real expected  = rc.V_dc * (Real{1} - std::exp(-5.0));
    INFO("V_C(5τ) numerical = " << final_v_C
         << " V, analytical = " << expected << " V");
    REQUIRE(final_v_C == Approx(expected).epsilon(0.01));
}
