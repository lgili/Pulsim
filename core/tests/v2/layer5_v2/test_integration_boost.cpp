// =============================================================================
// Layer 5 V2.1 — Integration: boost converter (with event iteration)
// =============================================================================
//
// The boost converter that DIDN'T work in Layer 5 V2 (the diode
// chattered at the DCM/CCM boundary during startup, producing
// v_sw blow-up and V_out stuck at ~25 mV). Layer 5 V2.1's
// event-iteration loop fixes this: at each step, we re-solve
// until the diode state is consistent with the just-computed
// (v, i). At commutation moments the loop runs 2-3 iterations;
// elsewhere it runs once.
//
//   V_in(12V) ──[L(100µH)]── v_sw ──┬──[Q (controlled switch)]── GND
//                                     │
//                                     └──[Diode]── V_out ──┬─[C(100µF)]── GND
//                                                            │
//                                                            └─[R_load(20Ω)]── GND
//
// PWM controls Q at 100 kHz, D = 0.5. Diode auto-commutates.
// Expected steady state: V_out = V_in/(1-D) = 24 V.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::topology;
using Catch::Approx;

namespace {

struct BoostConverter {
    static constexpr Real V_in   = 12.0;
    static constexpr Real D_pwm  = 0.5;
    static constexpr Real f_sw   = 100e3;
    static constexpr Real T_sw   = 1.0 / f_sw;
    static constexpr Real L      = 100e-6;
    static constexpr Real C      = 100e-6;
    static constexpr Real R_load = 20.0;
    static constexpr Real g_on   = 1e3;
    static constexpr Real g_off  = 1e-9;

    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index v_in_idx  = -1;
    Index v_sw_idx  = -1;
    Index v_out_idx = -1;
    Index i_L_idx   = -1;

    explicit BoostConverter(Real dt) {
        v_in_idx  = g.add_node("v_in");
        v_sw_idx  = g.add_node("v_sw");
        v_out_idx = g.add_node("v_out");

        // Branch order locks switch_idx → bit position:
        //   0: V_in source
        //   1: L  (v_in → v_sw)
        //   2: Q  (v_sw → GND)   ← switch_idx = 0, user-controlled
        //   3: D  (v_sw → v_out) ← switch_idx = 1, auto-controlled
        //   4: C  (v_out → GND)
        //   5: R_load (v_out → GND)
        g.add_branch(v_in_idx,  g.ground(),  BranchKind::Source);
        g.add_branch(v_in_idx,  v_sw_idx,    BranchKind::PassiveLinear); // L
        g.add_branch(v_sw_idx,  g.ground(),  BranchKind::Switch);        // Q
        g.add_branch(v_sw_idx,  v_out_idx,   BranchKind::Switch);        // D
        g.add_branch(v_out_idx, g.ground(),  BranchKind::PassiveLinear); // C
        g.add_branch(v_out_idx, g.ground(),  BranchKind::PassiveLinear); // R

        pool.add_voltage_source(0, {.V = V_in});
        pool.add_inductor(1, {.L = L});
        pool.add_switch(2, g_on, g_off);
        pool.add_diode(3, g_on, g_off, /*V_th=*/0.0);
        pool.add_capacitor(4, {.C = C});
        pool.add_resistor(5, {.G = 1.0 / R_load});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build(dt);

        i_L_idx = pool.branch_var_id_for_inductor(1, g);
    }

    static SwitchStateMask pwm_schedule(Real t) {
        const Real phase = std::fmod(t, T_sw);
        const bool q_on = phase < D_pwm * T_sw;
        SwitchStateMask mask(2);
        mask.set(0, q_on);
        mask.set(1, false);   // diode bit (overwritten by tracker)
        return mask;
    }
};

}  // namespace

TEST_CASE("Boost converter: cache builds 4 segments",
          "[v2][layer5_v2][integration][boost]") {
    BoostConverter b(1e-7);
    REQUIRE(b.cache->num_segments() == 4);
}

TEST_CASE("Boost converter: V_out = V_in/(1-D) with event iteration",
          "[v2][layer5_v2][integration][boost]") {
    constexpr Real dt = 1e-7;
    constexpr Real t_end = 10e-3;
    BoostConverter b(dt);

    SimulationOptions opts{
        .t_start = 0,
        .t_end   = t_end,
        .dt      = dt,
        .max_event_iterations = 16,
    };

    const auto t_wall_start = std::chrono::steady_clock::now();
    auto result = run_transient(
        *b.cache, b.g, b.pool, opts, &BoostConverter::pwm_schedule);
    const auto t_wall_end = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            t_wall_end - t_wall_start).count();

    REQUIRE(result.num_steps() == 100001);
    REQUIRE(result.event_iteration_count.size() ==
            result.num_steps());

    // -------- Diagnostics: event-iteration statistics
    Size max_iters = 0;
    Size total_extra_solves = 0;
    Size num_with_iteration = 0;
    for (Size k = 0; k < result.num_steps(); ++k) {
        const Size c = result.event_iteration_count[k];
        max_iters = std::max(max_iters, c);
        total_extra_solves += c;
        if (c > 0) ++num_with_iteration;
    }

    // -------- Steady-state metrics over the last 1 ms.
    const Size k_start = result.num_steps() - 10000;
    Real sum_v_out = 0, sum_i_L = 0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        sum_v_out += result.states[k][b.v_out_idx];
        sum_i_L   += result.states[k][b.i_L_idx];
    }
    const Real n = static_cast<Real>(result.num_steps() - k_start);
    const Real mean_v_out = sum_v_out / n;
    const Real mean_i_L   = sum_i_L   / n;

    const Real V_out_ana =
        BoostConverter::V_in / (1.0 - BoostConverter::D_pwm);  // 24
    const Real I_L_ana =
        V_out_ana * V_out_ana
        / (BoostConverter::V_in * BoostConverter::R_load);     // 2.4

    INFO("Boost steady state:");
    INFO("  mean V_out = " << mean_v_out << " V (analytical: "
         << V_out_ana << " V)");
    INFO("  mean I_L   = " << mean_i_L << " A (analytical: "
         << I_L_ana << " A)");
    INFO("  Wall clock = " << elapsed_ms << " ms for "
         << result.num_steps() << " samples");
    INFO("Event iteration stats:");
    INFO("  max iterations on any step = " << max_iters);
    INFO("  total extra solves         = " << total_extra_solves);
    INFO("  steps that triggered iter  = " << num_with_iteration);

    // No step should have hit the iteration limit.
    REQUIRE(max_iters < 16);

    // Steady state must match analytical within 10 %.
    REQUIRE(mean_v_out == Approx(V_out_ana).epsilon(0.10));
    REQUIRE(mean_i_L   == Approx(I_L_ana).epsilon(0.10));

    // Performance: 100k steps with event iteration should still
    // finish in well under 30 s on a Debug build.
    REQUIRE(elapsed_ms < 30000);
}
