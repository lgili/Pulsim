// =============================================================================
// Layer 5 V1.5 — Integration: synchronous buck converter
// =============================================================================
//
// The architectural proof point. Every layer working together on
// a real PE workload:
//
//   V_in (12 V) ──[Q1 high-side]──┐
//                                   ├──[L (100 µH)]── V_out ──┬─[R_load (10 Ω)]── GND
//   GND ──[Q2 low-side]─────────────┘                          │
//                                                               └─[C (100 µF)]── GND
//
// PWM: f_sw = 100 kHz, duty D = 0.5. Q1 ON during [0, T/2); Q2 ON
// during [T/2, T). Always exactly one switch on.
//
// Simulate from rest for 1 ms (100 PWM periods) at dt = 100 ns
// (10 001 samples, 100 per period). Measure steady-state over the
// LAST 10 PWM periods.
//
// Analytical expectations in CCM at this operating point:
//   V_out  = V_in · D                            = 6.0 V
//   I_L    = V_out / R_load                       = 0.6 A
//   ΔI_L   = V_in · D · (1-D) / (L · f_sw)        = 0.3 A pk-pk
//   ΔV_out = ΔI_L / (8 · C · f_sw)                = 3.75 mV pk-pk
//
// If this passes, every layer of v2 is proven on a real workload.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;
using Catch::Approx;

namespace {

struct BuckConverter {
    // Circuit parameters.
    //
    // R_load picked for ζ ≈ 0.5 of the output LC tank so it
    // settles in well under 1 ms (vs the 8 ms for ζ = 0.05 with
    // R_load = 10 Ω). The output LC dynamics:
    //   ω_n = 1/√(LC) = 1/√(100µ · 100µ) = 1e4 rad/s
    //   ζ   = 1/(2·R·ω_n·C)
    //   settling ≈ 4/(ζ·ω_n)
    // With R_load = 1 Ω, ζ = 0.5 and settling ≈ 0.8 ms.
    // Mean I_L = V_out / R = 6 A at this operating point (higher
    // than a typical PE example but keeps the tank well damped).
    static constexpr Real V_in    = 12.0;
    static constexpr Real D       = 0.5;
    static constexpr Real f_sw    = 100e3;     // 100 kHz
    static constexpr Real T_sw    = 1.0 / f_sw;
    static constexpr Real L       = 100e-6;    // 100 µH
    static constexpr Real C       = 100e-6;    // 100 µF
    static constexpr Real R_load  = 1.0;       // 1 Ω → ζ = 0.5
    static constexpr Real G_load  = 1.0 / R_load;
    static constexpr Real g_on    = 1e3;
    static constexpr Real g_off   = 1e-9;

    // Graph + pool + cache
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index v_in_idx  = -1;
    Index v_sw_idx  = -1;
    Index v_out_idx = -1;
    Index i_L_idx   = -1;

    explicit BuckConverter(Real dt) {
        // Nodes (order locks the v_*_idx values).
        v_in_idx  = g.add_node("v_in");      // 0
        v_sw_idx  = g.add_node("v_sw");      // 1
        v_out_idx = g.add_node("v_out");     // 2

        // Branches (order locks branch_var_id offsets):
        //   0: V_in source
        //   1: Q1 (v_in → v_sw)
        //   2: Q2 (v_sw → GND)
        //   3: L  (v_sw → v_out)
        //   4: R  (v_out → GND)
        //   5: C  (v_out → GND)
        g.add_branch(v_in_idx,  g.ground(),  BranchKind::Source);
        g.add_branch(v_in_idx,  v_sw_idx,    BranchKind::Switch);
        g.add_branch(v_sw_idx,  g.ground(),  BranchKind::Switch);
        g.add_branch(v_sw_idx,  v_out_idx,   BranchKind::PassiveLinear);  // L
        g.add_branch(v_out_idx, g.ground(),  BranchKind::PassiveLinear);  // R
        g.add_branch(v_out_idx, g.ground(),  BranchKind::PassiveLinear);  // C

        pool.add_voltage_source(0, {.V = V_in});
        pool.add_switch(1, g_on, g_off);
        pool.add_switch(2, g_on, g_off);
        pool.add_inductor(3, {.L = L});
        pool.add_resistor(4, {.G = G_load});
        pool.add_capacitor(5, {.C = C});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build(dt);

        i_L_idx = pool.branch_var_id_for_inductor(3, g);
    }

    /// Complementary PWM: Q1 ON during [0, D·T); Q2 ON during
    /// [D·T, T). Exactly one switch on at any time.
    static SwitchStateMask pwm_schedule(Real t) {
        const Real phase = std::fmod(t, T_sw);
        const bool q1_on = phase < D * T_sw;
        SwitchStateMask mask(2);
        mask.set(0, q1_on);
        mask.set(1, !q1_on);
        return mask;
    }
};

}  // namespace

TEST_CASE(
    "Synchronous buck converter: cache builds 4 segments",
    "[v2][layer5_v1][integration][buck]") {
    BuckConverter buck(1e-7);   // dt = 100 ns
    REQUIRE(buck.cache->num_segments() == 4);
}

TEST_CASE(
    "Synchronous buck converter: steady-state matches analytical",
    "[v2][layer5_v1][integration][buck]") {
    constexpr Real dt = 1e-7;                  // 100 ns
    // With R_load = 1 Ω the output LC tank has ζ = 0.5,
    // settling in ~0.8 ms. Simulate 5 ms (well past settling) and
    // measure the last 1 ms (= 100 PWM periods) for steady-state
    // metrics.
    constexpr Real t_end = 5e-3;               // 5 ms
    BuckConverter buck(dt);

    SimulationOptions opts{
        .t_start = 0,
        .t_end   = t_end,
        .dt      = dt,
    };
    REQUIRE(opts.valid());

    const auto t_wall_start = std::chrono::steady_clock::now();
    auto result = run_transient(
        *buck.cache, buck.g, buck.pool, opts,
        &BuckConverter::pwm_schedule);
    const auto t_wall_end = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            t_wall_end - t_wall_start).count();

    REQUIRE(result.num_steps() == 50001);

    // -------- Steady-state measurement window: last 1 ms
    //          (= 100 PWM periods). At ζ = 0.5 the residual LC
    //          envelope at t = 4 ms is e^{-4/0.8·0.5} ≈ 1 % of
    //          initial, so the measurement reflects the
    //          PWM-driven steady state.
    constexpr Real meas_window = 1e-3;
    const Size k_start = static_cast<Size>(
        (t_end - meas_window) / dt);

    Real sum_v_out = 0;
    Real sum_i_L   = 0;
    Real min_v_out =  std::numeric_limits<Real>::infinity();
    Real max_v_out = -std::numeric_limits<Real>::infinity();
    Real min_i_L   =  std::numeric_limits<Real>::infinity();
    Real max_i_L   = -std::numeric_limits<Real>::infinity();
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real v_out = result.states[k][buck.v_out_idx];
        const Real i_L   = result.states[k][buck.i_L_idx];
        sum_v_out += v_out;
        sum_i_L   += i_L;
        min_v_out = std::min(min_v_out, v_out);
        max_v_out = std::max(max_v_out, v_out);
        min_i_L   = std::min(min_i_L,   i_L);
        max_i_L   = std::max(max_i_L,   i_L);
    }
    const Size n_meas = result.num_steps() - k_start;
    const Real mean_v_out = sum_v_out / static_cast<Real>(n_meas);
    const Real mean_i_L   = sum_i_L   / static_cast<Real>(n_meas);
    const Real delta_v_out = max_v_out - min_v_out;
    const Real delta_i_L   = max_i_L   - min_i_L;

    // Analytical CCM expectations
    const Real V_out_ana = BuckConverter::V_in * BuckConverter::D;
    const Real I_L_ana   = V_out_ana / BuckConverter::R_load;
    const Real dI_L_ana  = BuckConverter::V_in * BuckConverter::D *
                            (Real{1} - BuckConverter::D) /
                            (BuckConverter::L * BuckConverter::f_sw);
    const Real dV_out_ana = dI_L_ana /
                             (Real{8} * BuckConverter::C *
                              BuckConverter::f_sw);

    INFO("Buck converter steady-state (last 10 PWM periods):");
    INFO("  mean V_out  = " << mean_v_out << " V (analytical: "
         << V_out_ana << " V)");
    INFO("  mean I_L    = " << mean_i_L   << " A (analytical: "
         << I_L_ana   << " A)");
    INFO("  ΔI_L  pk-pk = " << delta_i_L  << " A (analytical: "
         << dI_L_ana  << " A)");
    INFO("  ΔV_out pk-pk = " << delta_v_out * 1000 << " mV (analytical: "
         << dV_out_ana * 1000 << " mV)");
    INFO("  Wall clock  = " << elapsed_ms << " ms for "
         << result.num_steps() << " samples");

    // -------- Analytical validation.
    REQUIRE(mean_v_out == Approx(V_out_ana).epsilon(0.05));
    REQUIRE(mean_i_L   == Approx(I_L_ana).epsilon(0.05));
    REQUIRE(delta_i_L  == Approx(dI_L_ana).epsilon(0.15));
    REQUIRE(delta_v_out == Approx(dV_out_ana).epsilon(0.30));

    // Wall-clock budget — generous (Debug + sanitizer builds are slow;
    // the CI's "Linux Debug + Sanitizers" job runs ASan + UBSan, which
    // bumps the runtime past 5 s on shared GHA runners).
    REQUIRE(elapsed_ms < 15000);
}
