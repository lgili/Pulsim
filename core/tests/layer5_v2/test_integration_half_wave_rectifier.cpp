// =============================================================================
// Layer 5 V2 — Integration: half-wave rectifier
// =============================================================================
//
// Simplest diode use case.
//
//   V_sine ──[Source]── n0 ──[Diode]── n1 ──[R_load]── GND
//
// V_sine is a 60 Hz sinusoid (V_amp = 10 V). The voltage-source
// branch is registered with V = 0 in the pool; the actual time-
// varying value comes in via `b_extra_fn` modulating the source's
// constraint row each step.
//
// Diode behaviour:
//   * V_sine > 0 → diode forward-biased → ON → output ≈ V_sine
//   * V_sine < 0 → diode reverse-biased → OFF → output ≈ 0
//
// Mean output power of an ideal half-wave rectifier:
//   P_avg = V_amp² / (4·R)

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <memory>
#include <numbers>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;
using Catch::Approx;

namespace {

struct HalfWaveRectifier {
    static constexpr Real V_amp  = 10.0;
    static constexpr Real f_line = 60.0;
    static constexpr Real R_load = 10.0;
    static constexpr Real g_on   = 1e3;
    static constexpr Real g_off  = 1e-9;

    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n0 = -1, n1 = -1;
    Index source_branch_var = -1;

    HalfWaveRectifier(Real dt) {
        n0 = g.add_node("n0");
        n1 = g.add_node("n1");
        g.add_branch(n0, g.ground(), BranchKind::Source);   // V_sine source
        g.add_branch(n0, n1,         BranchKind::Switch);   // diode
        g.add_branch(n1, g.ground(), BranchKind::PassiveLinear);  // R_load

        pool.add_voltage_source(0, {.V = 0.0});             // V via b_extra
        pool.add_diode(1, g_on, g_off, /*V_th=*/0.0);
        pool.add_resistor(2, {.G = 1.0 / R_load});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build(dt);

        source_branch_var =
            pool.branch_var_id_for_source(0, g);
    }

    /// Returns b_extra(t) that overlays a sinusoidal voltage on
    /// the source's constraint row. The constraint reads
    /// (v_n0 - 0) - V = 0; with V_constant = 0 (registered),
    /// the assembled b_constant has 0 on the constraint row, and
    /// the source's effective voltage is -b_extra(constraint_row).
    /// So setting b_extra = -V_sine(t) gives effective V_dc =
    /// +V_sine(t).
    BExtraFn make_sinusoidal_source() const {
        const Index src_var = source_branch_var;
        const Eigen::Index state_size =
            static_cast<Eigen::Index>(pool.state_size(g));
        return [src_var, state_size](Real t) {
            Vector b = Vector::Zero(state_size);
            const Real omega = 2.0 * std::numbers::pi_v<Real> * f_line;
            const Real V_sine = V_amp * std::sin(omega * t);
            b[src_var] = -V_sine;
            return b;
        };
    }

    /// Diode is the only switch in this circuit, so the user's
    /// switch_fn just returns an empty mask — the bit at position
    /// 0 will be overwritten by the diode tracker.
    static SwitchStateMask switch_fn(Real /*t*/) {
        return SwitchStateMask(1);
    }
};

}  // namespace

TEST_CASE("Half-wave rectifier: output tracks sine on positive half",
          "[v2][layer5_v2][integration][half_wave]") {
    constexpr Real dt = 1e-5;           // 10 µs (1700 samples / cycle)
    constexpr Real T_line = 1.0 / HalfWaveRectifier::f_line;
    constexpr Real t_end = 2.0 * T_line;   // 2 full cycles
    HalfWaveRectifier rect(dt);

    SimulationOptions opts{
        .t_start = 0,
        .t_end   = t_end,
        .dt      = dt,
    };

    auto result = run_transient(
        *rect.cache, rect.g, rect.pool, opts,
        &HalfWaveRectifier::switch_fn,
        rect.make_sinusoidal_source());

    REQUIRE(result.num_steps() > 3000);  // 2 cycles · 1670 samples

    const Real omega =
        2.0 * std::numbers::pi_v<Real> * HalfWaveRectifier::f_line;

    Size n_pos_checked = 0, n_pos_match = 0;
    Size n_neg_checked = 0, n_neg_match = 0;
    Real sum_v_out_pos = 0, sum_v_out_neg = 0;

    // Skip the first half-period to clear any startup transient
    // (the first step's diode commutation takes ~1 dt to settle).
    const Size k_start = static_cast<Size>(T_line / 2.0 / dt);

    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real t = result.times[k];
        const Real v_sine = HalfWaveRectifier::V_amp * std::sin(omega * t);
        const Real v_out  = result.states[k][rect.n1];

        if (v_sine > 0.5) {
            // Positive half-cycle. Expect v_out ≈ v_sine (small
            // drop through the diode's g_on resistive divider).
            ++n_pos_checked;
            if (std::abs(v_out - v_sine) < 0.5) ++n_pos_match;
            sum_v_out_pos += v_out;
        } else if (v_sine < -0.5) {
            // Negative half-cycle. Expect |v_out| ≈ 0 (diode OFF).
            ++n_neg_checked;
            if (std::abs(v_out) < 0.1) ++n_neg_match;
            sum_v_out_neg += v_out;
        }
    }

    INFO("Half-wave: positive-half match " << n_pos_match
         << "/" << n_pos_checked);
    INFO("Half-wave: negative-half match " << n_neg_match
         << "/" << n_neg_checked);

    // Allow a few mismatches per cycle for the boundary samples
    // (where v_sine ≈ 0 and the diode is right at commutation).
    REQUIRE(n_pos_checked > 0);
    REQUIRE(n_neg_checked > 0);
    REQUIRE(n_pos_match >= n_pos_checked * 99 / 100);
    REQUIRE(n_neg_match >= n_neg_checked * 99 / 100);
}

TEST_CASE("Half-wave rectifier: mean output power matches analytical",
          "[v2][layer5_v2][integration][half_wave]") {
    constexpr Real dt = 1e-5;
    constexpr Real T_line = 1.0 / HalfWaveRectifier::f_line;
    constexpr Real t_end = 2.0 * T_line;
    HalfWaveRectifier rect(dt);

    SimulationOptions opts{
        .t_start = 0,
        .t_end   = t_end,
        .dt      = dt,
    };
    auto result = run_transient(
        *rect.cache, rect.g, rect.pool, opts,
        &HalfWaveRectifier::switch_fn,
        rect.make_sinusoidal_source());

    // Average over the LAST full cycle.
    const Size k_start = static_cast<Size>(T_line / dt);
    Real sum_power = 0;
    Size n = 0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real v_out = result.states[k][rect.n1];
        sum_power += (v_out * v_out) / HalfWaveRectifier::R_load;
        ++n;
    }
    const Real mean_power = sum_power / static_cast<Real>(n);

    // Analytical half-wave rectifier mean power:
    //   P = V_amp² / (4·R) = 100 / 40 = 2.5 W
    const Real P_ana =
        HalfWaveRectifier::V_amp * HalfWaveRectifier::V_amp
        / (4.0 * HalfWaveRectifier::R_load);
    INFO("Half-wave: mean output power = " << mean_power
         << " W (analytical: " << P_ana << " W)");
    REQUIRE(mean_power == Approx(P_ana).epsilon(0.05));
}
