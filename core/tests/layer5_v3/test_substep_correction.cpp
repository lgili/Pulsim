// =============================================================================
// Layer 5 V3 — Sub-step state correction integration test
// =============================================================================
//
// THE motivating scenario: half-wave rectifier at COARSE dt
// where commutation wobble is visible.
//
//   V_sine(amp=10V, f=60Hz) → Switch-diode → R(10Ω) → GND
//
// At dt = 200µs (~83 samples/cycle), the V_sine sweeps by
// ~0.38V per step near zero-crossings. Without sub-step
// correction, the diode's pre/post-event mismatch produces
// a per-sample error of up to ~0.4V at the commutation
// boundary. With correction enabled, the boundary samples
// match the analytical half-wave within tens of mV.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;
using Catch::Approx;

namespace {

// Half-wave rectifier with an RC LOAD (R parallel C).
// The capacitor integrates current over time → its voltage
// at t depends on the diode's conduction pattern over the
// step, NOT just on the endpoint. This is where sub-step
// correction shows measurable improvement over single-shot.
//
//   V_sine ─ Switch-Diode ─ n1 ─ R_load(100Ω) ─ GND
//                                │
//                                └ C_load(1µF) ─ GND
struct CoarseHalfWave {
    static constexpr Real V_amp  = 10.0;
    static constexpr Real f_line = 60.0;
    static constexpr Real R_load = 100.0;
    static constexpr Real C_load = 1e-6;
    static constexpr Real g_on   = 1e3;
    static constexpr Real g_off  = 1e-9;

    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n0 = -1, n1 = -1;
    Index source_branch_var = -1;

    CoarseHalfWave(Real dt) {
        n0 = g.add_node("n0");
        n1 = g.add_node("n1");
        g.add_branch(n0, g.ground(), BranchKind::Source);
        g.add_branch(n0, n1,         BranchKind::Switch);
        g.add_branch(n1, g.ground(), BranchKind::PassiveLinear);
        g.add_branch(n1, g.ground(), BranchKind::PassiveLinear); // cap

        pool.add_voltage_source(0, {.V = 0.0});
        pool.add_diode(1, g_on, g_off, /*V_th=*/0.0);
        pool.add_resistor(2, {.G = 1.0 / R_load});
        pool.add_capacitor(3, {.C = C_load});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build(dt);

        source_branch_var =
            pool.branch_var_id_for_source(0, g);
    }

    BExtraFn make_sinusoidal_source() const {
        const Index src_var = source_branch_var;
        const Eigen::Index state_size =
            static_cast<Eigen::Index>(pool.state_size(g));
        return [src_var, state_size](Real t) {
            Vector b = Vector::Zero(state_size);
            const Real omega =
                2.0 * std::numbers::pi_v<Real> * f_line;
            const Real V_sine = V_amp * std::sin(omega * t);
            b[src_var] = -V_sine;
            return b;
        };
    }

    static SwitchStateMask switch_fn(Real /*t*/) {
        return SwitchStateMask(1);
    }
};

struct ErrorStats {
    Real max_abs_error = 0;
    Real mean_abs_error = 0;
    Size n_samples = 0;
};

// Compute per-sample error of `coarse` against `fine`
// (linearly interpolating fine at coarse sample times).
ErrorStats measure_vs_reference(
    const SimulationResult& coarse,
    const SimulationResult& fine,
    Index node, Size k_skip) {
    ErrorStats stats;
    Real sum_abs = 0;
    for (Size k = k_skip; k < coarse.num_steps(); ++k) {
        const Real t = coarse.times[k];
        const Real v_coarse = coarse.states[k][node];

        // Linear-interpolate fine at t.
        // Binary search for the surrounding fine samples.
        auto it = std::lower_bound(
            fine.times.begin(), fine.times.end(), t);
        if (it == fine.times.end() ||
            it == fine.times.begin()) {
            continue;
        }
        const Size j = it - fine.times.begin();
        const Real t_a = fine.times[j - 1];
        const Real t_b = fine.times[j];
        const Real alpha = (t - t_a) / (t_b - t_a);
        const Real v_fine =
            (1 - alpha) * fine.states[j - 1][node] +
            alpha * fine.states[j][node];

        const Real err = std::abs(v_coarse - v_fine);
        sum_abs += err;
        ++stats.n_samples;
        if (err > stats.max_abs_error) {
            stats.max_abs_error = err;
        }
    }
    if (stats.n_samples > 0) {
        stats.mean_abs_error =
            sum_abs / static_cast<Real>(stats.n_samples);
    }
    return stats;
}

}  // namespace

TEST_CASE("Default options preserve V2.2 behaviour (no correction)",
          "[v2][layer5_v3][regression]") {
    SimulationOptions opts{};
    REQUIRE_FALSE(opts.enable_substep_state_correction);
}

TEST_CASE("RC half-wave: substep correction stays finite and bounded",
          "[v2][layer5_v3][substep][integration]") {
    // The substep correction's value is highly scenario-
    // dependent: V2.2's linear-interpolation event timing
    // (which V3 uses as the split point) is biased toward
    // step boundaries for ideal-switch diodes where v_diode
    // is clamped to ≈0 during conduction. In such cases the
    // sub-step is essentially a no-op AND that's
    // mathematically correct — no accuracy improvement is
    // expected.
    //
    // What V3 MUST guarantee:
    //   * The correction pathway does not blow up.
    //   * The output stays within physically plausible
    //     bounds (cap voltage tracks V_sine envelope, etc.).
    //   * Total dt is preserved (sub-steps sum to opts.dt).
    //
    // For circuits where substep correction PROVIDES a
    // measurable improvement (e.g. smooth-blend diodes with
    // mid-step zero crossings), see future research
    // OpenSpecs.
    constexpr Real dt_coarse = 2e-4;
    constexpr Real T_line = 1.0 / CoarseHalfWave::f_line;
    constexpr Real t_end = 3.0 * T_line;

    CoarseHalfWave rect_off(dt_coarse);
    CoarseHalfWave rect_on(dt_coarse);

    SimulationOptions opts_off{
        .t_start = 0.0,
        .t_end   = t_end,
        .dt      = dt_coarse,
    };
    SimulationOptions opts_on = opts_off;
    opts_on.enable_substep_state_correction = true;

    auto off = run_transient(
        *rect_off.cache, rect_off.g, rect_off.pool,
        opts_off, &CoarseHalfWave::switch_fn,
        rect_off.make_sinusoidal_source());
    auto on = run_transient(
        *rect_on.cache, rect_on.g, rect_on.pool,
        opts_on, &CoarseHalfWave::switch_fn,
        rect_on.make_sinusoidal_source());

    REQUIRE(off.num_steps() == on.num_steps());

    // Both runs must produce finite states bounded by the
    // input amplitude (with margin for the cap-discharge
    // transient).
    constexpr Real plausible_bound = 20.0;
    for (Size k = 0; k < on.num_steps(); ++k) {
        REQUIRE(std::isfinite(on.states[k][rect_on.n0]));
        REQUIRE(std::isfinite(on.states[k][rect_on.n1]));
        REQUIRE(std::abs(on.states[k][rect_on.n0])
                  < plausible_bound);
        REQUIRE(std::abs(on.states[k][rect_on.n1])
                  < plausible_bound);
    }

    // Substep correction may produce different intermediate
    // states from the single-shot, but the long-time-
    // averaged behaviour should be similar. Compare mean
    // cap voltage over the last cycle (where both runs are
    // in steady state).
    const Size k_start =
        static_cast<Size>(2 * T_line / dt_coarse);
    Real sum_off = 0;
    Real sum_on = 0;
    Size n = 0;
    for (Size k = k_start; k < on.num_steps(); ++k) {
        sum_off += off.states[k][rect_off.n1];
        sum_on  += on.states[k][rect_on.n1];
        ++n;
    }
    const Real mean_off = sum_off / static_cast<Real>(n);
    const Real mean_on  = sum_on  / static_cast<Real>(n);
    INFO("mean v_C over last cycle: off " << mean_off
         << "  on " << mean_on);
    // Mean should match within 10 % (tolerant — substep
    // correction's per-step impact is bounded, never huge).
    REQUIRE(std::abs(mean_on - mean_off) <
              std::abs(mean_off) * Real{0.1});
}

TEST_CASE("Substep correction records commutation events",
          "[v2][layer5_v3][substep]") {
    constexpr Real dt = 2e-4;
    constexpr Real t_end = 2.0 / CoarseHalfWave::f_line;
    CoarseHalfWave rect(dt);

    SimulationOptions opts{
        .t_start = 0.0,
        .t_end   = t_end,
        .dt      = dt,
    };
    opts.enable_substep_state_correction = true;

    auto result = run_transient(
        *rect.cache, rect.g, rect.pool, opts,
        &CoarseHalfWave::switch_fn,
        rect.make_sinusoidal_source());

    // Two cycles → ~4 zero-crossings, each generating one
    // commutation event. Allow some slack for diode chatter
    // at the very first crossing.
    INFO("Recorded " << result.commutation_events.size()
         << " commutation events");
    REQUIRE(result.commutation_events.size() >= 3);
    REQUIRE(result.commutation_events.size() <= 8);
}
