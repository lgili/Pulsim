// =============================================================================
// Layer 5 — Integration: chopper at 10 kHz PWM
// =============================================================================
//
//   V_dc ──[Source]── vin ──[Switch M1]── vout ──[R_load]── GND
//
// Drive M1 with a 10 kHz, 50 %-duty PWM schedule. Run 1 ms (= 10
// full PWM periods) at dt = 1 µs (= 1001 simulation steps).
//
// This is the FIRST end-to-end demonstration that the v2
// architecture works: build a graph → register devices → build
// cache → call run_transient → get back (t, x(t)) waveforms. The
// per-step hot path is ~1 µs (one map probe + one O(nnz)
// triangular solve on a cached factor). 1000 steps = ~ms total.
//
// Verify:
//   * num_steps == 1001
//   * mean(v_out) ≈ V_dc · duty = 6 V  within 1 %
//   * waveform is a clean square wave between the analytical ON
//     and OFF values
//   * total wall-clock is well under 1 second

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/solver/options.hpp"
#include "pulsim/v2/solver/result.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <chrono>
#include <cmath>
#include <memory>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::topology;
using Catch::Approx;

namespace {

struct Chopper {
    Graph g;
    DevicePool pool;
    std::unique_ptr<PwlStateSpaceCache> cache;
    Index n_in = -1;
    Index n_out = -1;
    Real V_dc = 12.0;
    Real g_on = 1e3;
    Real g_off = 1e-9;
    Real G_R = 0.1;
    Size state_size = 0;

    Chopper() {
        n_in  = g.add_node("vin");
        n_out = g.add_node("vout");
        g.add_branch(n_in, g.ground(), BranchKind::Source);
        g.add_branch(n_in, n_out,      BranchKind::Switch);
        g.add_branch(n_out, g.ground(),BranchKind::PassiveLinear);

        pool.add_voltage_source(0, {V_dc});
        pool.add_switch(1, g_on, g_off);
        pool.add_resistor(2, {G_R});

        cache = std::make_unique<PwlStateSpaceCache>(g, pool);
        cache->build();
        state_size = pool.state_size(g);
    }

    // 10 kHz, duty=0.5 PWM schedule. Returns ON when
    // `fmod(t, T) < duty · T`, OFF otherwise.
    static bool pwm_on(Real t) {
        const Real T_pwm = Real{1e-4};   // 10 kHz period = 100 µs
        const Real duty = Real{0.5};
        const Real phase = std::fmod(t, T_pwm);
        return phase < duty * T_pwm;
    }
};

}  // namespace

TEST_CASE(
    "Chopper PWM: 10 kHz / 50% duty / 1 ms simulation → 1001 samples",
    "[v2][layer5][integration][chopper][pwm]") {
    Chopper c;
    SimulationOptions opts{
        .t_start = 0,
        .t_end   = Real{1e-3},   // 1 ms
        .dt      = Real{1e-6},   // 1 µs
    };
    REQUIRE(opts.valid());
    REQUIRE(opts.expected_step_count() == 1001);

    SwitchScheduleFn pwm_schedule = [](Real t) {
        SwitchStateMask mask(1);
        mask.set(0, Chopper::pwm_on(t));
        return mask;
    };

    const auto t_wall_start = std::chrono::steady_clock::now();
    SimulationResult result = run_transient(
        *c.cache, c.state_size, opts, pwm_schedule);
    const auto t_wall_end = std::chrono::steady_clock::now();

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            t_wall_end - t_wall_start).count();
    INFO("1001-step PWM simulation took " << elapsed_ms << " ms");

    REQUIRE(result.num_steps() == 1001);
    // Performance smoke — extremely generous bound. The hot
    // path is a hash probe + triangular solve, so this should
    // be well under 100 ms on the test host.
    REQUIRE(elapsed_ms < 1000);
}

TEST_CASE(
    "Chopper PWM: mean(v_out) ≈ V_dc · duty (50 %) within 1 %",
    "[v2][layer5][integration][chopper][pwm]") {
    Chopper c;
    SimulationOptions opts{
        .t_start = 0, .t_end = Real{1e-3}, .dt = Real{1e-6}};
    SwitchScheduleFn pwm_schedule = [](Real t) {
        SwitchStateMask mask(1);
        mask.set(0, Chopper::pwm_on(t));
        return mask;
    };

    SimulationResult result = run_transient(
        *c.cache, c.state_size, opts, pwm_schedule);

    // Compute the time-domain mean of v_out across all samples.
    Real sum_v_out = Real{0};
    for (Size k = 0; k < result.num_steps(); ++k) {
        sum_v_out += result.states[k][c.n_out];
    }
    const Real mean_v_out =
        sum_v_out / static_cast<Real>(result.num_steps());
    const Real expected = c.V_dc * Real{0.5};   // 6 V

    INFO("mean(v_out) = " << mean_v_out
         << " V, expected = " << expected << " V");

    // The instantaneous ON value is slightly below V_dc (resistive
    // divider with g_on=1e3, G_R=0.1), so mean is slightly less
    // than 6 V. 1 % tolerance is comfortable.
    REQUIRE(mean_v_out == Approx(expected).epsilon(0.01));
}

TEST_CASE(
    "Chopper PWM: waveform is a clean square wave",
    "[v2][layer5][integration][chopper][pwm]") {
    Chopper c;
    SimulationOptions opts{
        .t_start = 0, .t_end = Real{1e-3}, .dt = Real{1e-6}};
    SwitchScheduleFn pwm_schedule = [](Real t) {
        SwitchStateMask mask(1);
        mask.set(0, Chopper::pwm_on(t));
        return mask;
    };

    SimulationResult result = run_transient(
        *c.cache, c.state_size, opts, pwm_schedule);

    const Real expected_on =
        c.V_dc * c.g_on / (c.g_on + c.G_R);
    const Real expected_off =
        c.V_dc * c.g_off / (c.g_off + c.G_R);

    INFO("expected_on = " << expected_on
         << " V, expected_off = " << expected_off << " V");

    // Every sample is either at the ON value or at the OFF value.
    Size n_on = 0, n_off = 0;
    for (Size k = 0; k < result.num_steps(); ++k) {
        const Real t = result.times[k];
        const Real v_out = result.states[k][c.n_out];
        const bool is_on = Chopper::pwm_on(t);
        if (is_on) {
            INFO("ON  k=" << k << " t=" << t << " v_out=" << v_out);
            REQUIRE(v_out == Approx(expected_on).margin(1e-6));
            ++n_on;
        } else {
            INFO("OFF k=" << k << " t=" << t << " v_out=" << v_out);
            REQUIRE(v_out == Approx(expected_off).margin(1e-6));
            ++n_off;
        }
    }

    // At 50 % duty and 1001 samples, the ON/OFF counts should be
    // close to equal (within the boundary samples at period
    // transitions).
    INFO("n_on = " << n_on << ", n_off = " << n_off);
    REQUIRE(n_on + n_off == 1001);
    REQUIRE(static_cast<Real>(n_on) / 1001.0 == Approx(0.5).epsilon(0.05));
}
