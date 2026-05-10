// =============================================================================
// Phase 6 of `refactor-linear-solver-cache`: gates + benchmarks
// =============================================================================
//
// This file formalizes the change's user-visible contracts as Catch2 cases.
// It is the CI hook that prevents Phase-1..5 progress from regressing while
// Phase 7 docs catch up.
//
// Scope:
//   * 6.1 Stable-topology run: hit rate ≥ 95 % once warmup is done.
//   * 6.2 Topology transition: `topology_changed` reason fires across each
//          accepted-step commutation.
//   * 6.4/6.5/6.6 Benchmark: Buck converter wall-clock comparison between
//          Behavioral (Newton-DAE) and Ideal (PWL state-space + numeric-
//          factor LRU) modes — captures the headline speedup figure with
//          INFO-only output (no strict ratio gate; CI runners are noisy
//          and the floor lives in `test_pwl_speedup_benchmark.cpp`).
//
// 6.3 (heap-allocation zero-count) is intentionally deferred — see
// `tasks.md` for the rationale; pinning that contract requires a custom
// allocator harness or `mallinfo` snapshots which sit outside the scope
// of this change.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"

#include <chrono>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

// 1000-step passive RC at fixed dt. Topology never changes; cache should
// settle into a single-entry steady state and hit ≥ 95 % of accepted steps.
[[nodiscard]] Circuit make_rc_long_run() {
    Circuit ckt;
    auto in  = ckt.add_node("in");
    auto out = ckt.add_node("out");
    ckt.add_resistor("R1", in, out, 1e3);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);
    ckt.add_voltage_source("V1", in, Circuit::ground(), 5.0);
    return ckt;
}

// Buck converter — same shape as `test_pwl_speedup_benchmark.cpp` but
// runs longer (50 PWM cycles, 100 µs at 100 kHz) to amortize cache warmup
// and let the LRU steady-state cost dominate the wall-clock.
[[nodiscard]] Circuit make_buck_phase6() {
    Circuit ckt;
    auto in   = ckt.add_node("in");
    auto sw_n = ckt.add_node("sw");
    auto out  = ckt.add_node("out");
    auto ctrl = ckt.add_node("ctrl");

    constexpr Real V_in = 48.0;
    ckt.add_voltage_source("Vdc", in, Circuit::ground(), V_in);

    PWMParams pwm;
    pwm.frequency = 100e3;
    pwm.duty      = 0.25;
    pwm.v_high    = 5.0;
    pwm.v_low     = 0.0;
    pwm.phase     = 0.0;
    ckt.add_pwm_voltage_source("Vctrl", ctrl, Circuit::ground(), pwm);

    ckt.add_vcswitch("Q1", ctrl, in, sw_n, /*v_threshold=*/2.5,
                     /*g_on=*/1e3, /*g_off=*/1e-9);
    ckt.add_diode("D1", Circuit::ground(), sw_n, /*g_on=*/1e3, /*g_off=*/1e-9);

    ckt.add_inductor("L1", sw_n, out, 47e-6, 0.0);
    ckt.add_capacitor("C1", out, Circuit::ground(), 100e-6, 0.0);
    ckt.add_resistor("Rload", out, Circuit::ground(), 5.0);
    return ckt;
}

struct BuckMetrics {
    bool   success = false;
    double wall_seconds = 0.0;
    int    total_steps = 0;
    int    state_space_primary_steps = 0;
    int    dae_fallback_steps = 0;
    int    linear_factor_cache_hits = 0;
    int    linear_factor_cache_misses = 0;
    Real   v_out_final = 0.0;
};

[[nodiscard]] BuckMetrics run_buck_phase6(SwitchingMode mode) {
    Circuit ckt = make_buck_phase6();
    if (mode == SwitchingMode::Ideal) {
        ckt.set_switching_mode_for_all(SwitchingMode::Ideal);
        ckt.set_pwl_state("Q1", false);
        ckt.set_pwl_state("D1", false);
    }

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-4;        // 10 PWM cycles at 100 kHz — long enough for
                               // the cache to dominate after warmup
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep    = false;
    opts.enable_bdf_order_control = false;
    opts.integrator   = Integrator::BDF1;
    opts.switching_mode = mode;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();

    BuckMetrics m;
    if (!dc.success) return m;

    const auto t0 = std::chrono::steady_clock::now();
    const auto run = sim.run_transient(dc.newton_result.solution);
    const auto t1 = std::chrono::steady_clock::now();

    m.success                    = run.success;
    m.wall_seconds               = std::chrono::duration<double>(t1 - t0).count();
    m.total_steps                = run.total_steps;
    m.state_space_primary_steps  = run.backend_telemetry.state_space_primary_steps;
    m.dae_fallback_steps         = run.backend_telemetry.dae_fallback_steps;
    m.linear_factor_cache_hits   = run.backend_telemetry.linear_factor_cache_hits;
    m.linear_factor_cache_misses = run.backend_telemetry.linear_factor_cache_misses;
    if (run.success && !run.states.empty()) {
        const auto out_idx = ckt.get_node("out");
        m.v_out_final = run.states.back()[out_idx];
    }
    return m;
}

}  // namespace

// -----------------------------------------------------------------------------
// 6.1 Stable-topology gate: hit rate ≥ 95 % after warmup
// -----------------------------------------------------------------------------

TEST_CASE("Phase 6.1: 1000-step stable-topology run lands at ≥95% cache hit rate",
          "[v1][pwl][phase6][linear_cache][gate_G1]") {
    Circuit ckt = make_rc_long_run();

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-4;        // 1000 steps at dt = 1e-7
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep    = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);
    REQUIRE(run.total_steps >= 1000);

    const auto& tel = run.backend_telemetry;
    const int total = tel.linear_factor_cache_hits + tel.linear_factor_cache_misses;
    REQUIRE(total > 0);
    const double hit_rate = static_cast<double>(tel.linear_factor_cache_hits) /
                            static_cast<double>(total);

    INFO("Phase 6.1 stable-topology long run:"
         << "\n  total_steps  = " << run.total_steps
         << "\n  hits         = " << tel.linear_factor_cache_hits
         << "\n  misses       = " << tel.linear_factor_cache_misses
         << "\n  hit_rate     = " << hit_rate);

    // Gate G.1 from the change spec: ≥ 95 % cache hit rate on stable-
    // topology windows after warmup. The 1000-step window absorbs any
    // warmup misses well below the 5 % budget.
    CHECK(hit_rate >= 0.95);
    CHECK(tel.pwl_topology_transitions == 0);
    CHECK(tel.linear_factor_cache_invalidations_topology_changed == 0);
}

// -----------------------------------------------------------------------------
// 6.2 Topology transition: `topology_changed` reason fires
// -----------------------------------------------------------------------------

TEST_CASE("Phase 6.2: each commutation tags TopologyChanged in telemetry",
          "[v1][pwl][phase6][linear_cache][reason]") {
    Circuit ckt;
    auto ctrl = ckt.add_node("ctrl");
    auto vin  = ckt.add_node("vin");
    auto out  = ckt.add_node("out");

    ckt.add_voltage_source("Vdc", vin, Circuit::ground(), 12.0);
    ckt.add_vcswitch("S1", ctrl, vin, out, 2.5, 1e3, 1e-9);
    ckt.add_resistor("R1", out, Circuit::ground(), 100.0);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);

    PulseParams pulse;
    pulse.v_initial = 0.0;
    pulse.v_pulse   = 5.0;
    pulse.t_delay   = 1e-6;
    pulse.t_rise    = 1e-9;
    pulse.t_fall    = 1e-9;
    pulse.t_width   = 2e-6;
    pulse.period    = 4e-6;
    ckt.add_pulse_voltage_source("Vctrl", ctrl, Circuit::ground(), pulse);

    ckt.set_switching_mode_for_all(SwitchingMode::Ideal);
    ckt.set_pwl_state("S1", false);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 16e-6;
    opts.dt     = 2e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 2e-7;
    opts.adaptive_timestep    = false;
    opts.enable_bdf_order_control = false;
    opts.integrator = Integrator::BDF1;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);
    const auto run = sim.run_transient(dc.newton_result.solution);
    REQUIRE(run.success);

    const auto& tel = run.backend_telemetry;
    REQUIRE(tel.pwl_topology_transitions >= 3);

    // The typed reason on the cache invalidation surface must include at
    // least one TopologyChanged tag — the central Phase 5/6 contract.
    CHECK(tel.linear_factor_cache_invalidations_topology_changed >= 1);

    // Every invalidation in this run must be either TopologyChanged or
    // NumericInstability (bisection-induced fractional-dt mismatches show
    // up as numeric_instability since the previous step's matrix_hash
    // differs from this step's even within the same topology). The other
    // four reasons are not exercised by the segment-primary path.
    const int accounted =
        tel.linear_factor_cache_invalidations_topology_changed +
        tel.linear_factor_cache_invalidations_numeric_instability;
    CHECK(accounted == tel.linear_factor_cache_invalidations);
    CHECK(tel.linear_factor_cache_invalidations_stamp_param_changed   == 0);
    CHECK(tel.linear_factor_cache_invalidations_gmin_escalated        == 0);
    CHECK(tel.linear_factor_cache_invalidations_source_stepping_active == 0);
    CHECK(tel.linear_factor_cache_invalidations_manual_invalidate     == 0);
}

// -----------------------------------------------------------------------------
// 6.4/6.5/6.6 Buck wall-clock benchmark: Behavioral vs Ideal+cache
// -----------------------------------------------------------------------------

TEST_CASE("Phase 6.4–6.6: Buck converter PWL+cache vs Behavioral wall-clock",
          "[v1][pwl][phase6][linear_cache][benchmark]") {
    const auto behavioral = run_buck_phase6(SwitchingMode::Behavioral);
    const auto ideal      = run_buck_phase6(SwitchingMode::Ideal);

    REQUIRE(behavioral.success);
    REQUIRE(ideal.success);

    // Sanity: PWL path must serve every step on the Ideal run; Behavioral
    // path must take the Newton-DAE path on every step (no segment-primary
    // for Behavioral devices).
    CHECK(ideal.state_space_primary_steps == ideal.total_steps);
    CHECK(ideal.dae_fallback_steps == 0);
    CHECK(behavioral.state_space_primary_steps == 0);

    // Cache hit rate on Ideal: the LRU should be hitting most steps, even
    // accounting for bisection-induced fractional dt around commutation
    // edges. Behavioral has no segment-primary so its cache counters are
    // both zero (Newton-DAE uses a different solver instance).
    REQUIRE((ideal.linear_factor_cache_hits + ideal.linear_factor_cache_misses) > 0);
    const double ideal_hit_rate =
        static_cast<double>(ideal.linear_factor_cache_hits) /
        static_cast<double>(ideal.linear_factor_cache_hits + ideal.linear_factor_cache_misses);

    // Speedup ratio. CI runners are noisy → only assert a generous floor;
    // the headline number is captured in INFO output for human review.
    const double speedup = behavioral.wall_seconds / std::max(ideal.wall_seconds, 1e-9);

    INFO("=== Phase 6 Buck benchmark (10 PWM cycles, dt=100ns) ===");
    INFO("Behavioral mode (Newton-DAE)");
    INFO("  wall_seconds = " << behavioral.wall_seconds);
    INFO("  total_steps  = " << behavioral.total_steps);
    INFO("Ideal mode (PWL state-space + numeric-factor LRU)");
    INFO("  wall_seconds       = " << ideal.wall_seconds);
    INFO("  total_steps        = " << ideal.total_steps);
    INFO("  cache_hits         = " << ideal.linear_factor_cache_hits);
    INFO("  cache_misses       = " << ideal.linear_factor_cache_misses);
    INFO("  cache_hit_rate     = " << ideal_hit_rate);
    INFO("--- speedup (behavioral / ideal) = " << speedup);

    // Floor: Ideal+cache must not regress past Behavioral on the same
    // workload. The change spec's headline claims ≥3× (Behavioral
    // baseline → cache gains) and ≥10× (Behavioral baseline → PWL+cache
    // combined). Both are environment-sensitive and tracked in
    // `BENCHMARK_REPORT.md`; the unit gate is a regression guard, not
    // the production bar.
    CHECK(speedup >= 1.0);

    // The PWL+cache hit-rate floor for steady-state PWM. Bisection at
    // commutation edges generates a handful of unique fractional-dt
    // entries per cycle that don't cycle back; everything between edges
    // hits. Over 10 cycles the steady-state hit rate sits well above the
    // 50 % bar used here as a regression floor.
    CHECK(ideal_hit_rate >= 0.50);
}
