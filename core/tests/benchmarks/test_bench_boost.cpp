// =============================================================================
// Benchmark — synchronous-style boost converter (open-loop CCM)
// =============================================================================
//
// Topology:
//
//   Vin ──[L]── sw ──[Q]── gnd               (Q low-side)
//                    └──[D]── vout ──┬──[Rload]── gnd
//                                     └──[Cout ]── gnd
//
// Q (controlled switch) drives a 100 kHz / D = 0.5 square wave.
// D is event-driven (DiodeEventState handles commutation).
//
// Steady-state CCM theory:
//   V_out = V_in / (1 − D)        (24 V for 12 V → 0.5 duty)
//   ΔV_out / V_out ≈ D · T_sw / (R · C)
//
// The bench measures:
//   * cache build time (2² = 4 switch configs)
//   * per-step solve cost over 10 ms (∼1 000 PWM cycles)
//   * mean V_out over the last 0.5 ms (≈ 50 cycles)
//   * peak-to-peak ripple in that window

#include <catch2/catch_test_macros.hpp>

#include "bench_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace pulsim;
using namespace pulsim::bench;

namespace {

constexpr Real V_in    = 12.0;
constexpr Real L       = 100e-6;
constexpr Real C_out   = 100e-6;
constexpr Real R_load  = 20.0;       // P_out ≈ 28.8 W (matches buck bench)
constexpr Real f_sw    = 100e3;
constexpr Real T_sw    = 1.0 / f_sw;
constexpr Real duty    = 0.5;

builder::CircuitBuilder make_boost() {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vin",   "vin", "gnd",  V_in);
    b.add_inductor      ("L1",    "vin", "sw",   L);
    b.add_switch        ("Q",     "sw",  "gnd",  /*g_on=*/1e3, /*g_off=*/1e-9);
    b.add_diode         ("D",     "sw",  "vout",
                         /*g_on=*/1e3, /*g_off=*/1e-9, /*V_th=*/0.0);
    b.add_capacitor     ("Cout",  "vout","gnd",  C_out);
    b.add_resistor      ("Rload", "vout","gnd",  R_load);
    return b;
}

}  // namespace

TEST_CASE("Bench: boost converter — steady-state V_out = V_in / (1−D)",
          "[bench][boost][smps]") {
    print_report_header();

    auto cb = make_boost();
    const int vout_idx = static_cast<int>(cb.node_id_of("vout"));

    // PWM switch_fn: Q (bit 0) ON during [0, duty·T_sw).
    // D (bit 1) is event-driven; we always "arm" it and rely on
    // DiodeEventState to flip it on i_D / v_D crossings.
    auto switch_fn = [](Real t) {
        const auto n_sw = Size{2};
        topology::SwitchStateMask m(n_sw);
        const Real phase = std::fmod(t, T_sw) / T_sw;
        if (phase < duty) m.set(0, true);    // Q ON
        m.set(1, true);                       // arm D (event-driven)
        return m;
    };

    solver::SimulationOptions opts{
        .t_start = 0.0,
        .t_end   = 10e-3,          // 10 ms = 1000 PWM cycles
        .dt      = T_sw / 100.0,    // 1 µs steps → 100 samples/cycle
    };

    auto rpt = time_and_validate(
        "boost open-loop (1000 cycles)", cb, opts,
        /*probe_state_idx=*/-1, /*analytical_fn=*/{},
        switch_fn);
    print_report_row(rpt);

    // -------------------------------------------------------------------
    // Re-run to get the trace; the helper threw it away after timing.
    // -------------------------------------------------------------------
    pwl::PwlStateSpaceCache cache(cb.graph(), cb.pool());
    cache.build(opts.dt);
    auto sim = solver::run_transient(
        cache, cb.graph(), cb.pool(), opts, switch_fn);

    // Steady-state window: last 0.5 ms = 50 cycles.
    const Size window_steps = static_cast<Size>(0.5e-3 / opts.dt);
    const Size k_start =
        sim.times.size() > window_steps
            ? sim.times.size() - window_steps : 0;

    Real v_sum = 0;
    Real v_max = -1e9, v_min = 1e9;
    for (Size k = k_start; k < sim.times.size(); ++k) {
        const Real v = sim.states[k][vout_idx];
        v_sum += v;
        v_max  = std::max(v_max, v);
        v_min  = std::min(v_min, v);
    }
    const Real v_mean   = v_sum / static_cast<Real>(sim.times.size() - k_start);
    const Real v_ripple = v_max - v_min;
    const Real v_ideal  = V_in / (1.0 - duty);

    std::printf(" boost V_out: mean = %.3f V (target %.3f V),"
                " ripple = %.3f V p-p\n",
                v_mean, v_ideal, v_ripple);

    // CCM theory: V_out = V_in / (1 − D).  Switch on-resistance and
    // diode forward drop pull this a few % below ideal.  Allow 2 V.
    REQUIRE(std::abs(v_mean - v_ideal) < 2.0);
    // Ripple bound (continuous):
    //   ΔV/V ≈ D·T_sw / (R·C) = 0.5 · 1e-5 / (20 · 1e-4) = 0.25 %
    //   ΔV     ≈ 24 · 0.0025  ≈ 60 mV
    // Real-circuit ripple inflated by event jitter / ESR; cap at 1 V.
    REQUIRE(v_ripple < 1.0);
}
