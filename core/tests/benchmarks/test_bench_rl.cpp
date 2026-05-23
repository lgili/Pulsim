// =============================================================================
// Benchmark — RL ramp-up vs analytical
// =============================================================================
//
// Topology:  Vdc ──[R]── nL ──[L]── gnd
//
// Inductor current ramp: i_L(t) = (V/R) · (1 − e^(−t/τ)), τ = L/R.
// Steady-state: I_∞ = V/R.

#include <catch2/catch_test_macros.hpp>

#include "bench_helpers.hpp"

#include <cmath>

using namespace pulsim;
using namespace pulsim::bench;

namespace {

builder::CircuitBuilder make_rl(Real V_dc, Real R, Real L) {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vdc", "n0", "gnd", V_dc);
    b.add_resistor      ("R",   "n0", "nL",  R);
    b.add_inductor      ("L",   "nL", "gnd", L);
    return b;
}

}  // namespace

TEST_CASE("Bench: RL ramp-up — analytical agreement",
          "[bench][rl][analytical]") {
    print_report_header();

    const Real V_dc  = 12.0;
    const Real R     = 10.0;       // 10 Ω
    const Real L     = 1.0e-3;     // 1 mH
    const Real tau   = L / R;      // 100 µs

    // Inductor current is at the LAST state index — after node
    // voltages and source branch currents.
    SECTION("dt = tau/100 over 5τ — current ramp accuracy") {
        auto cb = make_rl(V_dc, R, L);
        // Probe = state size − 1 (the single inductor branch current).
        const int probe = static_cast<int>(
            cb.pool().state_size(cb.graph())) - 1;
        solver::SimulationOptions opts{
            .t_start = 0.0,
            .t_end   = 5.0 * tau,
            .dt      = tau / 100.0,
        };
        auto rpt = time_and_validate(
            "RL tau/100 (current)", cb, opts, probe,
            [V_dc, R, tau](double t) {
                return (V_dc / R) * (1.0 - std::exp(-t / tau));
            });
        print_report_row(rpt);
        // I_∞ = V/R = 1.2 A. Trap rule half-step offset ≈ 6 mA.
        REQUIRE(rpt.max_abs_error < 0.02);
    }
}
