// =============================================================================
// Benchmark — RC step response vs analytical
// =============================================================================
//
// Topology:  Vdc ──[R]── vc ──[C]── gnd
//
// Step input from a constant 5 V voltage source. The capacitor
// charges through R, so v_C(t) = V·(1 − e^(−t/τ)), τ = RC.
//
// We sweep two grids: a "tight" grid (dt = τ/100) for accuracy
// check, and a "long" run (10 × τ over a finer dt) for timing.

#include <catch2/catch_test_macros.hpp>

#include "bench_helpers.hpp"

#include <cmath>

using namespace pulsim;
using namespace pulsim::bench;

namespace {

/// Build a fresh RC circuit. Always uses the same numeric values:
/// V = 5 V, R = 1 kΩ, C = 1 µF → τ = 1 ms.
builder::CircuitBuilder make_rc(Real V_dc, Real R, Real C) {
    builder::CircuitBuilder b;
    b.add_voltage_source("Vdc", "n0", "gnd", V_dc);
    b.add_resistor      ("R",   "n0", "vc",  R);
    b.add_capacitor     ("C",   "vc", "gnd", C);
    return b;
}

}  // namespace

TEST_CASE("Bench: RC step response — analytical agreement",
          "[bench][rc][analytical]") {
    print_report_header();

    const Real V_dc = 5.0;
    const Real R    = 1.0e3;
    const Real C    = 1.0e-6;
    const Real tau  = R * C;

    auto analytical = [V_dc, tau](double t) {
        return V_dc * (1.0 - std::exp(-t / tau));
    };

    // Tolerances expressed as ABSOLUTE error in volts. Relative error
    // is reported as well but isn't gated — it inflates near t=0 (when
    // analytical ≈ 0) because the trap rule has its own half-step
    // sample-time offset, and that's a sampling-convention artefact,
    // not a solver bug.
    SECTION("dt = tau/100 (1% of time constant)") {
        auto cb = make_rc(V_dc, R, C);
        const int probe_idx = static_cast<int>(cb.node_id_of("vc"));
        solver::SimulationOptions opts{
            .t_start = 0.0,
            .t_end   = 5.0 * tau,
            .dt      = tau / 100.0,
        };
        auto rpt = time_and_validate(
            "RC tau/100", cb, opts, probe_idx, analytical);
        print_report_row(rpt);
        // dt/τ = 0.01 → half-step trap-rule offset ≈ V·dt/(2τ) =
        // 25 mV peak. Allow 50 mV to keep headroom.
        REQUIRE(rpt.max_abs_error < 0.05);
    }

    SECTION("dt = tau/1000 (long run, 10001 steps)") {
        auto cb = make_rc(V_dc, R, C);
        const int probe_idx = static_cast<int>(cb.node_id_of("vc"));
        solver::SimulationOptions opts{
            .t_start = 0.0,
            .t_end   = 10.0 * tau,
            .dt      = tau / 1000.0,
        };
        auto rpt = time_and_validate(
            "RC tau/1000 (10τ run)", cb, opts, probe_idx, analytical);
        print_report_row(rpt);
        // dt/τ = 0.001 → half-step offset ≈ 2.5 mV. Tightened to 5 mV.
        REQUIRE(rpt.max_abs_error < 5e-3);
    }
}
