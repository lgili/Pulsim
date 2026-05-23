// =============================================================================
// Benchmark — series RLC step response vs damped-oscillator analytical
// =============================================================================
//
// Topology:  Vstep ──[L]── nA ──[R]── nB ──[C]── gnd
//
// Driven by a delayed step (Vstep idles at 0 until t_start, then jumps
// to V_step). The capacitor voltage v_C(t) follows the canonical
// second-order step response:
//
//   ω_n  = 1/√(LC)
//   ζ    = (R/2) · √(C/L)
//   ω_d  = ω_n · √(1 − ζ²)
//   v_C(t > t_start) = V_step · (1 − e^{−ζ·ω_n·t'}
//                       · (cos(ω_d·t') + (ζ/√(1−ζ²)) · sin(ω_d·t')))
//   where t' = t − t_start.
//
// Values: L = 100 µH, C = 100 µF → ω_n ≈ 10 000 rad/s, T ≈ 628 µs.
// R = 0.1 Ω → ζ = 0.05 (lightly damped, ~95 % first overshoot).

#include <catch2/catch_test_macros.hpp>

#include "bench_helpers.hpp"

#include <cmath>

using namespace pulsim;
using namespace pulsim::bench;

namespace {

builder::CircuitBuilder make_rlc(Real V_step, Real R, Real L, Real C,
                                  Real t_start) {
    builder::CircuitBuilder b;
    b.add_pulse_voltage_source(
        "Vstep", "in", "gnd",
        /*v_initial=*/0.0, /*v_pulsed=*/V_step,
        /*t_start=*/t_start, /*pulse_width=*/10.0);
    b.add_inductor ("L",  "in", "nA",  L);
    b.add_resistor ("R",  "nA", "nB",  R);
    b.add_capacitor("C",  "nB", "gnd", C);
    return b;
}

}  // namespace

TEST_CASE("Bench: RLC underdamped step — analytical agreement",
          "[bench][rlc][analytical]") {
    print_report_header();

    const Real V_step = 10.0;
    const Real R      = 0.1;
    const Real L      = 100e-6;
    const Real C      = 100e-6;
    const Real t0     = 100e-6;     // pre-pulse baseline
    const Real omega_n = 1.0 / std::sqrt(L * C);
    const Real zeta    = (R / 2.0) * std::sqrt(C / L);
    const Real omega_d = omega_n * std::sqrt(1.0 - zeta * zeta);

    auto analytical = [V_step, t0, omega_n, omega_d, zeta]
                      (double t) {
        if (t < t0) return 0.0;
        const double tp = t - t0;
        const double decay = std::exp(-zeta * omega_n * tp);
        const double phase = std::cos(omega_d * tp)
            + (zeta / std::sqrt(1.0 - zeta * zeta))
                * std::sin(omega_d * tp);
        return V_step * (1.0 - decay * phase);
    };

    SECTION("dt = T/200 (~0.5 % of oscillation period)") {
        auto cb = make_rlc(V_step, R, L, C, t0);
        const int probe = static_cast<int>(cb.node_id_of("nB"));
        const Real T_osc = 2.0 * M_PI / omega_d;
        solver::SimulationOptions opts{
            .t_start = 0.0,
            .t_end   = t0 + 8.0 * T_osc,   // see full ringdown
            .dt      = T_osc / 200.0,
        };
        auto rpt = time_and_validate(
            "RLC underdamped (5 cycles)", cb, opts, probe, analytical);
        print_report_row(rpt);
        // Lightly damped (ζ=0.05) → peak overshoot ≈ 8.5 V → ~18.5 V
        // peak v_C. 200 samples/cycle is plenty; trap rule error
        // dominated by O(dt²) ≈ 0.025² ≈ 6e-4 of peak ≈ 0.012 V.
        // Allow 0.1 V (≈ 0.5 % of peak) — keeps headroom for the
        // resonant peak which is the hardest spot.
        REQUIRE(rpt.max_abs_error < 0.1);
    }
}
