// =============================================================================
// Saturable inductor — flux-state model with Steinmetz core loss.
//
// Build (header-only — links pulsim::core):
//
//   cmake -S examples/cpp -B build/examples
//   cmake --build build/examples --target pulsim_example_saturable_inductor
//   ./build/examples/pulsim_example_saturable_inductor
//
// Demonstrates:
//   - BHCurveTable from datasheet-style (H, B) measurements.
//   - SaturableInductor<BHCurveTable> stamping i(λ) and L_d(λ).
//   - Trapezoidal flux-state advance under sinusoidal voltage.
//   - Saturation detection — print B_pk so the user can compare against
//     the material's B_sat.
//
// See also: docs/magnetic-models.md
// =============================================================================
#include "pulsim/v1/magnetic/bh_curve.hpp"
#include "pulsim/v1/magnetic/saturable_inductor.hpp"

#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

using namespace pulsim::v1;
using magnetic::BHCurveTable;
using magnetic::SaturableInductor;

int main() {
    // Datasheet-style B-H table for a soft-magnetic core (M-3 grade, 4 mil
    // silicon-iron, μ_max ≈ 18000). Numbers are for illustration only.
    BHCurveTable bh(
        /*H*/ {-2000.0, -200.0, -50.0, -10.0, 0.0, 10.0, 50.0, 200.0, 2000.0},
        /*B*/ {-1.55,   -1.42,  -1.20, -0.90, 0.0, 0.90, 1.20, 1.42,  1.55});

    SaturableInductor<BHCurveTable>::Geometry geom{
        .turns       = 100.0,    // N turns
        .area        = 1.5e-4,   // A_e = 1.5 cm²
        .path_length = 7.0e-2,   // l_e = 7.0 cm (mean magnetic path)
    };
    SaturableInductor<BHCurveTable> L(geom, bh, "L_main");

    // Drive with v(t) = V_pk·sin(ω·t). For a sinusoidal flux path the
    // peak flux density is B_pk = V_pk / (ω·N·A). Two operating points:
    //   - V_pk = 4 V → B_pk ≈ 0.85 T (linear, no saturation)
    //   - V_pk = 8 V → B_pk ≈ 1.70 T (deep into saturation)
    constexpr double f    = 50.0;          // Hz
    const double omega    = 2.0 * std::numbers::pi * f;
    constexpr double T    = 1.0 / 50.0;
    constexpr int    N    = 2000;
    const double     dt   = T / N;

    auto run = [&](double V_pk, const char* label) {
        // Initialize λ at the peak-negative steady-state flux so the
        // sinusoidal drive doesn't accumulate a one-shot transient
        // offset on top of the AC swing. Steady-state: λ(t) = -(V_pk/ω)·cos(ω·t).
        double lambda    = -V_pk / omega;
        double v_prev    = 0.0;       // v(0) = V_pk·sin(0) = 0
        double B_pk      = 0.0;
        double L_min_seen = std::numeric_limits<double>::infinity();
        for (int k = 0; k <= N; ++k) {
            const double t      = k * dt;
            const double v_now  = V_pk * std::sin(omega * t);
            // Trapezoidal: λ_{n+1} = λ_n + (dt/2)·(v_n + v_{n+1})
            lambda += 0.5 * dt * (v_prev + v_now);
            v_prev  = v_now;

            const double B  = lambda / (geom.turns * geom.area);
            const double Ld = L.differential_inductance(lambda);
            B_pk        = std::max(B_pk, std::abs(B));
            L_min_seen  = std::min(L_min_seen, Ld);
        }
        const double L_lin = L.differential_inductance(0.0);
        std::cout << "[" << label << "]  V_pk = " << V_pk << " V\n";
        std::cout << "    predicted B_pk = V_pk/(ω·N·A) = "
                  << V_pk / (omega * geom.turns * geom.area) << " T\n";
        std::cout << "    measured  B_pk = " << B_pk << " T\n";
        std::cout << "    L_d at λ_pk    = " << L_min_seen * 1e3 << " mH    "
                  << "(linear-region L_d = " << L_lin * 1e3 << " mH"
                  << ", L_d/L_lin = " << L_min_seen / L_lin << ")\n";
        if (B_pk > 1.4) {
            std::cout << "    ⚠ saturated (B_pk > 1.4 T) — L_d collapses, current rises sharply\n";
        } else {
            std::cout << "    ✓ linear region — L_d ≈ L_lin, no saturation\n";
        }
        std::cout << "\n";
    };

    std::cout << "SaturableInductor diagnostics:\n"
              << "  geometry: N=" << geom.turns
              << "  A_e=" << geom.area << " m²"
              << "  l_e=" << geom.path_length << " m\n"
              << "  curve:    soft-magnetic core, B_sat ≈ 1.55 T\n\n";

    run(4.0, "linear   ");
    run(8.0, "saturated");
    return 0;
}
