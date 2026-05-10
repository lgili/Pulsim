// =============================================================================
// Three-phase synchronous-reference-frame PLL — lock to a 50 Hz grid.
//
// Build:
//   cmake -S examples/cpp -B build/examples
//   cmake --build build/examples --target pulsim_example_three_phase_pll
//   ./build/examples/pulsim_example_three_phase_pll
//
// Demonstrates:
//   - ThreePhaseSource generating a balanced 3φ grid voltage.
//   - SrfPll with V_pk-normalized gains (independent of grid level):
//       Kp = 2·ζ·ω_pll / V_pk
//       Ki = ω_pll²    / V_pk
//   - Lock acquisition timing — gate G.1 of `add-three-phase-grid-library`
//     contracts the SRF-PLL to lock within 50 ms on a nominal grid.
//
// See also: docs/three-phase-grid.md
// =============================================================================
#include "pulsim/v1/grid/pll.hpp"
#include "pulsim/v1/grid/three_phase_source.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>

using namespace pulsim::v1;
using grid::SrfPll;
using grid::ThreePhaseSource;

int main() {
    constexpr double f_grid = 50.0;
    constexpr double V_rms  = 230.0;
    const double V_pk = V_rms * std::numbers::sqrt2_v<double>;

    // ω_pll = 2π·f_bw with critical damping ζ=1/√2.
    constexpr double f_bw = 30.0;     // 30 Hz PLL bandwidth
    const double omega_pll = 2.0 * std::numbers::pi * f_bw;
    SrfPll::Params params;
    params.kp        = 2.0 * (1.0 / std::numbers::sqrt2_v<double>) * omega_pll / V_pk;
    params.ki        = (omega_pll * omega_pll) / V_pk;
    params.freq_init = f_grid;
    params.omega_min = 2.0 * std::numbers::pi * 10.0;
    params.omega_max = 2.0 * std::numbers::pi * 200.0;
    SrfPll pll(params);

    ThreePhaseSource src{.v_rms = V_rms, .frequency = f_grid};

    constexpr double dt = 1e-4;       // 10 kHz loop
    constexpr int    N  = 500;        // 50 ms (the gate G.1 horizon)
    constexpr int    settle_min_idx = 50;     // ignore the first ms for lock detection
    std::cout << "SRF-PLL targeting " << f_grid << " Hz / " << V_rms << " V_rms\n";
    std::cout << "  bandwidth = " << f_bw << " Hz   "
              << "Kp = " << params.kp << "    Ki = " << params.ki << "\n";
    std::cout << "  expected lock time ≈ 50 ms (gate G.1)\n\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "    t (ms)  |  ω̂ (rad/s)  |  f̂ (Hz)  |  Δθ (deg)\n";
    std::cout << "------------+--------------+----------+----------\n";

    double last_freq = 0.0;
    int    lock_idx  = -1;
    for (int k = 0; k <= N; ++k) {
        const double t = k * dt;
        const auto [a, b, c] = src.evaluate(t);
        const auto [theta_hat, omega_hat] = pll.step(a, b, c, dt);
        const double freq_hat = omega_hat / (2.0 * std::numbers::pi);

        // SrfPll::step() returns the angle AFTER advancing by ω̂·dt, so it
        // corresponds to time `t + dt`. Compare on that grid for a fair
        // phase-error reading.
        const double theta_grid = std::fmod(
            2.0 * std::numbers::pi * f_grid * (t + dt),
            2.0 * std::numbers::pi);
        double dtheta = theta_hat - theta_grid;
        while (dtheta >  std::numbers::pi) dtheta -= 2.0 * std::numbers::pi;
        while (dtheta < -std::numbers::pi) dtheta += 2.0 * std::numbers::pi;

        // Require some settle time before declaring lock — the PLL is
        // initialized at the nominal frequency, so the first samples
        // trivially satisfy the tolerance.
        if (lock_idx < 0 && k >= settle_min_idx
                         && std::abs(freq_hat - f_grid) < 0.05
                         && std::abs(dtheta) < 0.02) {
            lock_idx = k;
        }
        if (k % 100 == 0) {
            std::cout << std::setw(10) << t * 1e3 << "  | "
                      << std::setw(12) << omega_hat << " | "
                      << std::setw(8)  << freq_hat << " | "
                      << std::setw(8)  << dtheta * 180.0 / std::numbers::pi << "\n";
        }
        last_freq = freq_hat;
    }

    std::cout << "\nFinal frequency estimate: " << last_freq << " Hz "
              << "(target " << f_grid << " Hz)\n";
    if (lock_idx >= 0) {
        std::cout << "✓ PLL locked at t = " << lock_idx * dt * 1e3 << " ms"
                  << " (gate ≤ 50 ms)\n";
    } else {
        std::cout << "✗ PLL did NOT lock within " << N * dt * 1e3 << " ms\n";
    }
    return 0;
}
