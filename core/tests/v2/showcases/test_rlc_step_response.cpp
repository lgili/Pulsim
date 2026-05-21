// =============================================================================
// Layer 9 — RLC step-response open-loop showcase
// =============================================================================
//
// END-TO-END validation of V12 PulseVoltageSource in a 2nd-
// order linear system:
//   - YAML loader reads `examples/v2/rlc_step_response.yaml`:
//     V12 pulse source (10 V step delayed by 100 µs) drives
//     a series RLC: L=100 µH, R=0.1 Ω, C=100 µF →
//     ω_n=10 000 rad/s, ζ=0.05 (underdamped, lots of ring).
//   - No `switch_fn` action — pure linear RLC.
//
// Expected step response of V_C:
//   V_C(t) = V·(1 − e^(−ζω_n·t̄) · (cos(ω_d·t̄) +
//                                    ζ/√(1−ζ²)·sin(ω_d·t̄)))
//   where t̄ = t − t_start.
//
// Assertions:
//   * Pre-pulse: V_C ≈ 0 until t_start.
//   * Post-pulse peak (overshoot): V_C peaks WELL above
//     V_step due to ζ ≈ 0.05 → expected ~18.5 V around
//     t = t_start + π/ω_d ≈ t_start + 314 µs.
//   * After several oscillations the average converges to
//     V_step (10 V).
//   * Oscillation period matches T_osc = 2π/ω_d ≈ 629 µs
//     measured from peak-to-peak.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/solver/run_transient.hpp"
#include "pulsim/v2/topology/switch_state.hpp"
#include "pulsim/v2/yaml/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numbers>
#include <string>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::topology;
using Catch::Approx;

namespace {

std::string locate_rlc_step_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" / "v2" /
            "rlc_step_response.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/rlc_step_response.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("Showcase: RLC step response via YAML + V12 PulseVoltageSource",
          "[v2][layer9][showcase][rlc_step_response]") {
    const std::string path = locate_rlc_step_yaml();
    INFO("rlc_step_response.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/rlc_step_response.yaml not "
             "located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    // No switches anywhere — pure linear RLC.
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 0);
    auto switch_fn = [](Real) {
        return SwitchStateMask(0);
    };

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        loaded.options, switch_fn);
    REQUIRE(result.num_steps() > 1000);

    const Index vc_idx = loaded.builder.node_id_of("vc");
    REQUIRE(vc_idx >= 0);

    constexpr Real V_step  = 10.0;
    constexpr Real t_start = 1.0e-4;     // 100 µs delay
    constexpr Real L = 1e-4;
    constexpr Real R = 0.1;
    constexpr Real C = 1e-4;
    const Real omega_n = std::sqrt(Real{1} / (L * C));
    const Real zeta    = Real{0.5} * R *
                          std::sqrt(C / L);   // ≈ 0.05
    const Real omega_d = omega_n *
                         std::sqrt(Real{1} - zeta * zeta);
    const Real T_osc   = Real{2} *
                         std::numbers::pi_v<Real> / omega_d;

    INFO("ω_n=" << omega_n << " ζ=" << zeta
         << " ω_d=" << omega_d
         << " T_osc=" << T_osc << " s");

    // 1) Pre-pulse: V_C ≈ 0 until t_start.
    {
        const Size k_pre = static_cast<Size>(
            (t_start - 2e-5) / loaded.options.dt);
        REQUIRE(std::abs(result.states[k_pre][vc_idx])
                < 0.05);
    }

    // 2) Peak overshoot (occurs around t_start + π/ω_d).
    const Real t_peak_expected = t_start +
        std::numbers::pi_v<Real> / omega_d;
    const Real peak_expected = V_step * (Real{1} +
        std::exp(-std::numbers::pi_v<Real> * zeta /
                  std::sqrt(Real{1} - zeta * zeta)));
    INFO("Expected peak ≈ " << peak_expected << " V at t ≈ "
         << t_peak_expected << " s");

    Real v_peak = -1e9;
    Real t_peak_measured = 0;
    for (Size k = 0; k < result.num_steps(); ++k) {
        if (result.states[k][vc_idx] > v_peak) {
            v_peak = result.states[k][vc_idx];
            t_peak_measured = result.times[k];
        }
    }
    INFO("Measured peak = " << v_peak << " V at t = "
         << t_peak_measured << " s");
    // Underdamped → peak must significantly exceed V_step.
    REQUIRE(v_peak > V_step * 1.5);     // > 15 V
    REQUIRE(v_peak < V_step * 2.0);     // < 20 V (loose)

    // Peak time within 100 µs of analytical (allow for
    // finite-dt sampling).
    REQUIRE(std::abs(t_peak_measured - t_peak_expected)
            < 1e-4);

    // 3) Late-time average → V_step. Sample over the last
    //    1 ms (settling time τ_settle = 4/(ζω_n) ≈ 8 ms is
    //    actually beyond our 3 ms sim, but the AVERAGE
    //    converges quickly because positive and negative
    //    oscillations symmetric).
    const Size k_end = result.num_steps();
    const Size k_lateavg = result.num_steps() -
        static_cast<Size>(1e-3 / loaded.options.dt);
    Real v_sum = 0;
    for (Size k = k_lateavg; k < k_end; ++k) {
        v_sum += result.states[k][vc_idx];
    }
    const Real v_mean = v_sum /
        static_cast<Real>(k_end - k_lateavg);
    INFO("Late-time mean V_C = " << v_mean
         << " V (expected " << V_step << ")");
    REQUIRE(std::abs(v_mean - V_step) < 1.0);

    // 4) Find the 2nd peak and verify oscillation period.
    Real v_2nd_peak = -1e9;
    Real t_2nd_peak = 0;
    for (Size k = 0; k < result.num_steps(); ++k) {
        // Skip the first peak window.
        if (result.times[k] < t_peak_measured + 0.3 * T_osc) {
            continue;
        }
        // Stop searching after one period.
        if (result.times[k] > t_peak_measured + 1.5 * T_osc) {
            break;
        }
        if (result.states[k][vc_idx] > v_2nd_peak) {
            v_2nd_peak = result.states[k][vc_idx];
            t_2nd_peak = result.times[k];
        }
    }
    const Real T_measured = t_2nd_peak - t_peak_measured;
    INFO("Measured T_osc = " << T_measured << " s "
         "(expected " << T_osc << " s)");
    REQUIRE(T_measured == Approx(T_osc).margin(2e-5));
}
