// =============================================================================
// Layer 9 — 3-phase 6-pulse diode bridge rectifier showcase
// =============================================================================
//
// END-TO-END validation of V11 (SineVoltageSource) in the
// canonical 3-φ rectifier topology used by every industrial
// 3-phase AC-DC supply:
//   - YAML loader reads `examples/v2/three_phase_diode_
//     rectifier.yaml`: 3 sine sources (V11) at 100 V peak,
//     50 Hz, 120° apart + 6-diode full-wave bridge + 1 mF
//     filter capacitor + 100 Ω load.
//   - No `switch_fn` action — all 6 diodes are auto-
//     commutated by the v2 DiodeEventState.
//
// Expected steady-state DC bus voltage:
//   V_dc = (3·√6/π) · V_phase_peak ≈ 2.339 · 100 = 234 V
//   minus ~1.4 V of forward drops on the conducting pair
//   → real ≈ 232 V.
//
// Ripple frequency = 6 · f_line = 300 Hz (6-pulse rectifier).
// With C=1 mF / R=100 Ω → τ_RC = 100 ms, so 300-ms simulation
// reaches steady state.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/topology/switch_state.hpp"
#include "pulsim/yaml/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <numbers>
#include <string>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;
using Catch::Approx;

namespace {

std::string locate_three_phase_rectifier_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" / "v2" /
            "three_phase_diode_rectifier.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/three_phase_diode_rectifier.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("Showcase: 3-phase 6-pulse diode rectifier via YAML + V11",
          "[v2][layer9][showcase][rectifier][three_phase]") {
    const std::string path =
        locate_three_phase_rectifier_yaml();
    INFO("three_phase_diode_rectifier.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/three_phase_diode_rectifier.yaml "
             "not located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    // No controlled switches — all 6 diodes are auto-
    // commutated by the v2 DiodeEventState. Pass an empty
    // mask of size num_switches (= 6 diodes).
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 6);   // 6 bridge diodes

    auto switch_fn = [n_sw](Real) {
        return SwitchStateMask(n_sw);
    };

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        loaded.options, switch_fn);
    REQUIRE(result.num_steps() > 1000);

    // Sample DC bus over the LAST 20 ms (1 fundamental
    // cycle) — should be settled by 300 ms (3 τ_RC).
    const Index vdc_p_idx =
        loaded.builder.node_id_of("vdc_p");
    const Index vdc_n_idx =
        loaded.builder.node_id_of("vdc_n");
    REQUIRE(vdc_p_idx >= 0);
    REQUIRE(vdc_n_idx >= 0);

    constexpr Real T_fund = 1.0 / 50.0;   // 20 ms
    const Size k_start = result.num_steps() -
        static_cast<Size>(T_fund / loaded.options.dt);

    Real vdc_sum = 0;
    Real vdc_min = 1e9, vdc_max = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real v = result.states[k][vdc_p_idx] -
                       result.states[k][vdc_n_idx];
        vdc_sum += v;
        vdc_min = std::min(vdc_min, v);
        vdc_max = std::max(vdc_max, v);
    }
    const Real n = static_cast<Real>(
        result.num_steps() - k_start);
    const Real vdc_mean = vdc_sum / n;
    const Real vdc_ripple = vdc_max - vdc_min;

    // Analytical reference for a 6-pulse cap-input bridge
    // rectifier with Y-connected sources: in steady state
    // (large C), V_dc clamps near the LINE-TO-LINE peak,
    // which equals √3 · V_phase_peak. The average is
    // slightly below the LL peak due to the small dip
    // between conduction pulses (~few %% with 1 mF / 100 Ω):
    //   V_dc_ideal ≈ √3 · V_phase_peak ≈ 1.732 · 100 ≈ 173 V
    constexpr Real V_phase_peak = 100.0;
    const Real vdc_ideal =
        std::sqrt(Real{3}) * V_phase_peak;

    INFO("V_dc mean = " << vdc_mean
         << " V  (ideal = " << vdc_ideal
         << " V, ripple = " << vdc_ripple << " V)");

    // 1) V_dc within 10 % of analytical (covers diode drops
    //    + finite ripple + R_ref loading).
    REQUIRE(vdc_mean > vdc_ideal * 0.9);
    REQUIRE(vdc_mean < vdc_ideal * 1.05);

    // 2) Ripple bounded — 1 mF / 100 Ω with 6-pulse ripple
    //    yields ~8 V p-p; allow up to 20 V.
    REQUIRE(vdc_ripple < 20.0);

    // 3) Phase sources still oscillate at their nominal
    //    amplitude (sanity-check the AC overlay is being
    //    applied each step). Sample phase A over the last
    //    cycle and verify peak ≈ ±100 V.
    const Index a_idx = loaded.builder.node_id_of("a");
    REQUIRE(a_idx >= 0);
    Real a_min = 1e9, a_max = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        a_min = std::min(a_min, result.states[k][a_idx]);
        a_max = std::max(a_max, result.states[k][a_idx]);
    }
    INFO("phase A swing: [" << a_min << ", " << a_max
         << "] V (target ±100 V)");
    REQUIRE(a_max == Approx(V_phase_peak).margin(5.0));
    REQUIRE(a_min == Approx(-V_phase_peak).margin(5.0));
}
