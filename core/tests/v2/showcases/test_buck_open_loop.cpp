// =============================================================================
// Layer 9 — Buck converter open-loop showcase
// =============================================================================
//
// END-TO-END validation of v2:
//   - Layer 8 YAML loader reads `examples/v2/buck.yaml`.
//   - Layer 6 CircuitBuilder + Layer 4 PWL cache build
//     the segment for each switch state.
//   - Layer 5 run_transient drives a 100 kHz / 50 % PWM
//     switch_fn that toggles the high-side MOSFET.
//   - Free-wheeling diode (and MOSFET body diode) auto-
//     commutate via Layer 5 V2's DiodeEventState.
//
// Verifies steady-state V_out = V_in · D · η = 12 V ± 0.5
// and ripple < 1 V p-p.

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
#include <string>

using namespace pulsim::v2;
using namespace pulsim::v2::pwl;
using namespace pulsim::v2::solver;
using namespace pulsim::v2::topology;
using Catch::Approx;

namespace {

/// Locate the repo's `examples/v2/buck.yaml` by walking up
/// from the running binary until we find an `examples/v2`
/// directory. This avoids hard-coded paths in tests.
std::string locate_buck_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate =
            search / "examples" / "v2" / "buck.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    // CMAKE_SOURCE_DIR-based fallback via env var, set by
    // the build system if needed.
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) + "/buck.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("SMPS showcase: open-loop buck via YAML reaches V_in · D",
          "[v2][layer9][showcase][smps]") {
    const std::string path = locate_buck_yaml();
    INFO("buck.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/buck.yaml not located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    // Build the cache at the YAML-specified dt.
    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    // 100 kHz PWM, 50 % duty cycle, drives Q1's switch bit
    // (insertion order: Q1 is the first Switch branch, so
    // switch_idx = 0).
    constexpr Real f_sw = 100e3;
    constexpr Real T_sw = 1.0 / f_sw;
    constexpr Real duty = 0.5;

    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw > 0);
    auto switch_fn = [n_sw](Real t) {
        const Real phase = std::fmod(t, T_sw) / T_sw;
        SwitchStateMask mask(n_sw);
        if (phase < duty) {
            mask.set(0, true);   // Q1 ON
        }
        return mask;
    };

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        loaded.options, switch_fn);
    REQUIRE(result.num_steps() > 100);

    // Find vout node.
    const Index vout_idx =
        loaded.builder.node_id_of("vout");
    REQUIRE(vout_idx >= 0);

    // Measure steady-state V_out over the LAST 0.5 ms.
    const Real measurement_window = 0.5e-3;
    const Size k_start = result.num_steps() -
        static_cast<Size>(measurement_window /
                          loaded.options.dt);

    Real v_sum = Real{0};
    Real v_max = -1e9;
    Real v_min =  1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real v = result.states[k][vout_idx];
        v_sum += v;
        v_max  = std::max(v_max, v);
        v_min  = std::min(v_min, v);
    }
    const Real n_samples =
        static_cast<Real>(result.num_steps() - k_start);
    const Real v_mean = v_sum / n_samples;
    const Real v_ripple = v_max - v_min;
    const Real v_target = 24.0 * duty;   // V_in · D

    INFO("Buck steady-state V_out mean = " << v_mean
         << " V  (target = " << v_target << " V)");
    INFO("Buck V_out ripple p-p = " << v_ripple << " V");

    // Mean V_out within 0.5 V of analytical target (small
    // R_on + IR losses are tolerated).
    REQUIRE(v_mean == Approx(v_target).margin(0.5));

    // Ripple bounded by 1 V p-p (LC filter performance).
    REQUIRE(v_ripple < 1.0);
}

TEST_CASE("Buck showcase: alternative duty cycle ≈ 25 % → V_out ≈ 6 V",
          "[v2][layer9][showcase][smps]") {
    const std::string path = locate_buck_yaml();
    if (path.empty()) {
        WARN("examples/v2/buck.yaml not located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    constexpr Real f_sw = 100e3;
    constexpr Real T_sw = 1.0 / f_sw;
    constexpr Real duty = 0.25;
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw > 0);
    auto switch_fn = [n_sw](Real t) {
        const Real phase = std::fmod(t, T_sw) / T_sw;
        SwitchStateMask mask(n_sw);
        if (phase < duty) {
            mask.set(0, true);
        }
        return mask;
    };

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        loaded.options, switch_fn);

    const Index vout_idx = loaded.builder.node_id_of("vout");
    const Size k_start = result.num_steps() -
        static_cast<Size>(0.5e-3 / loaded.options.dt);
    Real v_sum = 0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        v_sum += result.states[k][vout_idx];
    }
    const Real v_mean = v_sum /
        static_cast<Real>(result.num_steps() - k_start);
    const Real v_target = 24.0 * duty;   // 6 V

    INFO("Buck D=25%: V_out mean = " << v_mean
         << " V (target " << v_target << " V)");
    REQUIRE(v_mean == Approx(v_target).margin(0.5));
}
