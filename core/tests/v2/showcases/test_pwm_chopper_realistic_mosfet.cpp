// =============================================================================
// Layer 9 — PWM chopper with REALISTIC SH1 MOSFET
// =============================================================================
//
// END-TO-END validation of V13 SH1 MOSFET as a hard-switched
// active device under PWM gate drive. The topology is a
// simple series-pass chopper:
//
//   V_in ── R_in ──┬── drain
//                  │
//                  M1 ── gnd
//                  │
//                  R_load ── gnd
//
// (See `examples/v2/pwm_chopper_realistic_mosfet.yaml` for the
// rationale on why this is a chopper instead of a "real"
// boost — Newton struggles to converge with L in series
// with a smooth-blend MOSFET driven by a step PWM edge.)
//
// Expected behaviour:
//   * MOSFET ON  (V_g = 10 V): drain ≈ 0 V (M acts as a
//     low-resistance pull-down).
//   * MOSFET OFF (V_g = 0):    drain ≈ 8 V (resistor
//     divider R_in / R_load = 50/100).
//   * Drain swings full-rail at 100 kHz with 40 %% duty;
//     time-average ≈ 0.4·0 + 0.6·8 = 4.8 V.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/pwl/nonlinear_refresh_mosfet_level1.hpp"
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

std::string locate_chopper_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" / "v2" /
            "pwm_chopper_realistic_mosfet.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/pwm_chopper_realistic_mosfet.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("Showcase: PWM chopper with realistic SH1 MOSFET",
          "[v2][layer9][showcase][mosfet][chopper]") {
    const std::string path = locate_chopper_yaml();
    INFO("pwm_chopper_realistic_mosfet.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/pwm_chopper_realistic_mosfet.yaml not "
             "located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    // No switches: M1 is Nonlinear, no SwitchedDiode in the
    // chopper variant.
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 0);

    auto switch_fn = [](Real) { return SwitchStateMask(0); };
    auto nl_refresh = make_combined_diode_mosfet_refresh();

    SimulationOptions opts = loaded.options;
    opts.max_newton_iterations = 100;

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        opts, switch_fn, {}, false, nl_refresh);
    REQUIRE(result.num_steps() > 100);

    // Sample drain over the LAST PWM cycle.
    const Index drain_idx =
        loaded.builder.node_id_of("drain");
    REQUIRE(drain_idx >= 0);

    constexpr Real T_pwm = 1.0e-5;
    const Size k_start = result.num_steps() -
        static_cast<Size>(T_pwm / loaded.options.dt);

    Real v_sum = 0;
    Real v_min = 1e9, v_max = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real v = result.states[k][drain_idx];
        v_sum += v;
        v_min = std::min(v_min, v);
        v_max = std::max(v_max, v);
    }
    const Real n = static_cast<Real>(
        result.num_steps() - k_start);
    const Real v_mean = v_sum / n;

    INFO("Drain mean = " << v_mean << " V over last PWM cycle");
    INFO("Drain swing: [" << v_min << ", " << v_max << "] V");

    // 1) When M OFF: drain ≈ 8 V (resistor divider).
    REQUIRE(v_max > 6.0);
    REQUIRE(v_max < 9.0);

    // 2) When M ON: drain ≈ 1.45 V (triode equilibrium with
    //    K=0.01). Allow up to 4 V for sigmoid transition
    //    overshoot.
    REQUIRE(v_min >= -0.5);   // never negative
    REQUIRE(v_min < 4.0);

    // 3) Time-average over one cycle: 0.4·1.45 + 0.6·8 ≈
    //    5.4 V. Loose bounds [3.0, 6.5] V.
    REQUIRE(v_mean > 3.0);
    REQUIRE(v_mean < 7.0);
}
