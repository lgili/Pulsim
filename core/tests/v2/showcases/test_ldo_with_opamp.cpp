// =============================================================================
// Layer 9 — LDO with op-amp feedback showcase
// =============================================================================
//
// First closed-loop ANALOG CONTROL CIRCUIT in v2. Combines:
//   * V13 SH1 MOSFET as the pass element (nonlinear).
//   * V15 ideal op-amp (VCVS with high gain) closing the
//     feedback loop.
//   * Resistor divider sampling V_out.
//
// Loop equation: V_out · R2/(R1+R2) = V_ref
//                V_out = V_ref · (R1+R2)/R2 = 2.5 · 15/5 = 7.5 V
//
// Expected: regardless of load current (I_load = V_out/R_load),
// the closed loop holds V_out at 7.5 V (within 1/gain · V_out
// errors). This is the canonical LDO behavior.

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

std::string locate_ldo_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" / "v2" /
            "ldo_with_opamp.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/ldo_with_opamp.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("Showcase: LDO with op-amp feedback (V13 + V15)",
          "[v2][layer9][showcase][ldo][opamp]") {
    const std::string path = locate_ldo_yaml();
    INFO("ldo_with_opamp.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/ldo_with_opamp.yaml not located — "
             "skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    // M1 is Nonlinear (Newton refresh stamped). Op-amp is a
    // Source-kind branch (no Newton). No SwitchedDiodes.
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 0);

    auto switch_fn = [](Real) {
        return SwitchStateMask(0);
    };
    auto nl_refresh = make_combined_diode_mosfet_refresh();

    SimulationOptions opts = loaded.options;
    opts.max_newton_iterations = 200;
    opts.enable_newton_line_search = true;

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        opts, switch_fn, {}, false, nl_refresh);
    REQUIRE(result.num_steps() > 50);

    // Steady-state V_out over last 200 µs.
    const Index vout_idx =
        loaded.builder.node_id_of("vout");
    REQUIRE(vout_idx >= 0);
    const Size k_start = result.num_steps() -
        static_cast<Size>(200e-6 / loaded.options.dt);

    Real v_sum = 0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        v_sum += result.states[k][vout_idx];
    }
    const Real v_out =
        v_sum / static_cast<Real>(
                    result.num_steps() - k_start);

    INFO("LDO V_out = " << v_out
         << " V (target 7.5 V from feedback divider)");

    // The closed loop should regulate V_out to V_ref·(R1+R2)/R2 = 7.5 V.
    // Real LDOs deviate by 1/loop_gain plus the MOSFET-V_GS dependence;
    // allow ±0.5 V.
    REQUIRE(v_out == Approx(7.5).margin(0.5));

    // Sanity check: V_in (12) > V_out (7.5) — the LDO is dropping.
    const Real v_in = 12.0;
    REQUIRE(v_out < v_in);
    REQUIRE(v_out > 0.5 * v_in);
}
