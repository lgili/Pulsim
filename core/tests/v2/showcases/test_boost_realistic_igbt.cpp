// =============================================================================
// Layer 9 — Boost converter with REALISTIC IGBT
// =============================================================================
//
// First real SMPS topology using a nonlinear active device
// in v2. The combination that finally converged:
//   * IGBT Level 1 (V14) — linear-conduction physics has
//     NO spurious roots in its I-V (unlike SH1 MOSFET).
//   * PulseVoltageSource with rise_time / fall_time (V14)
//     — ramps the gate over multiple timesteps, avoiding
//     the singularity at the gate-edge.
//   * Smooth-blend nonlinear_diode for the freewheel diode
//     — solves together with the IGBT in the same Newton
//     refresh, eliminating event-mask coupling.
//
// `examples/v2/boost_realistic_igbt.yaml` builds:
//   V_in=12V → L=100µH → sw → D_boost → vout → C=100µF, R=20Ω
//                            ↳ T1 → gnd (gate=V_g pulse)
//
// Expected V_out ≈ V_in / (1 − D) − losses.
// With duty_eff ≈ 0.3 (3µs pulse_width + 1µs rise + 1µs fall
// over 10µs period, but only the flat top really conducts
// hard), V_out ≈ V_in / (1 − 0.3) ≈ 17 V minus 1 V V_CE_sat
// and 0.5 V diode drop and conduction losses ≈ 12-18 V.

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

std::string locate_boost_igbt_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" / "v2" /
            "boost_realistic_igbt.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/boost_realistic_igbt.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("Showcase: real boost converter with IGBT + ramped gate",
          "[v2][layer9][showcase][smps][boost_igbt]") {
    const std::string path = locate_boost_igbt_yaml();
    INFO("boost_realistic_igbt.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/boost_realistic_igbt.yaml not "
             "located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    // T1 = IGBT (Nonlinear). D_boost = SwitchedDiode
    // (Switch-kind). So 1 switch-kind branch.
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 1);

    auto switch_fn_fix = [n_sw](Real) {
        return SwitchStateMask(n_sw);
    };

    auto nl_refresh = make_combined_diode_mosfet_refresh();

    SimulationOptions opts = loaded.options;
    opts.max_newton_iterations = 200;
    opts.tol_newton_dx  = 1e-5;
    opts.tol_newton_res = 1e-5;
    opts.enable_newton_line_search = true;

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        opts, switch_fn_fix, {}, false, nl_refresh);
    REQUIRE(result.num_steps() > 1000);

    // Sample V_out steady-state over last 500 µs.
    const Index vout_idx =
        loaded.builder.node_id_of("vout");
    REQUIRE(vout_idx >= 0);
    const Size k_start = result.num_steps() -
        static_cast<Size>(500e-6 / loaded.options.dt);

    Real v_sum = 0;
    Real v_min = 1e9, v_max = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real v = result.states[k][vout_idx];
        v_sum += v;
        v_min = std::min(v_min, v);
        v_max = std::max(v_max, v);
    }
    const Real n = static_cast<Real>(
        result.num_steps() - k_start);
    const Real v_mean = v_sum / n;
    const Real v_ripple = v_max - v_min;

    INFO("Boost V_out mean = " << v_mean
         << " V (V_in = 12 V; expected boosted to 12-20 V)");
    INFO("Boost V_out ripple = " << v_ripple << " V");

    // 1) V_out is BOOSTED above V_in.
    REQUIRE(v_mean > 12.5);

    // 2) V_out is below the ideal V_in/(1-D) (losses present).
    REQUIRE(v_mean < 25.0);

    // 3) Ripple bounded by LC filter.
    REQUIRE(v_ripple < 5.0);

    // 4) IGBT collector (sw) swings:
    //    M ON: sw near V_CE_sat (~1 V).
    //    M OFF: sw at v_out + V_F (diode forward).
    const Index sw_idx = loaded.builder.node_id_of("sw");
    REQUIRE(sw_idx >= 0);
    Real sw_min = 1e9, sw_max = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        sw_min = std::min(sw_min, result.states[k][sw_idx]);
        sw_max = std::max(sw_max, result.states[k][sw_idx]);
    }
    INFO("sw swing: [" << sw_min << ", " << sw_max << "] V");
    REQUIRE(sw_min < 3.0);              // IGBT on → V_CE small
    REQUIRE(sw_max > v_mean * 0.8);     // IGBT off → V_CE = v_out + V_F
}
