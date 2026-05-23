// =============================================================================
// Layer 9 — ZVS phase-shift full-bridge open-loop showcase
// =============================================================================
//
// END-TO-END validation of the V9 phase-shift full-bridge
// helper in a realistic isolated DC-DC topology:
//   - YAML loader reads `examples/v2/phase_shift_full_
//     bridge.yaml`: 100 V DC bus, 4-MOSFET primary full
//     bridge (+ body diodes), 1 µH leakage L, 2:1 trans-
//     former, 4-diode secondary bridge rectifier, 100 µH
//     + 100 µF LC filter, 10 Ω load.
//   - V9 `make_phase_shift_full_bridge_fn(...)` drives
//     the 4 controlled MOSFETs at 100 kHz with φ = π/2
//     (effective duty D_eff ≈ 0.5) and 100 ns dead-time.
//
// Expected steady-state at the load:
//   V_out ≈ V_bus · (Ns/Np) · D_eff
//         = 100 · 0.5 · 0.5  ≈  25 V (ideal lossless)
//
// Real loss budget: 2× bridge-rectifier forward drop +
// switch R_on losses + transformer leakage → expected
// V_out range [10, 30] V (loose to absorb non-idealities).
//
// Assertions:
//   * Steady-state V_out is positive and in [5, 35] V.
//   * V_out has low ripple after the LC filter (< 4 V p-p).
//   * Primary mid-points show full-rail switching.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/phase_shift_full_bridge_fn.hpp"
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
using namespace pulsim::sources;
using namespace pulsim::topology;
using Catch::Approx;

namespace {

std::string locate_psfb_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" / "v2" /
            "phase_shift_full_bridge.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/phase_shift_full_bridge.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("SMPS showcase: ZVS phase-shift full-bridge via YAML + V9",
          "[v2][layer9][showcase][smps][phase_shift_full_bridge]") {
    const std::string path = locate_psfb_yaml();
    INFO("phase_shift_full_bridge.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/phase_shift_full_bridge.yaml not "
             "located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    constexpr Real V_bus = 100.0;
    constexpr Real f_sw  = 100e3;
    constexpr Real dt_dead = 100e-9;
    // φ = π/2 → effective secondary duty ≈ 0.5.
    const Real phi = std::numbers::pi_v<Real> * Real{0.5};

    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 12);   // 4 MOSFETs + 4 body diodes + 4 bridge diodes

    // Insertion order: HS_A=0, LS_A=1, HS_B=2, LS_B=3.
    auto switch_fn = make_phase_shift_full_bridge_fn(
        f_sw, phi, 0, 1, 2, 3, n_sw, dt_dead);

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        loaded.options, switch_fn);
    REQUIRE(result.num_steps() > 1000);

    // Inspect v_out steady state over the LAST 0.2 ms.
    const Index vout_idx =
        loaded.builder.node_id_of("vout");
    REQUIRE(vout_idx >= 0);
    const Size k_start = result.num_steps() -
        static_cast<Size>(0.2e-3 / loaded.options.dt);

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

    INFO("PSFB steady-state V_out = " << v_mean
         << " V (ideal lossless = " << V_bus * 0.5 * 0.5
         << " V at φ=π/2 / Ns/Np=1/2)");
    INFO("PSFB V_out ripple = " << v_ripple << " V");

    // 1) Output is positive (rectifier works) and in a
    //    sensible range. Ideal lossless = 25 V; account
    //    for diode drops + leakage losses + simulation
    //    not fully settled at 2 ms.
    REQUIRE(v_mean > 5.0);
    REQUIRE(v_mean < 35.0);

    // 2) LC filter does its job — output ripple < 4 V p-p.
    REQUIRE(v_ripple < 4.0);

    // 3) Primary mid-points see full-rail switching activity
    //    (i.e. the bridge IS running, not stuck at one rail).
    const Index a_idx = loaded.builder.node_id_of("mid_a");
    const Index b_idx = loaded.builder.node_id_of("mid_b");
    REQUIRE(a_idx >= 0);
    REQUIRE(b_idx >= 0);
    Real a_min = 1e9, a_max = -1e9;
    Real b_min = 1e9, b_max = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        a_min = std::min(a_min, result.states[k][a_idx]);
        a_max = std::max(a_max, result.states[k][a_idx]);
        b_min = std::min(b_min, result.states[k][b_idx]);
        b_max = std::max(b_max, result.states[k][b_idx]);
    }
    INFO("mid_a swing: [" << a_min << ", " << a_max
         << "]  mid_b swing: [" << b_min << ", " << b_max
         << "]");
    REQUIRE(a_max - a_min > V_bus * 0.5);
    REQUIRE(b_max - b_min > V_bus * 0.5);
}
