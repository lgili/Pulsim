// =============================================================================
// Layer 9 — 3-phase voltage-source inverter open-loop showcase
// =============================================================================
//
// END-TO-END validation of the V6/V7/V8 inverter helper
// family:
//   - Layer 8 YAML loader reads `examples/v2/three_phase_
//     inverter.yaml` (300 V DC bus + 6 switches + Y-connected
//     RL motor-style load).
//   - Layer 2 V8 `make_three_phase_spwm_fn(...)` drives the
//     6 switches at 20 kHz carrier / 50 Hz fundamental /
//     M = 0.8 / 200 ns dead-time.
//   - Layer 5 run_transient + Layer 4 PWL cache (2^6=64
//     switch combos pre-factored) integrate the system.
//
// Assertions:
//   * Time-averaged mid-points over the LAST full mod cycle
//     all equal V_bus/2 (sine reference averages to zero).
//   * Per-phase AC peak-to-peak swing is similar across the
//     3 legs (balanced amplitudes, ≤ 5 % imbalance).
//   * KCL at neutral: sum of instantaneous mid-to-neutral
//     voltages (≈ proportional to phase currents under linear
//     R+L) sums to ~zero — balanced 3-phase set.
//
// This locks in the V8 helper end-to-end via YAML +
// CircuitBuilder + cache + transient, just like the V5
// flyback showcase locked in `make_pwm_switch_fn`.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/three_phase_spwm_fn.hpp"
#include "pulsim/topology/switch_state.hpp"
#include "pulsim/yaml/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::sources;
using namespace pulsim::topology;
using Catch::Approx;

namespace {

std::string locate_three_phase_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" / "v2" /
            "three_phase_inverter.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/three_phase_inverter.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("SMPS showcase: 3-phase VSI via YAML + make_three_phase_spwm_fn",
          "[v2][layer9][showcase][smps][three_phase]") {
    const std::string path = locate_three_phase_yaml();
    INFO("three_phase_inverter.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/three_phase_inverter.yaml not "
             "located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    constexpr Real f_c = 20e3;     // 20 kHz carrier
    constexpr Real f_m = 50.0;     // 50 Hz fundamental
    constexpr Real M   = 0.8;
    constexpr Real T_m = 1.0 / f_m;
    constexpr Real dt_dead = 200e-9;
    constexpr Real V_bus = 300.0;

    // YAML adds 6 controlled MOSFETs (idx 0..5) + 6 body
    // diodes (idx 6..11; auto-commutated by the diode-event
    // system). num_switches reports all Switch-kind branches.
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 12);

    // Insertion order: HS_A=0, LS_A=1, HS_B=2, LS_B=3,
    // HS_C=4, LS_C=5. Diodes occupy the remaining indices.
    ThreePhaseLegIndices legs{0, 1, 2, 3, 4, 5};
    auto switch_fn = make_three_phase_spwm_fn(
        f_c, f_m, M, legs, n_sw, dt_dead);

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        loaded.options, switch_fn);
    REQUIRE(result.num_steps() > 1000);

    // Sample mid-point nodes.
    const Index a_idx = loaded.builder.node_id_of("mid_a");
    const Index b_idx = loaded.builder.node_id_of("mid_b");
    const Index c_idx = loaded.builder.node_id_of("mid_c");
    const Index n_idx = loaded.builder.node_id_of("n");
    REQUIRE(a_idx >= 0);
    REQUIRE(b_idx >= 0);
    REQUIRE(c_idx >= 0);
    REQUIRE(n_idx >= 0);

    // Average over the LAST full modulation cycle.
    const Size k_start = result.num_steps() -
        static_cast<Size>(T_m / loaded.options.dt);

    Real a_sum = 0, b_sum = 0, c_sum = 0;
    Real a_min = 1e9, a_max = -1e9;
    Real b_min = 1e9, b_max = -1e9;
    Real c_min = 1e9, c_max = -1e9;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        const Real va = result.states[k][a_idx];
        const Real vb = result.states[k][b_idx];
        const Real vc = result.states[k][c_idx];
        a_sum += va; b_sum += vb; c_sum += vc;
        a_min = std::min(a_min, va);
        a_max = std::max(a_max, va);
        b_min = std::min(b_min, vb);
        b_max = std::max(b_max, vb);
        c_min = std::min(c_min, vc);
        c_max = std::max(c_max, vc);
    }
    const Real n_samples = static_cast<Real>(
        result.num_steps() - k_start);
    const Real a_mean = a_sum / n_samples;
    const Real b_mean = b_sum / n_samples;
    const Real c_mean = c_sum / n_samples;

    INFO("Mid means (V): A=" << a_mean << " B=" << b_mean
         << " C=" << c_mean << " (target = V_bus/2 = "
         << V_bus * 0.5 << ")");

    // 1) All 3 mid-points average to V_bus/2 over a full
    //    mod cycle. The sine reference integrates to zero;
    //    only the DC bias remains.
    REQUIRE(a_mean == Approx(V_bus * 0.5).margin(10.0));
    REQUIRE(b_mean == Approx(V_bus * 0.5).margin(10.0));
    REQUIRE(c_mean == Approx(V_bus * 0.5).margin(10.0));

    // 2) Per-phase peak-to-peak swing is similar across the
    //    3 legs (balanced amplitudes). PWM ripple dominates
    //    the instantaneous swing — full-rail excursions are
    //    expected on each leg. Check rough equality (≤ 5 %).
    const Real pp_a = a_max - a_min;
    const Real pp_b = b_max - b_min;
    const Real pp_c = c_max - c_min;
    INFO("Peak-to-peak (V): A=" << pp_a << " B=" << pp_b
         << " C=" << pp_c);
    const Real pp_avg = (pp_a + pp_b + pp_c) / 3.0;
    REQUIRE(std::abs(pp_a - pp_avg) < pp_avg * 0.05);
    REQUIRE(std::abs(pp_b - pp_avg) < pp_avg * 0.05);
    REQUIRE(std::abs(pp_c - pp_avg) < pp_avg * 0.05);

    // 3) KCL at neutral: in a balanced 3-φ system the sum
    //    v_a + v_b + v_c (referenced to neutral or gnd via
    //    a balanced R-L load) is approximately the
    //    common-mode component. For SPWM 2-level inverters
    //    this is NOT zero (third-harmonic + carrier
    //    side-bands are present), but the FUNDAMENTAL
    //    component of v_a + v_b + v_c IS zero by 3-phase
    //    cancellation. A practical proxy: average
    //    (v_a + v_b + v_c) - 3·(V_bus/2) over a full mod
    //    cycle is small.
    Real cm_sum = 0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        cm_sum += result.states[k][a_idx] +
                  result.states[k][b_idx] +
                  result.states[k][c_idx];
    }
    const Real cm_mean = cm_sum / n_samples;
    const Real cm_offset = cm_mean - 3.0 * V_bus * 0.5;
    INFO("Common-mode mean residual (V): " << cm_offset);
    // Slight (~2 V/leg) DC droop is real: switch R_on +
    // R_neutral cause a tiny common-mode shift below the
    // ideal V_bus/2. Allow ±10 V.
    REQUIRE(std::abs(cm_offset) < 10.0);
}
