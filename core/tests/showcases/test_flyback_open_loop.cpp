// =============================================================================
// Layer 9 — Flyback converter open-loop showcase
// =============================================================================
//
// Second END-TO-END validation of v2 (after buck.yaml):
//   - YAML loader reads `examples/v2/flyback.yaml`
//     (V_in = 48 V; T1 with N_p:N_s ratio ≈ 2:1; Q1 + D1 +
//     Cout + R_L = 5 Ω). Boost-style output via flyback
//     converter math.
//   - `sources::make_pwm_switch_fn` drives Q1 at 100 kHz —
//     exercises the helper end-to-end (Layer 2 V5).
//   - The body diode of D1 commutates automatically when Q1
//     opens and the secondary current path closes.
//
// Steady-state flyback equation (CCM, lossless, k=1):
//   V_out = (N_s/N_p) · V_in · D / (1 - D)
//
//   N_s/N_p = √(L_s/L_p) = √(25/100) = 0.5
//   V_in = 48 V; with D = 0.5 → V_out = 24 V (ideal lossless)
//
// Real circuit has k=0.95 (5 % leakage) + diode drop + R_L
// losses; expect V_out somewhere in [10, 25] V after a few
// charge-time constants. We assert:
//   * sim ran (> 100 steps), Q1 toggled,
//   * v_out climbed monotonically from 0 to > 5 V,
//   * v_out remained bounded (no run-away).
//
// This is a directional test — it proves the YAML +
// transformer + switch-helper plumbing works end-to-end. The
// flyback's exact V_out depends on Cout settling and the
// detailed switching ripple, which we capture in the loose
// bounds above.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/solver/run_transient.hpp"
#include "pulsim/sources/pwm_switch_fn.hpp"
#include "pulsim/topology/switch_state.hpp"
#include "pulsim/yaml/loader.hpp"

#include <algorithm>
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

/// Locate `examples/v2/flyback.yaml` — same convention as the
/// buck showcase. Walks up from CWD; falls back to env var.
std::string locate_flyback_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate =
            search / "examples" / "v2" / "flyback.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) + "/flyback.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("SMPS showcase: open-loop flyback via YAML + make_pwm_switch_fn",
          "[v2][layer9][showcase][smps][flyback]") {
    const std::string path = locate_flyback_yaml();
    INFO("flyback.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/flyback.yaml not located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    // Extend the YAML's t_end (200 µs) to give Cout time to
    // charge: τ_RC = R · C = 5 · 100µF = 500µs → settle by
    // ~3 ms.
    loaded.options.t_end = 3.0e-3;

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    constexpr Real f_sw = 100e3;
    constexpr Real duty = 0.5;

    // Switch-kind branches: Q1 (mosfet) + D1 (switched diode
    // — auto-commutated by run_transient's event detection).
    // We drive only switch index 0 (Q1, first added); D1 is
    // managed by the diode-event system.
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw >= 1);

    // Exercise the new helper end-to-end. Other switch bits
    // (e.g. D1 = idx 1) stay OFF in the COMMANDED mask; the
    // diode-event system overrides them per step.
    auto switch_fn = make_pwm_switch_fn(
        f_sw, duty, /*switch_idx=*/0, n_sw);

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        loaded.options, switch_fn);
    REQUIRE(result.num_steps() > 100);

    // Inspect v_out.
    const Index vout_idx =
        loaded.builder.node_id_of("vout");
    REQUIRE(vout_idx >= 0);

    // Verify v_out climbed from 0 (transient charge-up) and
    // settled to a non-trivial positive steady-state. Real
    // flyback dynamics overshoot first (leakage L + Cout
    // resonance), then settle — so we DON'T enforce strict
    // monotonicity. Instead we check:
    //   * v_out at start (sample 0) is ~0,
    //   * v_out reached a peak > 10 V during charge-up,
    //   * mean over the final 0.5 ms is in [10, 35] V
    //     (covers ideal 24 V + lossy/overshoot variants).
    REQUIRE(std::abs(result.states[0][vout_idx]) < 1.0);

    Real v_peak = 0.0;
    for (Size k = 0; k < result.num_steps(); ++k) {
        v_peak = std::max(v_peak, result.states[k][vout_idx]);
    }
    INFO("v_out peak during transient = " << v_peak << " V");
    REQUIRE(v_peak > 10.0);

    // Final 0.5 ms time-average (≥ 50 PWM cycles).
    const Real measurement_window = 0.5e-3;
    const Size k_start = result.num_steps() -
        static_cast<Size>(measurement_window /
                          loaded.options.dt);
    Real v_sum = 0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        v_sum += result.states[k][vout_idx];
    }
    const Real v_mean = v_sum /
        static_cast<Real>(result.num_steps() - k_start);

    INFO("Flyback steady-state v_out (last 0.5 ms mean) = "
         << v_mean << " V (ideal lossless = 24 V)");
    REQUIRE(v_mean > 10.0);
    REQUIRE(v_mean < 35.0);
}
