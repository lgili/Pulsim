// =============================================================================
// Layer 9 — Common-source MOSFET amplifier showcase
// =============================================================================
//
// END-TO-END validation of V13 SH1 MOSFET in a textbook
// common-source amplifier:
//   - YAML loader reads `examples/v2/common_source_amplifier.
//     yaml`: V_DD = 10 V, R_D = 5 kΩ, M1 (K=1e-3, V_T=2),
//     V_in = 3 V DC + 0.1 V·sin(2π·1 kHz·t) → gate.
//   - `make_combined_diode_mosfet_refresh()` provides the
//     Newton refresh that stamps M1's 3-terminal Jacobian
//     per iteration.
//
// Expected behavior:
//   * DC operating point: V_drain ≈ 4.55 V (verified in
//     the layer2 unit test for the same parameters).
//   * Small-signal: g_m = 2·K·V_OV = 2 mA/V at V_OV = 1 V.
//   * Voltage gain Av = −g_m·R_D = −10 V/V → 0.1 V gate
//     sinusoid → ~1 V drain sinusoid (inverted phase).
//   * Peak-to-peak v_drain ≈ 2 V.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/solver/run_transient.hpp"
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
using namespace pulsim::topology;
using Catch::Approx;

namespace {

std::string locate_cs_amp_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" / "v2" /
            "common_source_amplifier.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/common_source_amplifier.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("Showcase: common-source MOSFET amplifier (V13 + V11)",
          "[v2][layer9][showcase][mosfet][common_source]") {
    const std::string path = locate_cs_amp_yaml();
    INFO("common_source_amplifier.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/common_source_amplifier.yaml not "
             "located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    // No switches; M1 is the only nonlinear device.
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 0);
    auto switch_fn = [](Real) {
        return SwitchStateMask(0);
    };

    auto nl_refresh = make_combined_diode_mosfet_refresh();

    // Increase Newton iter cap — the MOSFET startup from
    // x=0 needs a few warm-up Newton steps per timestep.
    SimulationOptions opts = loaded.options;
    opts.max_newton_iterations = 100;

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        opts, switch_fn, {}, false, nl_refresh);
    REQUIRE(result.num_steps() > 100);

    const Index drain_idx =
        loaded.builder.node_id_of("drain");
    const Index gate_idx =
        loaded.builder.node_id_of("gate");
    REQUIRE(drain_idx >= 0);
    REQUIRE(gate_idx >= 0);

    // Sample the LAST AC cycle (1 kHz period = 1 ms; sim
    // ran 3 ms → drop first 2 cycles for warm-up).
    constexpr Real T_ac = 1.0 / 1000.0;
    const Size k_start = result.num_steps() -
        static_cast<Size>(T_ac / loaded.options.dt);
    const Size k_end = result.num_steps();

    Real vd_sum = 0;
    Real vd_min = 1e9, vd_max = -1e9;
    Real vg_min = 1e9, vg_max = -1e9;
    for (Size k = k_start; k < k_end; ++k) {
        const Real vd = result.states[k][drain_idx];
        const Real vg = result.states[k][gate_idx];
        vd_sum += vd;
        vd_min = std::min(vd_min, vd);
        vd_max = std::max(vd_max, vd);
        vg_min = std::min(vg_min, vg);
        vg_max = std::max(vg_max, vg);
    }
    const Real vd_mean = vd_sum /
        static_cast<Real>(k_end - k_start);
    const Real vd_pkpk = vd_max - vd_min;
    const Real vg_pkpk = vg_max - vg_min;

    INFO("V_drain mean = " << vd_mean << " V");
    INFO("V_drain peak-to-peak = " << vd_pkpk << " V");
    INFO("V_gate peak-to-peak = " << vg_pkpk << " V");

    // 1) DC operating point ~ 4.55 V (verified by the
    //    layer2 unit test for identical parameters).
    REQUIRE(vd_mean == Approx(4.55).margin(0.15));

    // 2) Gate swing matches the input source amplitude
    //    (0.1 V → 0.2 V p-p).
    REQUIRE(vg_pkpk == Approx(0.2).margin(0.01));

    // 3) Drain swing matches gate swing × |Av| where
    //    |Av| = g_m·R_D = 2 mA/V · 5 kΩ = 10. So
    //    v_drain_pkpk ≈ 2 V. Allow ±20 %% slack to absorb
    //    bias-dependent variation in g_m within the swing
    //    range.
    REQUIRE(vd_pkpk == Approx(2.0).margin(0.4));
    REQUIRE(vd_pkpk > vg_pkpk * 5.0);   // gain > 5 (clearly amplifying)
}
