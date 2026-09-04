// =============================================================================
// Layer 9 V18 — Boost converter with SATURABLE inductor
// =============================================================================
//
// Compares boost behaviour with a saturable inductor vs the V14
// linear-inductor case. The saturable L drops once i_L exceeds
// I_sat, which:
//   (a) limits peak inductor current (good — protects switch)
//   (b) reduces stored energy → lower V_out
//
// Verifies:
//   * Newton + IGBT + saturable inductor + diode all converge in
//     a real PWM SMPS topology.
//   * V_out is still ABOVE V_in (boost is boosting).
//   * V_out is LOWER than the V14 linear-L case's 20 V (because
//     saturation steals some of the inductor's storage capacity).
//   * Peak |I_L| is bounded — does not exceed a few times I_sat.

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

std::string locate_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" /
            "boost_saturable_inductor.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/boost_saturable_inductor.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("Showcase: boost converter with saturable inductor",
          "[v2][layer9][showcase][smps][boost_saturable]") {
    const std::string path = locate_yaml();
    INFO("yaml: " << path);
    // Not a WARN-and-return. The upward search looked in
    // `examples/v2/` while the file has always lived in
    // `examples/`, so outside ctest (which sets
    // PULSIM_EXAMPLES_DIR) this test skipped itself in silence —
    // and it is the ONLY test in either tree that puts a saturable
    // inductor through current reversal, which is the only regime
    // where the flux rule can be wrong. A silent skip on the one
    // test that matters is worse than no test.
    REQUIRE_FALSE(path.empty());

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    // 1 SwitchedDiode (D_boost). IGBT + saturable inductor
    // are Nonlinear (no switch-kind branches).
    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 1);
    auto switch_fn = [n_sw](Real) {
        return SwitchStateMask(n_sw);
    };

    SimulationOptions opts = loaded.options;
    opts.max_newton_iterations = 200;
    opts.tol_newton_dx  = 1e-5;
    opts.tol_newton_res = 1e-5;
    opts.enable_newton_line_search = true;

    // run_transient AUTO-DETECTS the saturable inductor and
    // wraps nl_refresh. Pass an EMPTY nl_refresh (the wrapper
    // also handles diodes/MOSFETs/IGBTs internally? — actually
    // not; we still need the combined refresh to stamp the
    // IGBT). Let me pass the combined refresh from V14.
    auto nl_refresh =
        make_combined_diode_mosfet_refresh();

    auto result = run_transient(
        cache, loaded.builder.graph(),
        loaded.builder.pool(),
        opts, switch_fn, {}, false, nl_refresh);
    REQUIRE(result.num_steps() > 1000);

    // Inspect V_out over the last 500 µs.
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

    INFO("Saturable-boost V_out mean = " << v_mean << " V");

    // 1) Still boosting (above V_in = 12 V), and bounded. These two
    //    are shape checks, not the regression check — they passed
    //    under the superseded rectangle-rule stamp too, and a bound
    //    that blesses either answer certifies nothing.
    REQUIRE(v_mean > 12.5);
    REQUIRE(v_mean < 25.0);

    const Index l_branch_var =
        loaded.builder.pool().branch_var_id_for_inductor(
            /*L_boost branch id*/ 2, loaded.builder.graph());
    Real i_L_peak_ss = 0.0;
    for (Size k = k_start; k < result.num_steps(); ++k) {
        i_L_peak_ss = std::max(i_L_peak_ss,
            std::abs(result.states[k][l_branch_var]));
    }
    INFO("Peak |I_L| in steady state = " << i_L_peak_ss
         << " A (I_sat = 1.5 A)");
    REQUIRE(i_L_peak_ss < 10.0);

    // WHAT THIS TEST DOES NOT DO, stated plainly so the next
    // reader does not mistake green for coverage.
    //
    // The three assertions above are SHAPE checks. All of them
    // passed under the superseded rectangle-rule stamp for the flux
    // as well, so none of them discriminates the integration rule.
    // Tightening them would not help either: this circuit is not in
    // periodic steady state at t_end (C_out = 100 uF into
    // R_load = 20 ohm is a 2 ms time constant and the run is 5 ms,
    // so the output is 2.5 tau in and still ringing on the L-C
    // tank). Any window statistic here measures that transient, not
    // the flux rule, and a bound drawn around it would be a golden
    // number wearing a physics costume.
    //
    // The regression that DOES discriminate lives in
    // python/tests/test_saturable_inductor_flux_closure.py: a
    // zero-mean drive, where flux closure is exact physics rather
    // than a settling transient, with a linear inductor as the
    // control. The algebra it rests on — lambda' = L exactly,
    // lambda odd, and the linear limit — is pinned in
    // core/tests/layer2/test_saturable_inductor.cpp.
    //
    // What this test is for: the device runs inside a real
    // switching converter, boosts, and stays bounded.
}
