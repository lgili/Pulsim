// =============================================================================
// Layer 9 — Boost converter with REALISTIC SH1 MOSFET (V14)
// =============================================================================
//
// Companion to test_boost_realistic_igbt.cpp — same topology
// but using V13 SH1 MOSFET instead of IGBT. With the V14
// PulseVoltageSource `rise_time` trick (ramped gate over
// 2 µs), Newton converges through the gate transition.

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

std::string locate_boost_mosfet_yaml() {
    namespace fs = std::filesystem;
    auto search = fs::current_path();
    for (int i = 0; i < 10; ++i) {
        const auto candidate = search / "examples" / "v2" /
            "boost_realistic_mosfet_v14.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (search.parent_path() == search) break;
        search = search.parent_path();
    }
    if (const char* env =
            std::getenv("PULSIM_EXAMPLES_DIR")) {
        return std::string(env) +
            "/boost_realistic_mosfet_v14.yaml";
    }
    return {};
}

}  // namespace

TEST_CASE("Showcase: real boost converter with SH1 MOSFET + ramped gate",
          "[v2][layer9][showcase][smps][boost_mosfet_v14]") {
    const std::string path = locate_boost_mosfet_yaml();
    INFO("boost_realistic_mosfet_v14.yaml path: " << path);
    if (path.empty()) {
        WARN("examples/v2/boost_realistic_mosfet_v14.yaml "
             "not located — skipping");
        return;
    }

    auto loaded = yaml::load_file(path);
    REQUIRE(loaded.builder.num_branches() > 0);

    PwlStateSpaceCache cache(loaded.builder.graph(),
                              loaded.builder.pool());
    cache.build(loaded.options.dt);

    const Size n_sw = loaded.builder.graph().num_switches();
    REQUIRE(n_sw == 1);   // D_boost

    auto switch_fn = [n_sw](Real) {
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
        opts, switch_fn, {}, false, nl_refresh);
    REQUIRE(result.num_steps() > 1000);

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

    INFO("MOSFET-boost V_out mean = " << v_mean
         << " V; ripple = " << v_ripple);

    REQUIRE(v_mean > 12.5);    // boosted above V_in
    REQUIRE(v_mean < 25.0);    // bounded by physics
    REQUIRE(v_ripple < 5.0);   // LC filter OK
}
