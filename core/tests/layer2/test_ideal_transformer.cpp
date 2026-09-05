// =============================================================================
// Phase 4 C.4 — ideal transformer
// =============================================================================
//
//   * v_s = n·v_p and i_p = −n·i_s, so power in equals power out
//     EXACTLY — the sign check that matters, because a wrong sign on
//     the reflected current converges just fine and delivers energy
//     from nowhere.
//   * It transforms DC (no frequency dependence): the DC operating
//     point of a resistively loaded transformer is the same ratio.
//   * The T-model built from it — leakage + ideal + magnetising —
//     reproduces the coupled-inductor transformer's waveforms below
//     saturation. That is the identity the saturable transformer
//     rests on: L_p = L_lp + L_m, M = n·L_m, L_s = n²·L_m + L_ls.
//     With all leakage on the primary (the cantilever choice):
//     n = √(L_s/L_p)/k, L_m = k²·L_p, L_lp = (1−k²)·L_p.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/pwl/cache.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/solver/run_transient.hpp"

#include <cmath>

using namespace pulsim;
using namespace pulsim::builder;
using namespace pulsim::pwl;
using namespace pulsim::solver;
using namespace pulsim::topology;
using Catch::Approx;

namespace {
SimulationOptions dc_like() {
    return SimulationOptions{.t_start = 0.0, .t_end = 1.0, .dt = 0.1};
}
}  // namespace

TEST_CASE("IdealTransformer: v_s = n v_p, i_p = -n i_s, power balances",
          "[v2][c4][ideal_transformer][unit]") {
    // 12 V source through 1 Ω into a 1:3 transformer loaded with 90 Ω.
    // Referred to the primary the load is 10 Ω, so i_p = 12/11 A,
    // v_p = 120/11 V, v_s = 3 v_p, i_s = −v_s/90.
    CircuitBuilder b;
    b.add_voltage_source("V", "src", "gnd", 12.0);
    b.add_resistor("Rs", "src", "p", 1.0);
    b.add_ideal_transformer("T", "p", "gnd", "s", "gnd", 3.0);
    b.add_resistor("RL", "s", "gnd", 90.0);

    PwlStateSpaceCache cache(b.graph(), b.pool());
    cache.build();
    auto sw = [](Real) { return SwitchStateMask(0); };
    auto r = run_transient(cache, b.graph(), b.pool(), dc_like(), sw);
    const auto& x = r.states[r.num_steps() - 1];

    const Real v_p = x[b.node_id_of("p")];
    const Real v_s = x[b.node_id_of("s")];
    const Real i_s = x[b.pool().branch_var_id_for_source(
        /*T is branch 2*/ 2, b.graph())];
    const Real i_p = (12.0 - v_p) / 1.0;          // through Rs
    CHECK(v_p == Approx(120.0 / 11.0).epsilon(1e-9));
    CHECK(v_s == Approx(3.0 * v_p).epsilon(1e-9));
    CHECK(i_p == Approx(12.0 / 11.0).epsilon(1e-9));
    // Secondary current: leaves s_from into the load, so negative
    // in the branch convention, and i_p = −n·i_s.
    CHECK(i_s == Approx(-v_s / 90.0).epsilon(1e-9));
    CHECK(i_p == Approx(-3.0 * i_s).epsilon(1e-9));
    // Exact power balance: nothing is created in the middle.
    CHECK(v_p * i_p + v_s * i_s == Approx(0.0).margin(1e-12));
}

TEST_CASE("IdealTransformer: refuses a non-positive ratio and shorted "
          "terminals by name", "[v2][c4][ideal_transformer][unit]") {
    CircuitBuilder b;
    CHECK_THROWS_WITH(b.add_ideal_transformer("T", "a", "gnd", "s", "gnd", 0.0),
                      Catch::Matchers::ContainsSubstring("positive"));
    CHECK_THROWS_WITH(b.add_ideal_transformer("T", "a", "gnd", "s", "gnd", -2.0),
                      Catch::Matchers::ContainsSubstring("Reverse"));
    CHECK_THROWS_WITH(b.add_ideal_transformer("T", "a", "a", "s", "gnd", 2.0),
                      Catch::Matchers::ContainsSubstring("same node"));
}

TEST_CASE("IdealTransformer: the T-model reproduces the coupled-inductor "
          "transformer below saturation", "[v2][c4][ideal_transformer]") {
    // The example flyback's transformer, driven linearly: a 100 kHz
    // sine through 0.5 Ω into the primary, 5 Ω on the secondary.
    const Real L_p = 100e-6, L_s = 25e-6, k = 0.95;
    const Real n    = std::sqrt(L_s / L_p) / k;      // 0.5263
    const Real L_m  = k * k * L_p;                   // 90.25 µH
    const Real L_lp = (1.0 - k * k) * L_p;           //  9.75 µH

    auto run = [&](bool t_model) {
        CircuitBuilder b;
        b.add_sine_voltage_source("V", "src", "gnd",
                                  /*v_dc=*/0.0, /*v_amplitude=*/48.0,
                                  /*frequency=*/100e3, /*phase=*/0.0);
        b.add_resistor("Rs", "src", "p", 0.5);
        if (t_model) {
            b.add_inductor("Llp", "p", "m", L_lp);
            b.add_inductor("Lm", "m", "gnd", L_m);
            b.add_ideal_transformer("T", "m", "gnd", "s", "gnd", n);
        } else {
            b.add_transformer("T", "p", "gnd", "s", "gnd", L_p, L_s, k);
        }
        b.add_resistor("RL", "s", "gnd", 5.0);
        PwlStateSpaceCache cache(b.graph(), b.pool());
        SimulationOptions opts{.t_start = 0.0, .t_end = 50e-6,
                               .dt = 1e-8};
        cache.build(opts.dt);
        auto sw = [](Real) { return SwitchStateMask(0); };
        auto r = run_transient(cache, b.graph(), b.pool(), opts, sw);
        std::vector<Real> vs;
        const Index s = b.node_id_of("s");
        for (Size i = 0; i < r.num_steps(); ++i) vs.push_back(r.states[i][s]);
        return vs;
    };
    const auto a = run(false);
    const auto t = run(true);
    REQUIRE(a.size() == t.size());
    Real peak = 0, err = 0;
    for (Size i = 0; i < a.size(); ++i) {
        peak = std::max(peak, std::abs(a[i]));
        err  = std::max(err, std::abs(a[i] - t[i]));
    }
    INFO("peak v_s = " << peak << " V, max |coupled - T-model| = " << err);
    REQUIRE(peak > 5.0);                 // actually transforming
    REQUIRE(err < 1e-6 * peak);          // same linear circuit
}
