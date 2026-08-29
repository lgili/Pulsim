// =============================================================================
// TR-BDF2 variable-step transient — kernel-level correctness
// =============================================================================
//
// v2.0 Phase 3 — the variable-step mode on the sparse MNA kernel.
// These pin the stepper against ANALYTIC references; the end-to-end
// gates (buck < 50 ms, flyback+snubber parity, rectifier) live in
// python/tests/test_engine_auto.py.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/builder/circuit_builder.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"

#include <cmath>

using namespace pulsim;
using Catch::Approx;

namespace {
topology::SwitchStateMask all_open(Size n) {
    return topology::SwitchStateMask(n);
}
}  // namespace

TEST_CASE("TR-BDF2: RC charge matches the analytic exponential",
          "[v2][trbdf2]") {
    // 5 V source, R = 1k, C = 1 µF (tau = 1 ms), 5 ms horizon.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 5.0);
    b.add_resistor("R", "in", "n1", 1e3);
    b.add_capacitor("C", "n1", "gnd", 1e-6);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);

    solver::TrBdf2Options o;
    o.t_start = 0.0;
    o.t_end = 5e-3;
    o.rtol = 1e-6;
    o.atol = 1e-9;
    // Unswitched circuit: lift the (gate-sampling) step ceiling so
    // the LTE controller, not the default h_max = span/1000, sets
    // the pace.
    o.h_max = 2e-4;
    const Size n_sw = b.graph().num_switches();

    solver::TrBdf2Stats st;
    const auto res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o,
        [&](Real) { return all_open(n_sw); }, {}, std::nullopt,
        &st);

    REQUIRE(res.times.size() > 4);
    // v(n1) is state row of node n1.
    const Index n1 = b.node_id_of("n1");
    const Real tau = 1e-3;
    // Check EVERY recorded sample against the analytic curve.
    Real worst = 0.0;
    for (Size k = 0; k < res.times.size(); ++k) {
        const Real t = res.times[k];
        const Real ref = 5.0 * (1.0 - std::exp(-t / tau));
        worst = std::max(worst,
                          std::abs(res.states[k][n1] - ref));
    }
    INFO("accepted=" << st.n_accept << " rejected=" << st.n_reject
                      << " solves=" << st.n_solves);
    // rtol is a LOCAL per-step tolerance; the GLOBAL error
    // accumulates over ~100 steps. 2e-4 on a 5 V scale = 4e-5
    // relative — consistent with rtol=1e-6 local.
    REQUIRE(worst < 2e-4);
    // The controller must be doing its job: a fixed grid needs
    // ~5000 steps at dt = 1e-6 for comparable resolution.
    REQUIRE(st.n_accept < 800);

    // Tolerance proportionality: two decades tighter rtol must cut
    // the global error by well over a decade (order-2 controller:
    // err_global ~ rtol^(2/3)).
    solver::TrBdf2Options o2 = o;
    o2.rtol = 1e-8;
    o2.atol = 1e-11;
    const auto res2 = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o2,
        [&](Real) { return all_open(n_sw); });
    Real worst2 = 0.0;
    for (Size k = 0; k < res2.times.size(); ++k) {
        const Real ref =
            5.0 * (1.0 - std::exp(-res2.times[k] / tau));
        worst2 = std::max(worst2,
                           std::abs(res2.states[k][n1] - ref));
    }
    REQUIRE(worst2 < worst / 10.0);
}

TEST_CASE("TR-BDF2: stiff snubber mode is crossed, not ground",
          "[v2][trbdf2]") {
    // tau2 = 1 ns snubber on a tau1 = 1 ms circuit: an explicit or
    // trap method grinds/rings; L-stable TR-BDF2 walks over the
    // fast mode once it has decayed.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 10.0);
    b.add_resistor("R1", "in", "n1", 1e3);        // with C1: 1 ms
    b.add_capacitor("C1", "n1", "gnd", 1e-6);
    b.add_resistor("R2", "n1", "n2", 1.0);        // with C2: 1 ns
    b.add_capacitor("C2", "n2", "gnd", 1e-9);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);

    solver::TrBdf2Options o;
    o.t_end = 5e-3;
    o.rtol = 1e-5;
    o.atol = 1e-8;
    const Size n_sw = b.graph().num_switches();
    solver::TrBdf2Stats st;
    const auto res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o,
        [&](Real) { return all_open(n_sw); }, {}, std::nullopt,
        &st);

    const Index n1 = b.node_id_of("n1");
    const Index n2 = b.node_id_of("n2");
    const Real v_end = res.states[res.times.size() - 1][n1];
    REQUIRE(v_end == Approx(10.0 * (1.0 - std::exp(-5.0)))
                          .epsilon(1e-3));
    // n2 follows n1 quasi-statically once the ns mode decays.
    const Real v2_end = res.states[res.times.size() - 1][n2];
    REQUIRE(v2_end == Approx(v_end).margin(1e-4));
    // The 1 ns mode must NOT pin the step: total steps stay small.
    INFO("accepted=" << st.n_accept << " rejected=" << st.n_reject);
    REQUIRE(st.n_accept < 2000);
}

TEST_CASE("TR-BDF2: gate edge lands exactly and restarts",
          "[v2][trbdf2]") {
    // Switched RC: switch closes at t = 1.2345e-3 (an awkward
    // instant no grid hits). The step must LAND on it and the
    // charge curve after it must match the analytic response.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 5.0);
    b.add_switch("S", "in", "n1", 1e3, 1e-9);
    b.add_resistor("R", "n1", "n2", 1e3);
    b.add_capacitor("C", "n2", "gnd", 1e-6);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);

    const Real t_close = 1.2345e-3;
    const Size n_sw = b.graph().num_switches();
    auto sf = [&](Real t) {
        topology::SwitchStateMask m(n_sw);
        m.set(0, t >= t_close);
        return m;
    };

    solver::TrBdf2Options o;
    o.t_end = 6e-3;
    o.rtol = 1e-6;
    o.atol = 1e-9;
    solver::TrBdf2Stats st;
    const auto res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o, sf, {}, std::nullopt, &st);

    REQUIRE(st.n_gate_events == 1);
    // One recorded sample sits ON the edge (within h_min).
    Real nearest = 1e9;
    for (const Real t : res.times) {
        nearest = std::min(nearest, std::abs(t - t_close));
    }
    REQUIRE(nearest < 1e-9);
    // After the edge: v(n2) = 5(1 - exp(-(t - t_close)/tau)) with
    // tau ≈ (1k + 1/g_on)·C.
    const Index n2 = b.node_id_of("n2");
    const Real tau = (1e3 + 1e-3) * 1e-6;
    Real worst = 0.0;
    for (Size k = 0; k < res.times.size(); ++k) {
        const Real t = res.times[k];
        if (t < t_close + 1e-9) { continue; }
        const Real ref =
            5.0 * (1.0 - std::exp(-(t - t_close) / tau));
        worst = std::max(worst,
                          std::abs(res.states[k][n2] - ref));
    }
    REQUIRE(worst < 1e-4);
}

TEST_CASE("TR-BDF2: half-wave rectifier localizes diode edges",
          "[v2][trbdf2]") {
    // Sine source, series switched diode, resistive load. The diode
    // turn-on instant (v >= V_th) and turn-off (current zero) are
    // localized between steps; the output must clamp negative
    // half-cycles and track the source on positive ones.
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0, 50.0,
                               0.0);
    b.add_diode("D", "ac", "out", 1e3, 1e-9, 0.7);
    b.add_resistor("Rl", "out", "gnd", 100.0);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);

    solver::TrBdf2Options o;
    o.t_end = 40e-3;               // two mains cycles
    o.rtol = 1e-6;
    o.atol = 1e-9;
    const Size n_sw = b.graph().num_switches();
    solver::TrBdf2Stats st;
    const auto res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o,
        [&](Real) { return all_open(n_sw); }, {}, std::nullopt,
        &st);

    REQUIRE(st.n_diode_events >= 4);   // 2 on + 2 off
    const Index out = b.node_id_of("out");
    Real v_peak = 0.0, v_min = 0.0;
    Real worst_on = 0.0;
    for (Size k = 0; k < res.times.size(); ++k) {
        const Real t = res.times[k];
        const Real v = res.states[k][out];
        v_peak = std::max(v_peak, v);
        v_min = std::min(v_min, v);
        // On strongly-conducting stretches the divider is exact:
        // v_out = (v_ac)·Rl/(Rl + 1/g_on) when v_ac well above
        // V_th.
        const Real v_ac = 10.0 * std::sin(2.0 * M_PI * 50.0 * t);
        if (v_ac > 2.0) {
            const Real ref = v_ac * 100.0 / (100.0 + 1e-3);
            worst_on = std::max(worst_on, std::abs(v - ref));
        }
    }
    REQUIRE(v_peak == Approx(10.0 * 100.0 / 100.001).epsilon(2e-3));
    REQUIRE(v_min > -1e-4);            // blocked half-cycle clamps
    REQUIRE(worst_on < 2e-2);
}

TEST_CASE("TR-BDF2: nonlinear devices refuse with the mechanism",
          "[v2][trbdf2]") {
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 5.0);
    b.add_resistor("R", "in", "n1", 1e3);
    b.add_nonlinear_diode("D", "n1", "gnd", {});
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    solver::TrBdf2Options o;
    o.t_end = 1e-3;
    const Size n_sw = b.graph().num_switches();
    REQUIRE_THROWS_AS(
        solver::run_transient_trbdf2(
            cache, b.graph(), b.pool(), o,
            [&](Real) { return all_open(n_sw); }),
        std::invalid_argument);
}
