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
#include "pulsim/pwl/nonlinear_refresh_mosfet_level1.hpp"
#include "pulsim/solver/trbdf2_transient.hpp"

#include <cmath>
#include <limits>
#include <vector>

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

TEST_CASE("TR-BDF2: nonlinear devices are stamped without a callback",
          "[v2][trbdf2]") {
    // add-engine-conformance-matrix: the kernel owns the static Newton
    // chain. This call used to be refused (and the raw fixed engine ran
    // the diode as an open circuit, 5.000 V); now the diode clamps.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 5.0);
    b.add_resistor("R", "in", "n1", 1e3);
    b.add_nonlinear_diode("D", "n1", "gnd", {});
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    solver::TrBdf2Options o;
    o.t_end = 1e-3;
    const Size n_sw = b.graph().num_switches();
    const auto r = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o,
        [&](Real) { return all_open(n_sw); });
    const Real v = r.states.back()[b.node_id_of("n1")];
    REQUIRE(v > 0.3);
    REQUIRE(v < 1.0);
}

// -----------------------------------------------------------------------------
// Engine robustness (v2.0 follow-up): opaque gates, held diodes, a
// floating DC side.
// -----------------------------------------------------------------------------

namespace {
/// Time-weighted mean of node `n` over exactly [t0, t_end] (trapezoid on
/// the variable grid, the window start interpolated: starting at the first
/// sample past t0 is an O(h) metric error that alone read 1.95 for a
/// second-order ratio of 4.09).
Real tail_mean(const solver::SimulationResult& res, Index n, Real t0) {
    Real acc = 0.0;
    for (Size k = 1; k < res.times.size(); ++k) {
        const Real ta = res.times[k - 1], tb = res.times[k];
        if (tb <= t0 || tb <= ta) { continue; }   // zero-width: no area
        const Real va = res.states[k - 1][n], vb = res.states[k][n];
        const Real a0 = std::max(ta, t0);
        const Real v0 = va + (vb - va) * (a0 - ta) / (tb - ta);
        acc += 0.5 * (tb - a0) * (v0 + vb);
    }
    return acc / (res.times.back() - t0);
}
}  // namespace

TEST_CASE("TR-BDF2: an opaque switch_fn lands every edge of a step "
          "that spans several", "[v2][trbdf2][robustness]") {
    // 100 kHz, D = 0.3 PWM from a plain lambda (no edge oracle) with a
    // step ceiling of 3.5 periods. The edge search used to sample only
    // t and t + h and bisect [t, t + h] as if it held ONE edge, so a
    // long step could skip whole pulses or land on a later edge; the
    // step now samples h/4 .. h and the first sub-interval that
    // changes, and caps h at twice the narrowest pulse seen (3 µs).
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 10.0);
    b.add_switch("S", "in", "x", 1e3, 1e-9);
    b.add_resistor("R", "x", "y", 10.0);
    b.add_capacitor("C", "y", "gnd", 1e-6);
    b.add_resistor("RL", "y", "gnd", 100.0);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    const Size n_sw = b.graph().num_switches();
    const Real T = 1e-5;
    auto sf = [&](Real t) {
        topology::SwitchStateMask m(n_sw);
        m.set(0, std::fmod(t, T) < 0.3 * T);
        return m;
    };
    solver::TrBdf2Options o;
    o.t_end = 1e-3;
    o.h_max = 3.5 * T;
    o.rtol = 1e-5;
    o.atol = 1e-8;
    solver::TrBdf2Stats st;
    const auto res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o, sf, {}, std::nullopt, &st);

    // 100 periods: a falling edge in each, a rising edge in all but
    // the first (t = 0 starts ON), the run ends on the last rising one.
    REQUIRE(st.n_gate_events == 199);
    REQUIRE(st.gate_pulse_min == Approx(3e-6).epsilon(1e-6));
    REQUIRE(st.n_gate_undersampled == 0);
    // The fixed engine gives 7.4907 V over the second half (dt = 1e-8
    // and 5e-9 agree to 2e-5 V).
    const Real v = tail_mean(res, b.node_id_of("y"), 0.5e-3);
    REQUIRE(v == Approx(7.4907).margin(2e-3));
}

TEST_CASE("TR-BDF2: a V_th = 0 bridge conducts on both half-cycles",
          "[v2][trbdf2][robustness]") {
    // 230 V rms behind 0.1 ohm, V_th = 0 bridge, 470 µF || 1 kOhm. D1
    // and D3 cross zero together; the crossing detector read the next
    // step's start from the PRE-cascade solve (D1 at +eps), saw no sign
    // change while D1's forward bias grew to 6.9 V, and the bridge ran
    // HALF-WAVE at 318.79 V. The fixed engine gives 322.059 V
    // (dt = 1e-6).
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vac", "s", "bb", 0.0, 325.27, 50.0, 0.0);
    b.add_resistor("Rs", "s", "a", 0.1);
    b.add_diode("D1", "a", "dc", 1e3, 1e-9, 0.0);
    b.add_diode("D2", "bb", "dc", 1e3, 1e-9, 0.0);
    b.add_diode("D3", "gnd", "bb", 1e3, 1e-9, 0.0);
    b.add_diode("D4", "gnd", "a", 1e3, 1e-9, 0.0);
    b.add_capacitor("C", "dc", "gnd", 470e-6);
    b.add_resistor("RL", "dc", "gnd", 1e3);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    const Size n_sw = b.graph().num_switches();
    solver::TrBdf2Options o;
    o.t_end = 0.1;
    o.rtol = 1e-5;
    o.atol = 1e-8;
    solver::TrBdf2Stats st;
    const auto res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o,
        [&](Real) { return all_open(n_sw); }, {}, std::nullopt, &st);

    REQUIRE(tail_mean(res, b.node_id_of("dc"), 0.08)
            == Approx(322.06).margin(0.05));
    // Every diode stamped ON somewhere in the last cycle.
    REQUIRE(res.has_switch_states);
    for (Size bit = 0; bit < n_sw; ++bit) {
        bool on = false;
        for (Size k = 0; k < res.times.size(); ++k) {
            if (res.times[k] < 0.08) { continue; }
            const auto byte =
                res.switch_state_bits[k * res.switch_state_stride
                                      + bit / 8];
            on = on || ((byte >> (bit % 8)) & 1U);
        }
        INFO("diode bit " << bit);
        REQUIRE(on);
    }
}

TEST_CASE("TR-BDF2: a floating DC side runs on a longer probe step",
          "[v2][trbdf2][robustness]") {
    // Bridge with ac_b = gnd and the DC side floating (load across
    // dc_pos / dc_neg only). With every diode OFF, the DC pair hangs
    // on 4 x g_off = 4e-9 S while the 6e-14 s probe stamps 470 µF as
    // 1.6e10 S: "factorize failed (numerically singular) ... node dn".
    // The probe step now grows until the factorization holds.
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 325.0, 50.0, 0.0);
    b.add_resistor("Rs", "ac", "acx", 1e-3);
    b.add_diode("D1", "acx", "dp", 1e3, 1e-9, 0.7);
    b.add_diode("D2", "gnd", "dp", 1e3, 1e-9, 0.7);
    b.add_diode("D3", "dn", "gnd", 1e3, 1e-9, 0.7);
    b.add_diode("D4", "dn", "acx", 1e3, 1e-9, 0.7);
    b.add_capacitor("C", "dp", "dn", 470e-6);
    b.add_resistor("RL", "dp", "dn", 1e3);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    const Size n_sw = b.graph().num_switches();
    solver::TrBdf2Options o;
    o.t_end = 0.06;
    solver::TrBdf2Stats st;
    solver::SimulationResult res;
    REQUIRE_NOTHROW(res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o,
        [&](Real) { return all_open(n_sw); }, {}, std::nullopt, &st));
    REQUIRE(st.probe_floor_used > 0.0);
    REQUIRE(st.probe_floor_used <= 0.06 * 1e-6);
    // V(dp) - V(dn) against the fixed engine's 321.85 V (dt = 1e-6).
    const Index dp = b.node_id_of("dp");
    const Index dn = b.node_id_of("dn");
    Real acc = 0.0, span = 0.0;
    for (Size k = 1; k < res.times.size(); ++k) {
        if (res.times[k - 1] < 0.04) { continue; }
        const Real dt = res.times[k] - res.times[k - 1];
        acc += 0.5 * dt
               * (res.states[k - 1][dp] - res.states[k - 1][dn]
                  + res.states[k][dp] - res.states[k][dn]);
        span += dt;
    }
    REQUIRE(acc / span == Approx(321.85).margin(0.05));
}

TEST_CASE("TR-BDF2: a gate edge at a step boundary re-settles the diodes",
          "[v2][trbdf2][robustness]") {
    // 10 kHz buck, D = 0.4, from a plain lambda, with a controller tick
    // every 50 µs. The tick ends a step exactly on k·T, where the
    // turn-on edge also sits; t + h rounded to the pre-edge side, so
    // that step saw no edge and the zero-time cascade a landed edge
    // gets never ran. The next step solved Q1 ON with D1 still ON:
    // a 2.1e-11 s sample at v_sw = +6 V across two 1 mOhm stamps.
    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "vin", "gnd", 12.0);
    b.add_mosfet_with_body_diode("Q1", "vin", "sw", 1e-3, 1e9, 0.7);
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, 0.7);
    b.add_inductor("L1", "sw", "vout", 220e-6);
    b.add_capacitor("Cout", "vout", "gnd", 220e-6);
    b.add_resistor("R_L", "vout", "gnd", 8.0);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    const Size n_sw = b.graph().num_switches();
    const Size q = static_cast<Size>(b.switch_index_of("Q1"));
    const Size d = static_cast<Size>(b.switch_index_of("D1"));
    const Real T = 1e-4;
    auto sf = [&](Real t) {
        topology::SwitchStateMask m(n_sw);
        m.set(q, std::fmod(t, T) / T < 0.4);
        return m;
    };
    solver::TrBdf2Options o;
    o.t_end = 1e-3;
    o.observer_period = 50e-6;
    solver::TrBdf2Stats st;
    Size ticks = 0;
    const auto res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o, sf, {}, std::nullopt, &st, {},
        {}, [&](Real, const Vector&) { ++ticks; });
    REQUIRE(ticks == 20);
    REQUIRE(st.n_gate_boundary_settles >= 1);
    REQUIRE(res.has_switch_states);
    Size shoot_through = 0;
    for (Size k = 0; k < res.times.size(); ++k) {
        const auto* row =
            &res.switch_state_bits[k * res.switch_state_stride];
        const bool q_on = (row[q / 8] >> (q % 8)) & 1U;
        const bool d_on = (row[d / 8] >> (d % 8)) & 1U;
        shoot_through += (q_on && d_on) ? 1 : 0;
    }
    REQUIRE(shoot_through == 0);
}

TEST_CASE("TR-BDF2: controller sources at three rates tick at their "
          "own instants", "[v2][trbdf2][control]") {
    // fix-control-tick-semantics: before, one observer_period served
    // every source, so a 1 kHz and a 4 kHz controller both ticked at
    // 4 kHz. Source j ticks at k*T_j, ceil(span/T_j) times, and
    // sources due at one instant fire in index order.
    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "in", "gnd", 1.0);
    b.add_resistor("R1", "in", "o", 1e3);
    b.add_capacitor("C1", "o", "gnd", 1e-6);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    const Size n_sw = b.graph().num_switches();
    auto sf = [n_sw](Real) { return topology::SwitchStateMask(n_sw); };
    solver::TrBdf2Options o;
    o.t_end = 10e-3;
    o.observer_periods = {1e-3, 0.5e-3, 0.25e-3};
    solver::TrBdf2Stats st;
    std::vector<std::pair<Size, Real>> ticks;
    solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o, sf, {}, std::nullopt, &st, {},
        {}, {}, {}, nullptr,
        [&](Size j, Real t, const Vector&) { ticks.emplace_back(j, t); });
    REQUIRE(st.ctrl_tick_counts == std::vector<Size>{10, 20, 40});
    REQUIRE(st.n_ctrl_ticks == 70);
    std::vector<Size> k(3, 0);
    for (const auto& [j, t] : ticks) {
        const Real want = static_cast<Real>(k[j]++) * o.observer_periods[j];
        REQUIRE(std::abs(t - want) <= 1e-9 * o.observer_periods[j]);
    }
    // At t = 0 and every 1 ms all three are due: index order.
    REQUIRE(ticks[0].first == 0);
    REQUIRE(ticks[1].first == 1);
    REQUIRE(ticks[2].first == 2);
}

TEST_CASE("TR-BDF2: a controller rate with an offset ticks at "
          "t0 + offset + k*T", "[v2][trbdf2][control]") {
    // add-unified-event-queue: per-source offsets on the event queue.
    builder::CircuitBuilder b;
    b.add_voltage_source("V1", "in", "gnd", 1.0);
    b.add_resistor("R1", "in", "o", 1e3);
    b.add_capacitor("C1", "o", "gnd", 1e-6);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    const Size n_sw = b.graph().num_switches();
    auto sf = [n_sw](Real) { return topology::SwitchStateMask(n_sw); };
    solver::TrBdf2Options o;
    o.t_start = 1e-3;
    o.t_end = 11e-3;
    o.observer_periods = {1e-3, 1e-3};
    o.observer_offsets = {0.0, 0.25e-3};
    solver::TrBdf2Stats st;
    std::vector<std::pair<Size, Real>> ticks;
    auto tick = [&](Size j, Real t, const Vector&) {
        ticks.emplace_back(j, t);
    };
    solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o, sf, {}, std::nullopt, &st, {},
        {}, {}, {}, nullptr, tick);
    REQUIRE(st.ctrl_tick_counts == std::vector<Size>{10, 10});
    std::vector<Size> k(2, 0);
    for (const auto& [j, t] : ticks) {
        const Real want = o.t_start + o.observer_offsets[j]
            + static_cast<Real>(k[j]++) * o.observer_periods[j];
        REQUIRE(std::abs(t - want) <= 1e-9 * o.observer_periods[j]);
    }
    o.observer_offsets = {0.0};
    REQUIRE_THROWS_AS(solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o, sf, {}, std::nullopt, &st, {},
        {}, {}, {}, nullptr, tick), std::invalid_argument);
    o.observer_offsets = {0.0, -1e-4};
    REQUIRE_THROWS_AS(solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o, sf, {}, std::nullopt, &st, {},
        {}, {}, {}, nullptr, tick), std::invalid_argument);
}

TEST_CASE("TR-BDF2: a line-inductance bridge does not grind at one "
          "commutation", "[v2][trbdf2][robustness]") {
    // V_th = 0.7 bridge behind 0.1 ohm + 1 mH, 470 µF || 1 kOhm. With
    // every bridge bit OFF the line inductor sees g_off alone (a 1 ps
    // mode); the probes pinned its -1e-7 A and kept deciding all-OFF,
    // the LTE read the mode's kink as error at every h, and h sat at
    // h_min until max_steps (abort at t = 0.176 s of 0.35 s, 73 s of
    // wall time). Now about 5k steps for 0.5 s; the fixed engine gives
    // 321.865 V (dt = 1e-6).
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vac", "s", "bb", 0.0, 325.27, 50.0, 0.0);
    b.add_resistor("Rs", "s", "sr", 0.1);
    b.add_inductor("Lline", "sr", "a", 1e-3);
    b.add_diode("D1", "a", "dc", 1e3, 1e-9, 0.7);
    b.add_diode("D2", "bb", "dc", 1e3, 1e-9, 0.7);
    b.add_diode("D3", "gnd", "bb", 1e3, 1e-9, 0.7);
    b.add_diode("D4", "gnd", "a", 1e3, 1e-9, 0.7);
    b.add_capacitor("C", "dc", "gnd", 470e-6);
    b.add_resistor("RL", "dc", "gnd", 1e3);

    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    const Size n_sw = b.graph().num_switches();
    solver::TrBdf2Options o;
    o.t_end = 0.5;
    o.max_steps = 200'000;
    solver::TrBdf2Stats st;
    solver::SimulationResult res;
    REQUIRE_NOTHROW(res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o,
        [&](Real) { return all_open(n_sw); }, {}, std::nullopt, &st));
    REQUIRE(res.times.back() == Approx(0.5));
    REQUIRE(st.n_accept < 20'000);
    REQUIRE(tail_mean(res, b.node_id_of("dc"), 0.48)
            == Approx(321.865).margin(0.1));
}

// update-trbdf2-post-event-restart: after a gate edge the history's
// derivative slots hold the POST-edge derivative (read from a probe
// under the new mask), not the open-switch value. The run ends on the
// edge, so the final snapshot's history is the refreshed one.
TEST_CASE("TR-BDF2: derivative slots are refreshed after a gate edge",
          "[v2][trbdf2][post-event]") {
    const Real V = 10.0, R = 10.0, r_on = 1e-3, t_close = 1.2345e-4;
    for (const bool refresh : {true, false}) {
        for (const bool inductor : {true, false}) {
            builder::CircuitBuilder b;
            b.add_voltage_source("V", "in", "gnd", V);
            b.add_switch("S", "in", "n1", 1.0 / r_on, 1e-9);
            b.add_resistor("R", "n1", "n2", R);
            if (inductor) {
                b.add_inductor("L", "n2", "gnd", 1e-3);
            } else {
                b.add_capacitor("C", "n2", "gnd", 1e-6);
            }
            pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
            cache.build_lazy(1e-6);
            const Size n_sw = b.graph().num_switches();
            auto sf = [&](Real t) {
                topology::SwitchStateMask m(n_sw);
                m.set(0, t >= t_close);
                return m;
            };
            solver::TrBdf2Options o;
            o.t_end = t_close;
            o.refresh_after_event = refresh;
            solver::TrBdf2Stats st;
            const auto res = solver::run_transient_trbdf2(
                cache, b.graph(), b.pool(), o, sf, {}, std::nullopt,
                &st);
            REQUIRE(st.n_gate_events == 1);
            const auto hist = res.final_snapshot.view(
                device::StateBlock::LinearCompanion);
            REQUIRE(hist.size() == 2);
            const Real v_prev = hist[0], i_prev = hist[1];
            if (inductor) {
                // v_L(te+) = V - (R + r_on)·i(te); 0 V (open switch)
                // without the refresh.
                const Real expect = V - (R + r_on) * i_prev;
                if (refresh) {
                    REQUIRE(v_prev == Approx(expect).epsilon(1e-8));
                    // 2: the edge, and the zero start (no initial_state):
                    // i(0) = 0 through the OFF switch's 1 GOhm leaves
                    // v_L(0+) = V - i (R + 1e9) = V = 10 V, not the init
                    // slot's 0 V, so the start takes the probe's slot.
                    REQUIRE(st.n_derivative_refreshes == 2);
                } else {
                    REQUIRE(std::abs(v_prev) < 1e-6);
                    REQUIRE(st.n_derivative_refreshes == 0);
                }
            } else {
                const Real expect = (V - v_prev) / (R + r_on);
                if (refresh) {
                    REQUIRE(i_prev == Approx(expect).epsilon(1e-8));
                } else {
                    REQUIRE(std::abs(i_prev) < 1e-6);
                }
            }
        }
    }
}

TEST_CASE("TR-BDF2: a failed Newton probe at a gate edge is retried, "
          "never read",
          "[v2][trbdf2][post-event]") {
    // A saturable inductor as a buck's inductor. At a turn-off edge the
    // cascade's probe (a 2e-16 s trap step) puts one ulp of inductor
    // current far above Newton's absolute tolerances (2L/h = 1e12 ohm).
    // Without roundoff acceptance that probe fails; its out_x was the
    // warm start, the cascade decided the diode bits on it, never
    // turned the freewheeling diode on, and one step under all-OFF bits
    // cut the 10 A to zero: v_out 7.68 V against 12.85 V on pwl. The
    // probe now climbs 100x per failure (a longer probe is still a
    // probe), and a cascade with no converged probe refuses by name.
    for (const bool roundoff : {false, true}) {
        builder::CircuitBuilder b;
        b.add_voltage_source("Vin", "vin", "gnd", 24.0);
        b.add_switch("S", "vin", "sw", 1e3, 1e-9);
        b.add_diode("D1", "gnd", "sw", 1e3, 1e-9);
        b.add_saturable_inductor("L", "sw", "out", 100e-6, 20.0, 10e-6);
        b.add_capacitor("C", "out", "gnd", 47e-6);
        b.add_resistor("RL", "out", "gnd", 2.0);
        pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
        cache.build_lazy(1e-6);
        const Size n_sw = b.graph().num_switches();
        const auto k = static_cast<Size>(b.switch_index_of("S"));
        const Real T = 1e-5;
        auto sf = [&](Real t) {
            topology::SwitchStateMask m(n_sw);
            const Real ph = t / T - std::floor(t / T + 1e-9);
            m.set(k, ph < 0.5 - 1e-9);
            return m;
        };
        solver::TrBdf2Options o;
        o.t_end = 20 * T;
        o.rtol = 1e-5;
        o.atol = 1e-8;
        o.newton_roundoff_acceptance = roundoff;
        solver::TrBdf2Stats st;
        const auto res = solver::run_transient_trbdf2(
            cache, b.graph(), b.pool(), o, sf, {}, std::nullopt, &st,
            {}, {}, {}, pwl::make_combined_diode_mosfet_refresh());
        REQUIRE(st.n_gate_events >= 38);
        if (!roundoff) {
            REQUIRE(st.n_probe_newton_retries > 0);
        }
        // engine='pwl' at T/400: 12.8491 V; trbdf2 12.8507 V.
        const Real v = tail_mean(res, b.node_id_of("out"), 10 * T);
        REQUIRE(v == Approx(12.849).epsilon(1e-3));
    }
}

TEST_CASE("TR-BDF2: one SDIRK step after a diode event, filter on or off",
          "[v2][trbdf2]") {
    // Half-wave RL rectifier: linear, so the filtered estimate applies.
    // The variants must agree on the output; the counters prove each
    // path ran (one SDIRK step per diode event).
    builder::CircuitBuilder b;
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0, 50.0, 0.0);
    b.add_diode("D", "ac", "a", 1e3, 1e-9, 0.7);
    b.add_inductor("L", "a", "out", 10e-3);
    b.add_resistor("Rl", "out", "gnd", 10.0);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    const Size n_sw = b.graph().num_switches();
    auto run = [&](Size n_sdirk, bool filter, solver::TrBdf2Stats& st) {
        solver::TrBdf2Options o;
        o.t_end = 60e-3;
        o.sdirk_after_diode_event = n_sdirk;
        o.filter_error_estimate = filter;
        o.sdirk_on_linear_circuits = true;
        const auto res = solver::run_transient_trbdf2(
            cache, b.graph(), b.pool(), o,
            [&](Real) { return all_open(n_sw); }, {}, std::nullopt, &st);
        return tail_mean(res, b.node_id_of("out"), 20e-3);
    };
    solver::TrBdf2Stats s_def, s_off, s_filt;
    const Real v_def = run(solver::TrBdf2Options{}.sdirk_after_diode_event,
                           false, s_def);
    const Real v_off = run(0, false, s_off);
    const Real v_filt = run(1, true, s_filt);
    REQUIRE(s_def.n_diode_events >= 4);
    REQUIRE(s_def.n_sdirk_steps > 0);
    REQUIRE(s_def.n_sdirk_steps <= s_def.n_diode_events);
    REQUIRE(s_def.n_filtered_estimates == 0);
    REQUIRE(s_off.n_sdirk_steps == 0);
    REQUIRE(s_filt.n_filtered_estimates > 0);
    REQUIRE(v_def > 1.0);
    REQUIRE(v_def == Approx(v_off).epsilon(1e-3));
    REQUIRE(v_filt == Approx(v_off).epsilon(1e-3));
    // Default (add-branch-form-ideal-switch DECISIONS 24): a piecewise-linear
    // circuit takes the SDIRK step after a diode turn-off, its estimate
    // filtered, and none after a turn-on.
    solver::TrBdf2Options o;
    o.t_end = 60e-3;
    solver::TrBdf2Stats s_lin;
    (void)solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o,
        [&](Real) { return all_open(n_sw); }, {}, std::nullopt, &s_lin);
    REQUIRE(s_lin.n_sdirk_steps > 0);
    REQUIRE(s_lin.n_sdirk_steps < s_lin.n_diode_events);
    REQUIRE(s_lin.n_filtered_estimates >= s_lin.n_sdirk_steps);

    // Order: every step after the first diode event an SDIRK(2,2) step,
    // the controller switched off (rtol 1) so every step IS the ceiling
    // (306 accepts at 200 us over 60 ms). Halving the ceiling divides the
    // error against a 12.5 us run by 4.09 (measured; 4.07, 4.04, 4.0 down
    // to 25 us): second order. Plain TR-BDF2 on the same grid gives 2.28,
    // the first-order edge restart. The window is interpolated at 20 ms
    // (tail_mean): starting at the first sample past it is an O(h) metric
    // error that alone measured 1.95.
    auto run_h = [&](Real h_max, solver::TrBdf2Stats& st) {
        solver::TrBdf2Options oh;
        oh.t_end = 60e-3;
        oh.h_max = h_max;
        oh.rtol = 1.0;
        oh.atol = 1e-3;
        oh.sdirk_after_diode_event = Size{1} << 40;
        oh.sdirk_on_linear_circuits = true;
        const auto res = solver::run_transient_trbdf2(
            cache, b.graph(), b.pool(), oh,
            [&](Real) { return all_open(n_sw); }, {}, std::nullopt, &st);
        return tail_mean(res, b.node_id_of("out"), 20e-3);
    };
    solver::TrBdf2Stats s1, s2, s_ref;
    const Real v1 = run_h(200e-6, s1);
    const Real v2 = run_h(100e-6, s2);
    const Real v_ref = run_h(12.5e-6, s_ref);
    REQUIRE(s1.n_sdirk_steps > s1.n_accept / 2);
    const Real ratio = std::abs(v1 - v_ref) / std::abs(v2 - v_ref);
    INFO("e(200us) = " << std::abs(v1 - v_ref) << ", e(100us) = "
         << std::abs(v2 - v_ref) << ", ratio " << ratio);
    REQUIRE(ratio > 3.5);
}

TEST_CASE("TR-BDF2: an ON diode's noise floor reads the voltage across it, "
          "not its common mode", "[v2][trbdf2][robustness]") {
    // g_off * (1 + |v|) with the NODE voltages put a diode floating at
    // 10 kV at a 2e-8 V floor (2e-5 A through g_on = 1e3 taken for noise);
    // with v_D it is the 0 V floor plus the round-off of forming v_D.
    topology::Graph g;
    const auto a = g.add_node("a");
    const auto k = g.add_node("k");
    g.add_branch(a, k, topology::BranchKind::Switch);
    pwl::DevicePool pool;
    pool.add_diode(0, /*g_on=*/1e3, /*g_off=*/1e-9, /*V_th=*/0.0);
    pwl::DiodeEventState diodes(g, pool);
    diodes.restore_on_bits({true});
    Vector x0(2), xh(2);
    x0 << 1e-3, 0.0;
    xh << 1e4 + 1e-3, 1e4;
    const Real tol0 = solver::detail_trbdf2::held_tolerance(diodes, 0, x0);
    const Real tolh = solver::detail_trbdf2::held_tolerance(diodes, 0, xh);
    REQUIRE(tol0 < 2e-12);
    REQUIRE(tolh < 1e-9);    // 64 eps * 2e4 = 2.8e-10; it was 2e-8
    REQUIRE(tolh > tol0);
}

TEST_CASE("TR-BDF2: step hooks see every landed gate edge as a commutation",
          "[v2][trbdf2][event_queue]") {
    // add-unified-event-queue: the commutation callback the in-kernel
    // event losses hook on. A PWM-driven RC: every edge the engine lands
    // is one mask change between two accepted steps, reported at the
    // edge with the state before (solved under the old mask) and after.
    builder::CircuitBuilder b;
    b.add_voltage_source("V", "in", "gnd", 5.0);
    b.add_switch("S", "in", "n1", 1e3, 1e-9);
    b.add_resistor("R", "n1", "n2", 1e3);
    b.add_capacitor("C", "n2", "gnd", 1e-6);
    pwl::PwlStateSpaceCache cache{b.graph(), b.pool()};
    cache.build_lazy(1e-6);
    const Size n_sw = b.graph().num_switches();
    const Real T = 1e-4, D = 0.3;
    auto edge = [&](Real t) {   // first edge strictly after t
        const Real k = std::floor(t / T);
        for (Real kk = k; kk <= k + 2; kk += 1) {
            if (kk * T > t) return kk * T;
            if (kk * T + D * T > t) return kk * T + D * T;
        }
        return std::numeric_limits<Real>::infinity();
    };
    auto sf = [&](Real t) {
        topology::SwitchStateMask m(n_sw);
        const Real k = std::floor(t / T);
        m.set(0, t >= k * T && t < k * T + D * T);
        return m;
    };
    solver::TrBdf2Options o;
    o.t_end = 1.05e-3;
    solver::TrBdf2Stats st;
    events::StepHooks hooks;
    std::vector<Real> t_com;
    std::size_t n_acc = 0, bad = 0;
    Real t_last = 0.0;
    hooks.on_accept([&](Real t0, Real t1, const Vector&,
                        const topology::SwitchStateMask&) {
        ++n_acc;
        bad += !(t1 > t0) || t0 < t_last;
        t_last = t1;
    });
    hooks.on_commutation([&](const events::Commutation& c) {
        t_com.push_back(c.t);
        bad += c.before == c.after || c.x_before.size() != c.x_after.size();
    });
    const auto res = solver::run_transient_trbdf2(
        cache, b.graph(), b.pool(), o, sf, {}, std::nullopt, &st, edge,
        {}, {}, {}, nullptr, {}, &hooks);
    CHECK(bad == 0);
    CHECK(n_acc > 0);
    // Edges in (0, t_end): 0.3T, T, 1.3T, ..., 10T, 10.3T = 21; the one
    // at 0 is the initial mask.
    CHECK(st.n_gate_events == 21);
    REQUIRE(t_com.size() == st.n_gate_events);
    for (std::size_t i = 0; i < t_com.size(); ++i) {
        const Real ref = (i % 2 == 0) ? (i / 2) * T + D * T : (i / 2 + 1) * T;
        CHECK(std::abs(t_com[i] - ref) < 1e-12);
    }
    CHECK(hooks.commutations() == t_com.size());
    (void)res;
}
