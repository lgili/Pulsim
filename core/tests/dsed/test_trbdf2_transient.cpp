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
/// Time-weighted mean of node `n` over [t0, t_end] (trapezoid on the
/// variable grid — a sample mean is biased there).
Real tail_mean(const solver::SimulationResult& res, Index n, Real t0) {
    Real acc = 0.0, span = 0.0;
    for (Size k = 1; k < res.times.size(); ++k) {
        if (res.times[k - 1] < t0) { continue; }
        const Real dt = res.times[k] - res.times[k - 1];
        acc += 0.5 * dt * (res.states[k - 1][n] + res.states[k][n]);
        span += dt;
    }
    return acc / span;
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
            const auto& hist = res.final_snapshot.history;
            REQUIRE(hist.size() == 2);
            const Real v_prev = hist[0], i_prev = hist[1];
            if (inductor) {
                // v_L(te+) = V - (R + r_on)·i(te); 0 V (open switch)
                // without the refresh.
                const Real expect = V - (R + r_on) * i_prev;
                if (refresh) {
                    REQUIRE(v_prev == Approx(expect).epsilon(1e-8));
                    REQUIRE(st.n_derivative_refreshes == 1);
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
