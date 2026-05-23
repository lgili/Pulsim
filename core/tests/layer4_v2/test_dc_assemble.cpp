// =============================================================================
// Layer 4 V2 — DC operating-point assembly tests
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/pwl/dc_assemble.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

using namespace pulsim;
using namespace pulsim::pwl;
using namespace pulsim::topology;
using Catch::Approx;

TEST_CASE("DC OP: V-R-GND gives v_node = V_dc",
          "[v2][layer4_v2][dc_op]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0});

    SwitchStateMask mask(0);
    const auto x = compute_dc_op(g, pool, mask);
    REQUIRE(x[0] == Approx(5.0).margin(1e-9));
}

TEST_CASE("DC OP: V-R-C-GND has v_C = V_dc (cap fully charged)",
          "[v2][layer4_v2][dc_op]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);    // V_dc
    g.add_branch(0, 1,          BranchKind::PassiveLinear);  // R
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);  // C

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 5.0});
    pool.add_resistor(1, {.G = 1.0});
    pool.add_capacitor(2, {.C = 1e-6});

    SwitchStateMask mask(0);
    const auto x = compute_dc_op(g, pool, mask);
    // v_C node 1: at DC, no current flows through R into cap →
    // v_n1 = v_n0 (no drop across R).
    REQUIRE(x[1] == Approx(5.0).margin(1e-9));
}

TEST_CASE("DC OP: V-R-L-GND gives i_L = V/R",
          "[v2][layer4_v2][dc_op]") {
    Graph g;
    g.add_node("n0");
    g.add_node("n1");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::PassiveLinear);  // R
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);  // L

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_resistor(1, {.G = 1.0});           // R = 1 Ω
    pool.add_inductor(2, {.L = 10e-6});

    SwitchStateMask mask(0);
    const auto x = compute_dc_op(g, pool, mask);
    const Index i_L_idx = pool.branch_var_id_for_inductor(2, g);
    INFO("dc_x = [" << x[0] << ", " << x[1] << ", "
         << x[2] << ", " << x[3] << "]");
    REQUIRE(x[0] == Approx(12.0).margin(1e-9));
    REQUIRE(x[1] == Approx(0.0).margin(1e-6));   // L = short
    REQUIRE(x[i_L_idx] == Approx(12.0).margin(1e-6));
}

TEST_CASE("DC OP: chopper with switch ON pulls v_out to V_dc",
          "[v2][layer4_v2][dc_op]") {
    Graph g;
    g.add_node("vin");
    g.add_node("vout");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, 1,          BranchKind::Switch);
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 12.0});
    pool.add_switch(1, /*g_on=*/1e3, /*g_off=*/1e-9);
    pool.add_resistor(2, {.G = 0.1});

    SwitchStateMask mask_on(1);
    mask_on.set(0, true);
    const auto x_on = compute_dc_op(g, pool, mask_on);
    REQUIRE(x_on[0] == Approx(12.0).margin(1e-9));
    REQUIRE(x_on[1] == Approx(11.998801).epsilon(1e-3));
}

// =============================================================================
// Proposal #3.2 — DC OP with time-varying sources (PWM / Sine / Pulse /
// CurrentSource / VCVS) frozen at t_eval.
// =============================================================================

TEST_CASE("DC OP: SineVoltageSource frozen at t_eval=T/4 → +amplitude",
          "[v2][layer4_v2][dc_op][ergonomics][quickwins]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_sine_voltage_source(
        0, {.v_dc = 0.0, .v_amplitude = 5.0,
            .frequency = 1000.0, .phase = 0.0});
    pool.add_resistor(1, {.G = 1.0});

    SwitchStateMask mask(0);
    // At t = T/4 = 250 µs, sin(2π·f·t) = sin(π/2) = 1 → V = 5.
    const Real t_eval = 250e-6;
    const auto x = compute_dc_op(g, pool, mask, t_eval);
    REQUIRE(x[0] == Approx(5.0).margin(1e-9));
}

TEST_CASE("DC OP: SineVoltageSource at t=0 with v_dc offset",
          "[v2][layer4_v2][dc_op][ergonomics][quickwins]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_sine_voltage_source(
        0, {.v_dc = 3.0, .v_amplitude = 5.0,
            .frequency = 1000.0, .phase = 0.0});
    pool.add_resistor(1, {.G = 1.0});

    SwitchStateMask mask(0);
    // sin(0) = 0 → V = v_dc = 3.
    const auto x = compute_dc_op(g, pool, mask, 0.0);
    REQUIRE(x[0] == Approx(3.0).margin(1e-9));
}

TEST_CASE("DC OP: PWMVoltageSource ON vs OFF at different t_eval",
          "[v2][layer4_v2][dc_op][ergonomics][quickwins]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    // 10 kHz, 50 % duty, v_high=12, v_low=0.
    pool.add_pwm_voltage_source(
        0, {.v_high = 12.0, .v_low = 0.0,
            .frequency = 10e3, .duty = 0.5, .phase = 0.0});
    pool.add_resistor(1, {.G = 1.0});

    SwitchStateMask mask(0);
    // T = 100 µs; t=10 µs (10% of period) → ON → 12 V.
    const auto x_on = compute_dc_op(g, pool, mask, 10e-6);
    REQUIRE(x_on[0] == Approx(12.0).margin(1e-9));
    // t = 60 µs (60% of period) → OFF → 0 V.
    const auto x_off = compute_dc_op(g, pool, mask, 60e-6);
    REQUIRE(x_off[0] == Approx(0.0).margin(1e-9));
}

TEST_CASE("DC OP: PulseVoltageSource before/during/after pulse",
          "[v2][layer4_v2][dc_op][ergonomics][quickwins]") {
    Graph g;
    g.add_node("n0");
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    // Single-shot pulse: 0V → 5V from t=1ms to t=2ms → 0V.
    pool.add_pulse_voltage_source(
        0, {.v_initial = 0.0, .v_pulsed = 5.0,
            .t_start = 1e-3, .pulse_width = 1e-3,
            .period = 0.0});
    pool.add_resistor(1, {.G = 1.0});

    SwitchStateMask mask(0);
    // Before pulse (t = 0.5 ms) → 0 V.
    const auto x_before = compute_dc_op(g, pool, mask, 0.5e-3);
    REQUIRE(x_before[0] == Approx(0.0).margin(1e-9));
    // During pulse (t = 1.5 ms) → 5 V.
    const auto x_during = compute_dc_op(g, pool, mask, 1.5e-3);
    REQUIRE(x_during[0] == Approx(5.0).margin(1e-9));
    // After pulse (t = 2.5 ms, single-shot) → 0 V.
    const auto x_after = compute_dc_op(g, pool, mask, 2.5e-3);
    REQUIRE(x_after[0] == Approx(0.0).margin(1e-9));
}

TEST_CASE("DC OP: CurrentSource across R gives I·R drop",
          "[v2][layer4_v2][dc_op][ergonomics][quickwins]") {
    Graph g;
    g.add_node("n0");
    // I-source 10 mA with `from = n0` (current EXITS source at n0 → flows
    // INTO external R back to gnd). Convention matches v2 EE direction.
    g.add_branch(0, g.ground(), BranchKind::Source);
    g.add_branch(0, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_current_source(0, {.I = 10e-3});
    pool.add_resistor(1, {.G = 1.0 / 1000.0});   // R = 1 kΩ

    SwitchStateMask mask(0);
    const auto x = compute_dc_op(g, pool, mask);
    // KCL at n0: -I_in (current entering ext. circuit) + v_n0/R = 0
    //          → v_n0 = I·R = 10 mA · 1 kΩ = 10 V.
    REQUIRE(x[0] == Approx(10.0).margin(1e-6));
}

TEST_CASE("DC OP: VCVS amplifies input voltage by gain",
          "[v2][layer4_v2][dc_op][ergonomics][quickwins]") {
    Graph g;
    g.add_node("v_in");
    g.add_node("v_out");
    // (0) Input source: 1V on v_in.
    g.add_branch(0, g.ground(), BranchKind::Source);
    // (1) VCVS branch (out_pos = v_out, out_neg = gnd).
    g.add_branch(1, g.ground(), BranchKind::Source);
    // (2) Output load.
    g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

    DevicePool pool;
    pool.add_voltage_source(0, {.V = 1.0});
    pool.add_vcvs(/*branch_id=*/1, /*in_pos_node=*/0,
                   /*in_neg_node=*/g.ground(),
                   {.gain = 10.0});
    pool.add_resistor(2, {.G = 1.0});

    SwitchStateMask mask(0);
    const auto x = compute_dc_op(g, pool, mask);
    // v_out = gain · v_in = 10.
    REQUIRE(x[0] == Approx(1.0).margin(1e-9));
    REQUIRE(x[1] == Approx(10.0).margin(1e-9));
}
