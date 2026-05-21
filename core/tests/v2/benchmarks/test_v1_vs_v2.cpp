// =============================================================================
// Layer 10 — Head-to-head v1 vs v2 wall-clock benchmarks
// =============================================================================
//
// Builds the SAME circuit in v1 and v2, runs the same
// simulation, measures wall-clock of the time-stepping
// loop, and reports speedup factors.
//
// The architectural claim: v2's PWL state-space cache pre-
// factors the MNA matrix once per switch combination, so
// each timestep is a back-substitution. v1 (and SPICE-
// style solvers) re-factor per timestep — slower for
// repetitive PWM workloads.
//
// V0 ships two scenarios:
//   S1: V_dc + R         (linear baseline, no switches)
//   S2: RC charging      (1 cap, no switches)
//
// More-stressed scenarios (rectifier, PWM) are deferred to
// V1 because v1's diode + event-iteration setup requires
// different bookkeeping than v2's `BranchKind::Switch` +
// `DiodeEventState` and would need a careful schema map.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

// v1
#include "pulsim/v1/core.hpp"

// v2
#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/pwl/cache.hpp"
#include "pulsim/v2/solver/run_transient.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sstream>

namespace {

/// Format a "v1 / v2 / speedup" table row as markdown.
std::string fmt_row(const std::string& scenario,
                     double v1_ms, double v2_ms) {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(28) << scenario
        << " | " << std::right << std::setw(10)
        << std::fixed << std::setprecision(3) << v1_ms
        << " | " << std::setw(10)
        << std::fixed << std::setprecision(3) << v2_ms
        << " | " << std::setw(7) << std::fixed
        << std::setprecision(2) << (v1_ms / v2_ms)
        << "× |";
    return oss.str();
}

/// Measure the wall-clock duration of `fn()` in milliseconds.
template <typename F>
double measure_ms(F&& fn) {
    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();
    fn();
    const auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0)
        .count();
}

}  // namespace

// -----------------------------------------------------------------------------
// S1: V_dc + R (sanity baseline — no switches, no dynamics)
// -----------------------------------------------------------------------------

TEST_CASE("S1 — V_dc + R: v1 vs v2 wall-clock",
          "[v2][layer10][benchmark][s1]") {
    constexpr pulsim::v1::Real V_dc = 10.0;
    constexpr pulsim::v1::Real R    = 100.0;
    constexpr pulsim::v1::Real dt   = 1e-6;
    constexpr pulsim::v1::Real tend = 1e-3;

    // --- v1 ---
    pulsim::v1::Circuit cv1;
    const auto v1_n0 = cv1.add_node("n0");
    cv1.add_voltage_source("Vin", v1_n0,
                             pulsim::v1::Circuit::ground(),
                             V_dc);
    cv1.add_resistor("R1", v1_n0,
                      pulsim::v1::Circuit::ground(), R);

    pulsim::v1::SimulationOptions v1_opts;
    v1_opts.tstart = 0.0;
    v1_opts.tstop  = tend;
    v1_opts.dt     = dt;
    v1_opts.dt_min = dt;
    v1_opts.dt_max = dt;
    v1_opts.adaptive_timestep = false;
    v1_opts.enable_bdf_order_control = false;
    v1_opts.integrator =
        pulsim::v1::Integrator::Trapezoidal;
    v1_opts.linear_solver.order = {
        pulsim::v1::LinearSolverKind::KLU};
    v1_opts.linear_solver.auto_select = false;
    v1_opts.linear_solver.allow_fallback = false;
    v1_opts.newton_options.num_nodes = cv1.num_nodes();
    v1_opts.newton_options.num_branches = cv1.num_branches();

    pulsim::v1::Simulator sim1(cv1, v1_opts);
    pulsim::v1::SimulationResult res1;
    const double v1_ms = measure_ms([&] {
        res1 = sim1.run_transient();
    });
    REQUIRE(res1.success);

    // --- v2 ---
    pulsim::v2::builder::CircuitBuilder cv2;
    cv2.add_voltage_source("Vin", "n0", "gnd", V_dc);
    cv2.add_resistor("R1", "n0", "gnd", R);

    pulsim::v2::pwl::PwlStateSpaceCache cache(
        cv2.graph(), cv2.pool());
    cache.build();   // static path (no dynamic devices)

    pulsim::v2::solver::SimulationOptions v2_opts;
    v2_opts.t_start = 0.0;
    v2_opts.t_end   = tend;
    v2_opts.dt      = dt;

    pulsim::v2::solver::SimulationResult res2;
    const double v2_ms = measure_ms([&] {
        res2 = pulsim::v2::solver::run_transient(
            cache, cv2.graph(), cv2.pool(), v2_opts,
            [](pulsim::v2::Real /*t*/) {
                return pulsim::v2::topology::SwitchStateMask(
                    0);
            });
    });
    REQUIRE(res2.num_steps() > 0);

    // Sanity: both should give v_n0 ≈ V_dc.
    REQUIRE(res1.states.back()[v1_n0] ==
              Catch::Approx(V_dc).margin(0.5));
    REQUIRE(res2.states.back()[0] ==
              Catch::Approx(V_dc).margin(0.5));

    INFO(fmt_row("S1: V_dc + R", v1_ms, v2_ms));
    // Always emit the row in stdout via stderr for easy
    // capture.
    std::cerr << fmt_row("S1: V_dc + R", v1_ms, v2_ms)
              << "\n";

    REQUIRE(v2_ms > 0.0);
}

// -----------------------------------------------------------------------------
// S2: RC charging (1 capacitor, no switches)
// -----------------------------------------------------------------------------

TEST_CASE("S2 — RC charging: v1 vs v2 wall-clock",
          "[v2][layer10][benchmark][s2]") {
    constexpr pulsim::v1::Real V_dc = 5.0;
    constexpr pulsim::v1::Real R    = 1e3;
    constexpr pulsim::v1::Real C    = 1e-6;
    constexpr pulsim::v1::Real dt   = 1e-6;
    constexpr pulsim::v1::Real tend = 5e-3;

    // --- v1 ---
    pulsim::v1::Circuit cv1;
    const auto v1_n0 = cv1.add_node("n0");
    const auto v1_n1 = cv1.add_node("n1");
    cv1.add_voltage_source("Vin", v1_n0,
                             pulsim::v1::Circuit::ground(),
                             V_dc);
    cv1.add_resistor("R1", v1_n0, v1_n1, R);
    cv1.add_capacitor("C1", v1_n1,
                       pulsim::v1::Circuit::ground(), C);

    pulsim::v1::SimulationOptions v1_opts;
    v1_opts.tstart = 0.0;
    v1_opts.tstop  = tend;
    v1_opts.dt     = dt;
    v1_opts.dt_min = dt;
    v1_opts.dt_max = dt;
    v1_opts.adaptive_timestep = false;
    v1_opts.enable_bdf_order_control = false;
    v1_opts.integrator =
        pulsim::v1::Integrator::Trapezoidal;
    v1_opts.linear_solver.order = {
        pulsim::v1::LinearSolverKind::KLU};
    v1_opts.linear_solver.auto_select = false;
    v1_opts.linear_solver.allow_fallback = false;
    v1_opts.newton_options.num_nodes = cv1.num_nodes();
    v1_opts.newton_options.num_branches = cv1.num_branches();

    pulsim::v1::Simulator sim1(cv1, v1_opts);
    pulsim::v1::SimulationResult res1;
    const double v1_ms = measure_ms([&] {
        res1 = sim1.run_transient();
    });
    REQUIRE(res1.success);

    // --- v2 ---
    pulsim::v2::builder::CircuitBuilder cv2;
    cv2.add_voltage_source("Vin", "n0", "gnd", V_dc);
    cv2.add_resistor      ("R1",  "n0", "n1", R);
    cv2.add_capacitor     ("C1",  "n1", "gnd", C);

    pulsim::v2::pwl::PwlStateSpaceCache cache(
        cv2.graph(), cv2.pool());
    cache.build(dt);   // dynamic path (cap present)

    pulsim::v2::solver::SimulationOptions v2_opts;
    v2_opts.t_start = 0.0;
    v2_opts.t_end   = tend;
    v2_opts.dt      = dt;

    pulsim::v2::solver::SimulationResult res2;
    const double v2_ms = measure_ms([&] {
        res2 = pulsim::v2::solver::run_transient(
            cache, cv2.graph(), cv2.pool(), v2_opts,
            [](pulsim::v2::Real /*t*/) {
                return pulsim::v2::topology::SwitchStateMask(
                    0);
            });
    });
    REQUIRE(res2.num_steps() > 0);

    // Sanity: v_n1 should be charged near V_dc (after
    // 5τ ≈ 5 ms with τ = RC = 1 ms).
    const double v1_v_n1 = res1.states.back()[v1_n1];
    const double v2_v_n1 = res2.states.back()[1];
    INFO("v1 v_n1 final = " << v1_v_n1 << ", v2 = " << v2_v_n1);
    REQUIRE(v1_v_n1 ==
              Catch::Approx(V_dc).margin(0.5));
    REQUIRE(v2_v_n1 ==
              Catch::Approx(V_dc).margin(0.5));

    INFO(fmt_row("S2: RC charging", v1_ms, v2_ms));
    std::cerr << fmt_row("S2: RC charging", v1_ms, v2_ms)
              << "\n";

    REQUIRE(v2_ms > 0.0);
}

// -----------------------------------------------------------------------------
// S3: Half-wave rectifier (1 switching diode auto-commutates)
// -----------------------------------------------------------------------------
//
// THIS is the scenario where v2's PWL cache should shine:
// every zero-crossing of the AC source toggles the diode's
// on/off state. v1 refactors the MNA matrix at each
// commutation; v2 just picks the OTHER cached factor.

TEST_CASE("S3 — Half-wave rectifier: v1 vs v2 wall-clock",
          "[v2][layer10][benchmark][s3]") {
    constexpr pulsim::v1::Real V_amp  = 10.0;
    constexpr pulsim::v1::Real f_line = 60.0;
    constexpr pulsim::v1::Real R_load = 10.0;
    constexpr pulsim::v1::Real g_on   = 1e3;
    constexpr pulsim::v1::Real g_off  = 1e-9;

    // 60 Hz × 2 cycles, dt = 50 µs → ~666 steps, with
    // 4 zero-crossings → 4 commutations in v2 (free) vs
    // 4 refactors in v1.
    constexpr pulsim::v1::Real dt   = 5e-5;
    constexpr pulsim::v1::Real tend = 2.0 / f_line;

    // --- v1 ---
    pulsim::v1::Circuit cv1;
    const auto v1_n0 = cv1.add_node("n0");
    const auto v1_n1 = cv1.add_node("n1");
    cv1.add_sine_voltage_source(
        "Vin", v1_n0, pulsim::v1::Circuit::ground(),
        /*amplitude=*/V_amp, /*frequency=*/f_line);
    cv1.add_diode("D1", v1_n0, v1_n1, g_on, g_off);
    cv1.add_resistor("R_L", v1_n1,
                      pulsim::v1::Circuit::ground(), R_load);

    pulsim::v1::SimulationOptions v1_opts;
    v1_opts.tstart = 0.0;
    v1_opts.tstop  = tend;
    v1_opts.dt     = dt;
    v1_opts.dt_min = dt;
    v1_opts.dt_max = dt;
    v1_opts.adaptive_timestep = false;
    v1_opts.enable_bdf_order_control = false;
    v1_opts.integrator =
        pulsim::v1::Integrator::Trapezoidal;
    v1_opts.linear_solver.order = {
        pulsim::v1::LinearSolverKind::KLU};
    v1_opts.linear_solver.auto_select = false;
    v1_opts.linear_solver.allow_fallback = false;
    v1_opts.newton_options.num_nodes = cv1.num_nodes();
    v1_opts.newton_options.num_branches = cv1.num_branches();

    pulsim::v1::Simulator sim1(cv1, v1_opts);
    pulsim::v1::SimulationResult res1;
    const double v1_ms = measure_ms([&] {
        res1 = sim1.run_transient();
    });
    REQUIRE(res1.success);

    // --- v2 ---
    // v2 uses a voltage source with V=0 baseline and the
    // sinusoidal value overlaid via `b_extra_fn`. This is
    // the established v2 pattern from the V8 layer5_v2
    // half-wave rectifier test.
    pulsim::v2::builder::CircuitBuilder cv2;
    cv2.add_voltage_source("Vin", "n0", "gnd", 0.0);
    cv2.add_diode("D1", "n0", "n1", g_on, g_off,
                   /*V_th=*/0.0);
    cv2.add_resistor("R_L", "n1", "gnd", R_load);

    pulsim::v2::pwl::PwlStateSpaceCache cache(
        cv2.graph(), cv2.pool());
    cache.build(dt);

    pulsim::v2::solver::SimulationOptions v2_opts;
    v2_opts.t_start = 0.0;
    v2_opts.t_end   = tend;
    v2_opts.dt      = dt;

    const pulsim::v2::Index src_var =
        cv2.pool().branch_var_id_for_source(0, cv2.graph());
    const auto state_size = static_cast<Eigen::Index>(
        cv2.pool().state_size(cv2.graph()));
    auto b_extra_fn = [src_var, state_size](
        pulsim::v2::Real t) {
        pulsim::v2::Vector b = pulsim::v2::Vector::Zero(
            state_size);
        const pulsim::v2::Real omega =
            2.0 * std::numbers::pi_v<pulsim::v2::Real> *
            f_line;
        const pulsim::v2::Real v_sine =
            V_amp * std::sin(omega * t);
        b[src_var] = -v_sine;
        return b;
    };

    pulsim::v2::solver::SimulationResult res2;
    const double v2_ms = measure_ms([&] {
        res2 = pulsim::v2::solver::run_transient(
            cache, cv2.graph(), cv2.pool(), v2_opts,
            [](pulsim::v2::Real /*t*/) {
                return pulsim::v2::topology::SwitchStateMask(
                    1);   // 1 switch (the diode)
            },
            b_extra_fn);
    });
    REQUIRE(res2.num_steps() > 0);

    INFO(fmt_row("S3: Half-wave rectifier",
                  v1_ms, v2_ms));
    std::cerr << fmt_row("S3: Half-wave rectifier",
                            v1_ms, v2_ms)
              << "\n";

    REQUIRE(v2_ms > 0.0);
}
