// Three-phase RL load helper — Catch2 tests.
//
// Verifies that the ``Circuit::add_three_phase_rl_load`` helper produces
// the expected per-phase current magnitudes for both star and delta
// connections, balanced and unbalanced, against analytical impedance.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

constexpr Real kPi = 3.14159265358979323846;

template <typename StatesContainer>
Real sample_at(const std::vector<Real>& t,
               const StatesContainer& states,
               Index node_idx, Real target_time) {
    if (t.empty()) return 0.0;
    if (target_time <= t.front()) return states.front()[node_idx];
    if (target_time >= t.back()) return states.back()[node_idx];
    for (std::size_t i = 1; i < t.size(); ++i) {
        if (t[i] >= target_time) {
            const Real t0 = t[i - 1];
            const Real t1 = t[i];
            const Real alpha = (target_time - t0) / (t1 - t0);
            return states[i - 1][node_idx] * (1.0 - alpha) +
                   states[i][node_idx] * alpha;
        }
    }
    return states.back()[node_idx];
}

template <typename StatesContainer>
Real compute_rms(const std::vector<Real>& t,
                 const StatesContainer& states,
                 Index var_idx,
                 Real t_window_start) {
    Real sum_sq = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i] < t_window_start) continue;
        const Real v = states[i][var_idx];
        sum_sq += v * v;
        ++count;
    }
    if (count == 0) return 0.0;
    return std::sqrt(sum_sq / static_cast<Real>(count));
}

}  // namespace

TEST_CASE("three-phase RL load (star): balanced impedance matches analytical V_LL/Z",
          "[load][three_phase][regression]") {
    Circuit circuit;
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    // 400 V_LL_RMS / 50 Hz source.
    Circuit::ThreePhaseSourceParams src_params{};
    src_params.line_to_line_voltage_rms = 400.0;
    src_params.frequency_hz = 50.0;
    src_params.positive_sequence = true;
    circuit.add_three_phase_source("Vsrc", na, nb, nc, Circuit::ground(),
                                   src_params);

    // RL star load: 30 Ω per phase + 50 mH per phase.
    Circuit::ThreePhaseRLLoadParams load_params{};
    load_params.resistance_per_phase = 30.0;
    load_params.inductance_per_phase = 50e-3;
    load_params.topology = Circuit::ThreePhaseLoadTopology::Star;
    circuit.add_three_phase_rl_load("RL", na, nb, nc, Circuit::ground(),
                                    load_params);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 0.2;          // 10 cycles at 50 Hz — plenty of steady state
    opts.dt = 50e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 50e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    INFO("status: " << static_cast<int>(result.final_status)
                    << " message: " << result.message);
    REQUIRE(result.success);
    REQUIRE(result.states.size() > 100);

    // Analytical per-phase line current (star, line-to-neutral source feeds
    // each Y leg with V_LN = V_LL / √3):
    //   V_LN_rms = 400 / √3 ≈ 230.94 V
    //   |Z| = √(R² + (2π·f·L)²) = √(30² + (2π·50·0.050)²) = √(900 + 246.74)
    //       ≈ 33.87 Ω
    //   I_line_rms ≈ 230.94 / 33.87 ≈ 6.82 A
    constexpr Real v_ln_rms = 400.0 / 1.7320508075688772;
    const Real omega = 2.0 * kPi * 50.0;
    const Real x_l = omega * 50e-3;
    const Real z_mag = std::sqrt(30.0 * 30.0 + x_l * x_l);
    const Real i_line_rms_expected = v_ln_rms / z_mag;

    // The internal A-phase inductor branch reserves the first new branch
    // index after the 3 source branches. With 3 sources and 3 R+L pairs in
    // star, the inductors are branches 3, 4, 5 (0-indexed after sources).
    // We compute RMS over the steady-state window (skip first 50 ms).
    const Index n_nodes = circuit.num_nodes();
    const Index i_la_idx = n_nodes + 3;   // L_A branch current
    const Real i_a_rms = compute_rms(result.time, result.states, i_la_idx, 0.05);

    INFO("Expected I_line_rms = " << i_line_rms_expected
         << " A, measured = " << i_a_rms << " A");
    // 5% tolerance — covers transient settling + numerical step quantization.
    CHECK(i_a_rms == Approx(i_line_rms_expected).epsilon(0.05));
}

TEST_CASE("three-phase RL load (delta): impedance is √3 times star",
          "[load][three_phase][regression]") {
    Circuit circuit;
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    // Same source as star test.
    Circuit::ThreePhaseSourceParams src_params{};
    src_params.line_to_line_voltage_rms = 400.0;
    src_params.frequency_hz = 50.0;
    circuit.add_three_phase_source("Vsrc", na, nb, nc, Circuit::ground(),
                                   src_params);

    // Delta load: 30 Ω + 50 mH per branch.
    Circuit::ThreePhaseRLLoadParams load_params{};
    load_params.resistance_per_phase = 30.0;
    load_params.inductance_per_phase = 50e-3;
    load_params.topology = Circuit::ThreePhaseLoadTopology::Delta;
    circuit.add_three_phase_rl_load("RL", na, nb, nc, Circuit::ground(),
                                    load_params);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 0.2;
    opts.dt = 50e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 50e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    INFO("status: " << static_cast<int>(result.final_status));
    REQUIRE(result.success);

    // Analytical: each delta-branch sees V_LL directly.
    //   I_branch_rms = V_LL / |Z| = 400 / 33.87 ≈ 11.81 A
    //   I_line_rms ≈ √3 · I_branch ≈ 20.46 A (sums of two branch currents)
    const Real omega = 2.0 * kPi * 50.0;
    const Real x_l = omega * 50e-3;
    const Real z_mag = std::sqrt(30.0 * 30.0 + x_l * x_l);
    const Real i_branch_rms_expected = 400.0 / z_mag;

    // Delta branch inductors: L_AB is branch 3 (after 3 source branches).
    const Index n_nodes = circuit.num_nodes();
    const Index i_lab_idx = n_nodes + 3;
    const Real i_branch_rms = compute_rms(result.time, result.states, i_lab_idx, 0.05);

    INFO("Expected I_branch_rms = " << i_branch_rms_expected
         << " A, measured (L_AB) = " << i_branch_rms << " A");
    CHECK(i_branch_rms == Approx(i_branch_rms_expected).epsilon(0.05));
}

TEST_CASE("three-phase RL load: unbalance factor scales phase B and C currents",
          "[load][three_phase][regression]") {
    Circuit circuit;
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    Circuit::ThreePhaseSourceParams src_params{};
    src_params.line_to_line_voltage_rms = 400.0;
    src_params.frequency_hz = 50.0;
    circuit.add_three_phase_source("Vsrc", na, nb, nc, Circuit::ground(),
                                   src_params);

    Circuit::ThreePhaseRLLoadParams load_params{};
    load_params.resistance_per_phase = 30.0;
    load_params.inductance_per_phase = 50e-3;
    load_params.topology = Circuit::ThreePhaseLoadTopology::Star;
    load_params.unbalance_factor = 0.2;   // |Z_b| = 0.8·Z, |Z_c| = 1.2·Z
    circuit.add_three_phase_rl_load("RL", na, nb, nc, Circuit::ground(),
                                    load_params);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 0.2;
    opts.dt = 50e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 50e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    // Branch order: L_A=3, L_B=4, L_C=5 (after 3 source branches).
    const Index n_nodes = circuit.num_nodes();
    const Real i_a_rms = compute_rms(result.time, result.states, n_nodes + 3, 0.05);
    const Real i_b_rms = compute_rms(result.time, result.states, n_nodes + 4, 0.05);
    const Real i_c_rms = compute_rms(result.time, result.states, n_nodes + 5, 0.05);

    INFO("I_a=" << i_a_rms << " A, I_b=" << i_b_rms << " A, I_c=" << i_c_rms << " A");

    // Phase B carries higher current (|Z_b| smaller); C lower (|Z_c| larger).
    CHECK(i_b_rms > i_a_rms);
    CHECK(i_c_rms < i_a_rms);
    // Approximate ratios — scaling is on both R and L so |Z| scales by
    // the same factor, hence I scales by 1/factor.
    CHECK(i_b_rms / i_a_rms == Approx(1.0 / 0.8).epsilon(0.05));
    CHECK(i_c_rms / i_a_rms == Approx(1.0 / 1.2).epsilon(0.05));
}
