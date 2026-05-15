// Three-phase voltage source helper smoke test.
//
// Validates the ``Circuit::add_three_phase_source`` helper introduced as part
// of the Phase-28 follow-up (sub-wave C bring-up). The helper decomposes a
// three-phase source into three internal SineVoltageSource branches; this
// suite checks:
//   1. All three line voltages reach the expected V_peak per phase.
//   2. Phase shifts are ±120° in positive- and negative-sequence modes.
//   3. The ``unbalance_factor`` knob scales B / C amplitudes correctly.
//
// The assertions sample voltages at four discrete fractions of a period so
// the test stays compiler-independent and avoids relying on FFT helpers.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

constexpr Real kPi = 3.14159265358979323846;

// Convert line-to-line RMS into per-phase peak (line-to-neutral).
constexpr Real ll_rms_to_phase_peak(Real v_ll_rms) noexcept {
    // V_ph_peak = V_LL_RMS * sqrt(2) / sqrt(3)
    return v_ll_rms * 0.81649658092772603;
}

// Linear-interp helper so we don't depend on dt landing exactly on a sample.
// Templated on the state-vector container type so we work with either Eigen
// vectors (Simulator::run_transient's actual return type) or plain
// std::vector<Real> for unit-level callers.
template <typename StatesContainer>
Real sample_at(const std::vector<Real>& t,
               const StatesContainer& states,
               Index node_idx, Real target_time) {
    if (t.empty()) {
        return 0.0;
    }
    if (target_time <= t.front()) {
        return states.front()[node_idx];
    }
    if (target_time >= t.back()) {
        return states.back()[node_idx];
    }
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

}  // namespace

TEST_CASE("three-phase source: balanced positive sequence reaches V_peak per leg",
          "[grid][three_phase][regression]") {
    Circuit circuit;
    const auto na = circuit.add_node("A");
    const auto nb = circuit.add_node("B");
    const auto nc = circuit.add_node("C");

    Circuit::ThreePhaseSourceParams params{};
    params.line_to_line_voltage_rms = 400.0;  // 400 V_LL_RMS (typical 3φ)
    params.frequency_hz = 50.0;
    params.phase_a_deg = 0.0;
    params.positive_sequence = true;
    params.unbalance_factor = 0.0;

    circuit.add_three_phase_source("Vgrid", na, nb, nc, Circuit::ground(), params);

    // Load on each phase so the MNA system has a well-defined solution.
    circuit.add_resistor("Ra", na, Circuit::ground(), 100.0);
    circuit.add_resistor("Rb", nb, Circuit::ground(), 100.0);
    circuit.add_resistor("Rc", nc, Circuit::ground(), 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 40e-3;       // 2 full periods at 50 Hz
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
    REQUIRE(result.time.size() > 10);

    const Real v_peak_expected = ll_rms_to_phase_peak(params.line_to_line_voltage_rms);

    // Quarter-period after the second zero-crossing for each phase puts the
    // peak at predictable times: A peaks at t = T/4, B at t = T/4 + T/3,
    // C at t = T/4 + 2T/3 (for positive sequence with phase_a=0°).
    const Real T = 1.0 / params.frequency_hz;
    const Real t_peak_a = T / 4.0;
    const Real t_peak_b = T / 4.0 + T / 3.0;
    const Real t_peak_c = T / 4.0 + 2.0 * T / 3.0;

    const Real v_a = sample_at(result.time, result.states, na, t_peak_a);
    const Real v_b = sample_at(result.time, result.states, nb, t_peak_b);
    const Real v_c = sample_at(result.time, result.states, nc, t_peak_c);

    // Allow 2% slack for the linear interpolation between MNA samples.
    CHECK(v_a == Approx(v_peak_expected).margin(0.02 * v_peak_expected));
    CHECK(v_b == Approx(v_peak_expected).margin(0.02 * v_peak_expected));
    CHECK(v_c == Approx(v_peak_expected).margin(0.02 * v_peak_expected));
}

TEST_CASE("three-phase source: negative sequence swaps B/C phase shift",
          "[grid][three_phase][regression]") {
    Circuit circuit_pos;
    Circuit circuit_neg;

    const auto na_pos = circuit_pos.add_node("A");
    const auto nb_pos = circuit_pos.add_node("B");
    const auto nc_pos = circuit_pos.add_node("C");
    const auto na_neg = circuit_neg.add_node("A");
    const auto nb_neg = circuit_neg.add_node("B");
    const auto nc_neg = circuit_neg.add_node("C");

    Circuit::ThreePhaseSourceParams params{};
    params.line_to_line_voltage_rms = 400.0;
    params.frequency_hz = 50.0;
    params.positive_sequence = true;
    circuit_pos.add_three_phase_source("Vp", na_pos, nb_pos, nc_pos,
                                       Circuit::ground(), params);

    params.positive_sequence = false;
    circuit_neg.add_three_phase_source("Vn", na_neg, nb_neg, nc_neg,
                                       Circuit::ground(), params);

    circuit_pos.add_resistor("Ra", na_pos, Circuit::ground(), 100.0);
    circuit_pos.add_resistor("Rb", nb_pos, Circuit::ground(), 100.0);
    circuit_pos.add_resistor("Rc", nc_pos, Circuit::ground(), 100.0);
    circuit_neg.add_resistor("Ra", na_neg, Circuit::ground(), 100.0);
    circuit_neg.add_resistor("Rb", nb_neg, Circuit::ground(), 100.0);
    circuit_neg.add_resistor("Rc", nc_neg, Circuit::ground(), 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 40e-3;
    opts.dt = 50e-6;
    opts.dt_min = 1e-9;
    opts.dt_max = 50e-6;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;

    opts.newton_options.num_nodes = circuit_pos.num_nodes();
    opts.newton_options.num_branches = circuit_pos.num_branches();
    Simulator sim_pos(circuit_pos, opts);
    const auto result_pos = sim_pos.run_transient();
    REQUIRE(result_pos.success);

    opts.newton_options.num_nodes = circuit_neg.num_nodes();
    opts.newton_options.num_branches = circuit_neg.num_branches();
    Simulator sim_neg(circuit_neg, opts);
    const auto result_neg = sim_neg.run_transient();
    REQUIRE(result_neg.success);

    // At t = T/4: positive sequence -> v_B is shifted by -120° from peak A,
    // so v_B(T/4) ≈ V_peak * sin(2π·50·(T/4) − 2π/3) = V_peak * sin(π/2 − 2π/3)
    //              = V_peak * sin(-π/6) = -V_peak / 2.
    // Negative sequence -> v_B is shifted by +120°, so v_B(T/4) = V_peak * sin(π/2 + 2π/3)
    //              = V_peak * sin(7π/6) = -V_peak / 2 too? No — sin(π/2 + 2π/3) = sin(7π/6) = -0.5
    // Hmm both give -0.5. Test v_C instead at T/4:
    //   pos seq: v_C(T/4) = V_peak * sin(π/2 − 4π/3) = sin(-5π/6) = -0.5
    //   neg seq: v_C(T/4) = V_peak * sin(π/2 + 4π/3) = sin(11π/6) = -0.5
    // Use t = 0 instead.
    //   At t=0:
    //     pos seq v_B = V_peak * sin(0 - 2π/3) = sin(-2π/3) = -√3/2 ≈ -0.866
    //     neg seq v_B = V_peak * sin(0 + 2π/3) = +√3/2 ≈ +0.866
    // That's a clear discriminator.
    const Real v_peak = ll_rms_to_phase_peak(400.0);
    const Real expected_pos_b_at_zero = v_peak * std::sin(-2.0 * kPi / 3.0);
    const Real expected_neg_b_at_zero = v_peak * std::sin(+2.0 * kPi / 3.0);

    // t=0 isn't reliable (initial transient); use t=T (one full period).
    const Real T = 1.0 / params.frequency_hz;
    const Real v_b_pos = sample_at(result_pos.time, result_pos.states, nb_pos, T);
    const Real v_b_neg = sample_at(result_neg.time, result_neg.states, nb_neg, T);

    CHECK(v_b_pos == Approx(expected_pos_b_at_zero).margin(0.03 * v_peak));
    CHECK(v_b_neg == Approx(expected_neg_b_at_zero).margin(0.03 * v_peak));
    CHECK(v_b_pos * v_b_neg < 0.0);  // Opposite sign confirms sequence swap
}

TEST_CASE("three-phase source: unbalance factor scales B and C amplitudes",
          "[grid][three_phase][regression]") {
    Circuit circuit;
    const auto na = circuit.add_node("A");
    const auto nb = circuit.add_node("B");
    const auto nc = circuit.add_node("C");

    Circuit::ThreePhaseSourceParams params{};
    params.line_to_line_voltage_rms = 400.0;
    params.frequency_hz = 50.0;
    params.unbalance_factor = 0.1;  // 10% asymmetry

    circuit.add_three_phase_source("Vu", na, nb, nc, Circuit::ground(), params);
    circuit.add_resistor("Ra", na, Circuit::ground(), 100.0);
    circuit.add_resistor("Rb", nb, Circuit::ground(), 100.0);
    circuit.add_resistor("Rc", nc, Circuit::ground(), 100.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 40e-3;
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

    const Real v_peak_nominal = ll_rms_to_phase_peak(params.line_to_line_voltage_rms);
    const Real expected_v_b_peak = v_peak_nominal * (1.0 - 0.1);
    const Real expected_v_c_peak = v_peak_nominal * (1.0 + 0.1);

    const Real T = 1.0 / params.frequency_hz;
    // Each phase's peak time (computed above for positive sequence with phase_a=0).
    const Real v_a_peak = sample_at(result.time, result.states, na, T / 4.0);
    const Real v_b_peak = sample_at(result.time, result.states, nb, T / 4.0 + T / 3.0);
    const Real v_c_peak = sample_at(result.time, result.states, nc, T / 4.0 + 2.0 * T / 3.0);

    CHECK(v_a_peak == Approx(v_peak_nominal).margin(0.03 * v_peak_nominal));
    CHECK(v_b_peak == Approx(expected_v_b_peak).margin(0.03 * v_peak_nominal));
    CHECK(v_c_peak == Approx(expected_v_c_peak).margin(0.03 * v_peak_nominal));
    // Sanity: B is below nominal, C is above.
    CHECK(v_b_peak < v_a_peak);
    CHECK(v_c_peak > v_a_peak);
}
