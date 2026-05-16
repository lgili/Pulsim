// PMSM steady-state helper — Catch2 tests.
//
// Verifies that ``Circuit::add_pmsm_steady_state`` reproduces the
// expected per-phase line current when driven by an ideal balanced
// three-phase source. The model is non-salient (L_d = L_q), constant
// rotor speed, fixed back-EMF amplitude.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

constexpr Real kPi = 3.14159265358979323846;

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

TEST_CASE("PMSM steady-state: zero-back-EMF case matches RL load",
          "[pmsm][motor][regression]") {
    // With λ_pm = 0, the PMSM degenerates to a passive 3-phase RL load.
    // This is the simplest sanity check — line current should match the
    // analytical V_LL / (Z·√3) for the star topology.
    Circuit circuit;
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    constexpr Real f_grid = 50.0;
    constexpr Real omega_e = 2.0 * kPi * f_grid;
    constexpr Real R_s = 0.5;
    constexpr Real L_s = 2e-3;

    Circuit::ThreePhaseSourceParams src{};
    src.line_to_line_voltage_rms = 220.0;
    src.frequency_hz = f_grid;
    circuit.add_three_phase_source("Vgrid", na, nb, nc, Circuit::ground(), src);

    Circuit::PmsmSteadyStateParams pmsm{};
    pmsm.R_s = R_s;
    pmsm.L_s = L_s;
    pmsm.lambda_pm = 0.0;       // No back-EMF
    pmsm.omega_electrical = omega_e;
    circuit.add_pmsm_steady_state("M1", na, nb, nc, Circuit::ground(), pmsm);

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

    // Analytical: V_LN_rms = V_LL/√3 = 127 V. Z = √(R² + (ωL)²).
    constexpr Real v_ln_rms = 220.0 / 1.7320508075688772;
    const Real x_l = omega_e * L_s;
    const Real z_mag = std::sqrt(R_s * R_s + x_l * x_l);
    const Real i_line_expected = v_ln_rms / z_mag;

    // Branch ordering after 3 source branches:
    // L_A (phase A), then E_A (sine), L_B, E_B, L_C, E_C — wait, actually
    // sources reserve branches at construction. The order is:
    //   0..2: Vgrid sine sources (A, B, C)
    //   3:    PMSM phase A inductor
    //   4:    PMSM phase A back-EMF sine source
    //   5:    PMSM phase B inductor
    //   6:    PMSM phase B back-EMF
    //   7:    PMSM phase C inductor
    //   8:    PMSM phase C back-EMF
    const Index n_nodes = circuit.num_nodes();
    const Real i_la = compute_rms(result.time, result.states,
                                  n_nodes + 3, 0.05);
    INFO("Expected I_line = " << i_line_expected << " A, measured = "
         << i_la << " A");
    CHECK(i_la == Approx(i_line_expected).epsilon(0.05));
}

TEST_CASE("PMSM steady-state: open-circuit terminals see only back-EMF",
          "[pmsm][motor][regression]") {
    // No external source; just observe the back-EMF at the line terminals
    // when the load is light (high-impedance test resistor for closure).
    Circuit circuit;
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    constexpr Real omega_e = 2.0 * kPi * 50.0;
    constexpr Real lambda_pm = 0.1;

    Circuit::PmsmSteadyStateParams pmsm{};
    pmsm.R_s = 0.5;
    pmsm.L_s = 2e-3;
    pmsm.lambda_pm = lambda_pm;
    pmsm.omega_electrical = omega_e;
    circuit.add_pmsm_steady_state("M1", na, nb, nc, Circuit::ground(), pmsm);

    // High-impedance load to give the solver a definite operating point.
    circuit.add_resistor("R_meas_A", na, Circuit::ground(), 10000.0);
    circuit.add_resistor("R_meas_B", nb, Circuit::ground(), 10000.0);
    circuit.add_resistor("R_meas_C", nc, Circuit::ground(), 10000.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 0.1;
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

    // Expected back-EMF peak: ω_e · λ_pm
    const Real e_peak_expected = omega_e * lambda_pm;
    const Real e_rms_expected = e_peak_expected / std::sqrt(2.0);

    // V at line A: with high-impedance load, terminal voltage ≈ back-EMF.
    const Real v_a_rms = compute_rms(result.time, result.states, na, 0.02);
    INFO("Expected back-EMF rms = " << e_rms_expected
         << " V, measured V_a rms = " << v_a_rms);
    CHECK(v_a_rms == Approx(e_rms_expected).epsilon(0.10));
}

TEST_CASE("PMSM steady-state: omega_e scaling — doubling speed doubles back-EMF",
          "[pmsm][motor][regression]") {
    // Sanity: doubling ω_e doubles the back-EMF amplitude (ω·λ_pm).
    auto measure_emf_peak = [&](Real omega_e) -> Real {
        Circuit circuit;
        const Index na = circuit.add_node("A");
        const Index nb = circuit.add_node("B");
        const Index nc = circuit.add_node("C");

        Circuit::PmsmSteadyStateParams pmsm{};
        pmsm.R_s = 0.5;
        pmsm.L_s = 2e-3;
        pmsm.lambda_pm = 0.1;
        pmsm.omega_electrical = omega_e;
        circuit.add_pmsm_steady_state("M1", na, nb, nc, Circuit::ground(), pmsm);
        circuit.add_resistor("R_a", na, Circuit::ground(), 10000.0);
        circuit.add_resistor("R_b", nb, Circuit::ground(), 10000.0);
        circuit.add_resistor("R_c", nc, Circuit::ground(), 10000.0);

        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 0.1;
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
        Real peak = 0.0;
        for (std::size_t i = 0; i < result.time.size(); ++i) {
            if (result.time[i] < 0.04) continue;
            peak = std::max(peak, std::abs(static_cast<Real>(result.states[i][na])));
        }
        return peak;
    };

    const Real peak_50hz = measure_emf_peak(2.0 * kPi * 50.0);
    const Real peak_100hz = measure_emf_peak(2.0 * kPi * 100.0);
    INFO("EMF peak @ 50Hz = " << peak_50hz << " V, @ 100Hz = " << peak_100hz
         << " V (expected ratio = 2.0)");
    CHECK(peak_100hz / peak_50hz == Approx(2.0).epsilon(0.05));
}

TEST_CASE("DC Motor: quadratic load coefficient drops steady-state speed below linear-friction case",
          "[dc_motor][motor][regression]") {
    // With τ_load_quad_coeff > 0, the motor sees additional load
    // proportional to ω², which means steady-state ω is LOWER than the
    // no-quadratic-load case under the same input voltage.
    auto measure_omega_ss = [](Real quad_coeff) -> Real {
        Circuit circuit;
        const Index n_arm = circuit.add_node("arm");
        motors::DcMotorParams p{};
        p.R_a = 0.5;
        p.L_a = 1e-2;
        p.K_e = 0.05;
        p.K_t = 0.05;
        p.J = 1e-4;
        p.b = 1e-5;
        p.tau_load_quad_coeff = quad_coeff;
        circuit.add_voltage_source("Va", n_arm, Circuit::ground(), 12.0);
        circuit.add_dc_motor("M1", n_arm, Circuit::ground(), p);

        SimulationOptions opts;
        opts.tstart = 0.0;
        opts.tstop = 2.0;     // 2 seconds — settles fully
        opts.dt = 200e-6;
        opts.dt_min = 1e-9;
        opts.dt_max = 200e-6;
        opts.adaptive_timestep = false;
        opts.enable_bdf_order_control = false;
        opts.newton_options.num_nodes = circuit.num_nodes();
        opts.newton_options.num_branches = circuit.num_branches();
        Simulator sim(circuit, opts);
        const auto result = sim.run_transient();
        REQUIRE(result.success);
        return circuit.motor_omega("M1");
    };

    const Real omega_no_quad = measure_omega_ss(0.0);
    const Real omega_with_quad = measure_omega_ss(1e-5);
    INFO("ω_ss without quad = " << omega_no_quad << " rad/s, "
         "with quad (1e-5) = " << omega_with_quad << " rad/s");
    CHECK(omega_with_quad < omega_no_quad);
    // The drop should be meaningful (>5%) at this coefficient and speed.
    CHECK((omega_no_quad - omega_with_quad) / omega_no_quad > 0.05);
}
