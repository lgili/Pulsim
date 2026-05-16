// PMSM dynamic device-variant — Catch2 tests.
//
// Validates the full device-variant integration of the permanent-magnet
// synchronous machine: 3 reserved MNA branch rows, 4 internal states
// (i_d, i_q, ω_m, θ_m), trapezoidal companion stamping per phase, and
// forward-Euler mechanical advancement with Park-transformed torque.
//
// Tagged [pmsm_dyn] to keep separation from the steady-state helper tests.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <algorithm>
#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

constexpr Real kPi = 3.14159265358979323846;

SimulationOptions make_opts(Real tstop, Real dt) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = tstop;
    opts.dt = dt;
    opts.dt_min = 1e-9;
    opts.dt_max = dt;
    opts.adaptive_timestep = false;
    opts.enable_bdf_order_control = false;
    return opts;
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

template <typename StatesContainer>
Real peak_after(const std::vector<Real>& t,
                const StatesContainer& states,
                Index var_idx,
                Real t_window_start) {
    Real peak = 0.0;
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (t[i] < t_window_start) continue;
        peak = std::max(peak, std::abs(static_cast<Real>(states[i][var_idx])));
    }
    return peak;
}

}  // namespace

TEST_CASE("PMSM dynamic: zero-magnet degenerates to passive 3-phase RL load",
          "[pmsm_dyn][motor][regression]") {
    // With psi_pm = 0, the PMSM has no back-EMF regardless of rotor speed.
    // It should behave exactly like a 3-phase R+L load. Drive with a
    // balanced source, expect I_line_rms = V_LN_rms / |Z|.
    Circuit circuit;
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    constexpr Real f_grid = 50.0;
    constexpr Real omega_e = 2.0 * kPi * f_grid;
    constexpr Real Rs = 0.5;
    constexpr Real Ls = 2e-3;
    constexpr Real V_LL = 220.0;

    Circuit::ThreePhaseSourceParams src{};
    src.line_to_line_voltage_rms = V_LL;
    src.frequency_hz = f_grid;
    circuit.add_three_phase_source("Vgrid", na, nb, nc, Circuit::ground(), src);

    motors::PmsmParams params{};
    params.Rs = Rs;
    params.Ld = Ls;
    params.Lq = Ls;
    params.psi_pm = 0.0;       // no permanent magnet → no back-EMF
    params.pole_pairs = 2;
    params.J = 1e-3;
    params.b_friction = 1e-4;
    circuit.add_pmsm("M1", na, nb, nc, Circuit::ground(), params);

    auto opts = make_opts(0.2, 50e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    // Analytical expectation for the star-connected branch:
    //   V_LN_rms = V_LL / √3, Z = √(R² + (ωL)²).
    constexpr Real v_ln_rms = V_LL / 1.7320508075688772;
    const Real x_l = omega_e * Ls;
    const Real z_mag = std::sqrt(Rs * Rs + x_l * x_l);
    const Real i_line_expected = v_ln_rms / z_mag;

    // Branch ordering: 3 source branches (Vgrid_a/b/c), then the 3 PMSM
    // phase branches.
    const Index n_nodes = circuit.num_nodes();
    const Real i_la_rms = compute_rms(result.time, result.states,
                                      n_nodes + 3, 0.10);
    INFO("Expected I_line = " << i_line_expected << " A, measured = "
         << i_la_rms << " A");
    CHECK(i_la_rms == Approx(i_line_expected).epsilon(0.07));
}

TEST_CASE("PMSM dynamic: open-circuit back-EMF amplitude = ω_e · ψ_PM",
          "[pmsm_dyn][motor][regression]") {
    // With high-impedance load and a non-zero initial rotor speed, the
    // line-to-neutral voltage approaches the back-EMF e_k. Peak |V_a| ≈
    // ω_e · ψ_PM in the early window before friction has slowed the rotor.
    auto measure_peak = [](Real omega_init) -> Real {
        Circuit circuit;
        const Index na = circuit.add_node("A");
        const Index nb = circuit.add_node("B");
        const Index nc = circuit.add_node("C");

        motors::PmsmParams params{};
        params.Rs = 0.5;
        params.Ld = params.Lq = 2e-3;
        params.psi_pm = 0.1;
        params.pole_pairs = 1;         // ω_e = ω_m for simple bookkeeping
        params.J = 10.0;               // huge inertia → ω stays ≈ constant
        params.b_friction = 0.0;
        params.omega_init = omega_init;
        params.theta_init = 0.0;
        circuit.add_pmsm("M1", na, nb, nc, Circuit::ground(), params);

        // Open-circuit terminals (very high impedance to ground per phase).
        circuit.add_resistor("R_a", na, Circuit::ground(), 1e6);
        circuit.add_resistor("R_b", nb, Circuit::ground(), 1e6);
        circuit.add_resistor("R_c", nc, Circuit::ground(), 1e6);

        auto opts = make_opts(0.04, 25e-6);
        opts.newton_options.num_nodes = circuit.num_nodes();
        opts.newton_options.num_branches = circuit.num_branches();

        Simulator sim(circuit, opts);
        const auto result = sim.run_transient();
        REQUIRE(result.success);

        return peak_after(result.time, result.states, na, 0.005);
    };

    const Real peak_100 = measure_peak(100.0);
    const Real peak_200 = measure_peak(200.0);

    // Expected back-EMF peak: ω_e · ψ_PM (peak phase voltage).
    INFO("Peak |V_a| @ ω=100 = " << peak_100 << " V, @ ω=200 = "
         << peak_200 << " V (expected: 10 and 20 V)");
    CHECK(peak_100 == Approx(10.0).epsilon(0.10));
    CHECK(peak_200 == Approx(20.0).epsilon(0.10));
    // And doubling ω doubles peak.
    CHECK(peak_200 / peak_100 == Approx(2.0).epsilon(0.05));
}

TEST_CASE("PMSM dynamic: spin-down from initial speed under load torque",
          "[pmsm_dyn][motor][regression]") {
    // Set ω_init > 0, no electrical excitation (open phases), positive load
    // torque → mechanical equation reduces to J·dω/dt = -τ_load - b·ω.
    // Expect ω(t) to decrease monotonically toward zero.
    Circuit circuit;
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    motors::PmsmParams params{};
    params.Rs = 0.5;
    params.Ld = params.Lq = 2e-3;
    params.psi_pm = 0.0;        // skip back-EMF for a clean mech-only check
    params.pole_pairs = 2;
    params.J = 1e-3;
    params.b_friction = 5e-4;
    params.omega_init = 100.0;  // start at 100 rad/s
    circuit.add_pmsm("M1", na, nb, nc, Circuit::ground(), params);

    // Loosely couple all phases to ground (large R) to give the solver a
    // closed loop without injecting energy.
    circuit.add_resistor("R_a", na, Circuit::ground(), 1e6);
    circuit.add_resistor("R_b", nb, Circuit::ground(), 1e6);
    circuit.add_resistor("R_c", nc, Circuit::ground(), 1e6);

    // Apply a constant load torque after construction.
    circuit.set_pmsm_tau_load("M1", 0.1);  // 0.1 N·m

    auto opts = make_opts(0.2, 100e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    const Real omega_end = circuit.pmsm_omega("M1");
    INFO("ω_init = 100 rad/s, ω_end (after 0.2s, τ_load=0.1 N·m, "
         "b=5e-4) = " << omega_end);
    // Analytical first-order decay with τ_load constant + linear friction:
    //   J·dω/dt = -τ_load - b·ω
    // At ω=100, dω/dt ≈ -150 rad/s² → over 0.2s expect a ~30 rad/s drop.
    CHECK(omega_end < 100.0);             // monotonic deceleration
    CHECK(100.0 - omega_end > 10.0);      // meaningful drop (>10%)

    // Sanity: motor accessors return non-NaN.
    CHECK(std::isfinite(circuit.pmsm_theta("M1")));
    CHECK(std::isfinite(circuit.pmsm_tau_em("M1")));
}

TEST_CASE("PMSM dynamic: balanced 3-phase drive produces non-trivial torque "
          "and finite dq currents",
          "[pmsm_dyn][motor][regression]") {
    // Open-loop voltage-source drives of a PMSM are notoriously fragile
    // (no current control to hold load angle), so we don't try to verify
    // exact synchronous operation here. What we DO verify is that the
    // device:
    //   1. Solves successfully through a 0.1 s transient.
    //   2. Produces finite, non-NaN dq currents and torque.
    //   3. Has produced some line current — i.e., the stamping is wired
    //      to MNA correctly.
    Circuit circuit;
    const Index na = circuit.add_node("A");
    const Index nb = circuit.add_node("B");
    const Index nc = circuit.add_node("C");

    constexpr Real V_LL = 30.0;
    constexpr Real f_grid = 50.0;

    Circuit::ThreePhaseSourceParams src{};
    src.line_to_line_voltage_rms = V_LL;
    src.frequency_hz = f_grid;
    circuit.add_three_phase_source("Vgrid", na, nb, nc, Circuit::ground(), src);

    motors::PmsmParams params{};
    params.Rs = 0.5;
    params.Ld = params.Lq = 2e-3;
    params.psi_pm = 0.1;
    params.pole_pairs = 2;
    params.J = 5e-3;
    params.b_friction = 1e-3;
    // Pre-spin the rotor to electrical-sync frequency so the back-EMF
    // matches the source frequency from t=0.
    params.omega_init = 2.0 * kPi * f_grid / params.pole_pairs;
    circuit.add_pmsm("M1", na, nb, nc, Circuit::ground(), params);

    auto opts = make_opts(0.1, 50e-6);
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();

    Simulator sim(circuit, opts);
    const auto result = sim.run_transient();
    REQUIRE(result.success);

    const Real i_d = circuit.pmsm_i_d("M1");
    const Real i_q = circuit.pmsm_i_q("M1");
    const Real tau_em = circuit.pmsm_tau_em("M1");
    const Real omega_end = circuit.pmsm_omega("M1");
    INFO("End-of-sim: ω = " << omega_end << " rad/s, i_d = " << i_d
         << " A, i_q = " << i_q << " A, τ_em = " << tau_em << " N·m");

    CHECK(std::isfinite(i_d));
    CHECK(std::isfinite(i_q));
    CHECK(std::isfinite(tau_em));
    CHECK(std::isfinite(omega_end));

    // Each accessor reads back per-phase currents → must be finite.
    CHECK(std::isfinite(circuit.pmsm_i_a("M1")));
    CHECK(std::isfinite(circuit.pmsm_i_b("M1")));
    CHECK(std::isfinite(circuit.pmsm_i_c("M1")));

    // The motor must have moved current through it — otherwise the stamping
    // is silently no-op. Pick a generous lower bound that is still well
    // above numerical noise.
    const Real i_sq_sum = i_d * i_d + i_q * i_q;
    CHECK(i_sq_sum > 1e-6);
}

TEST_CASE("PMSM dynamic: companion-conductance helper matches stamped value",
          "[pmsm_dyn][motor][unit]") {
    // Pure unit test of the device's own arithmetic — no solver involved.
    motors::PmsmParams params{};
    params.Rs = 0.5;
    params.Ld = 2e-3;
    params.Lq = 4e-3;     // salient → L_s mean = 3e-3
    params.psi_pm = 0.1;
    params.pole_pairs = 2;
    params.J = 1e-3;

    PmsmDevice motor(params, "M1");
    const Real dt = 100e-6;
    const Real l_s_expected = 0.5 * (params.Ld + params.Lq);
    const Real g_expected = 1.0 / (params.Rs + 2.0 * l_s_expected / dt);
    CHECK(motor.l_s_effective() == Approx(l_s_expected));
    CHECK(motor.companion_conductance(dt) == Approx(g_expected));
    CHECK(motor.theta_electrical() == Approx(0.0));
    CHECK(motor.omega_electrical() == Approx(0.0));
}
