// =============================================================================
// Phase 1 of `add-frequency-domain-analysis`: operating-point linearization
// =============================================================================
//
// Pins the contract that `Simulator::linearize_around(x_op, t_op)` returns a
// descriptor state-space form `E·dx/dt = A·x + B·u, y = C·x + D·u` whose
// entries match what an analytical hand-derivation of the circuit's MNA
// equations produces.
//
// Coverage:
//   * RC: `E = diag(C, 0_for_branches)`, `A = G_matrix` reflecting -1/RC on
//     the capacitor's diagonal entry.
//   * RLC: eigenvalues of the reduced `(A_state, E_state)` pencil match the
//     analytical roots `s = -α ± √(α² − ω₀²)` for `α = R/(2L)`,
//     `ω₀ = 1/√(LC)`.
//   * Failure modes: Behavioral-mode device produces a typed
//     `failure_reason` and `ok() == false`.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/core.hpp"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <complex>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

[[nodiscard]] Real entry(const SparseMatrix& M, Index r, Index c) {
    return Eigen::MatrixXd(M)(r, c);
}

[[nodiscard]] Eigen::MatrixXd dense(const SparseMatrix& M) {
    return Eigen::MatrixXd(M);
}

}  // namespace

// -----------------------------------------------------------------------------
// RC: linearization reflects the closed-form trapezoidal companion
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1.3: RC linearization places C on E's capacitor diagonal",
          "[v1][frequency_analysis][phase1][rc]") {
    constexpr Real R = 1e3;
    constexpr Real C_val = 1e-6;
    constexpr Real V_in  = 5.0;

    Circuit ckt;
    auto in  = ckt.add_node("in");
    auto out = ckt.add_node("out");
    ckt.add_resistor("R1", in, out, R);
    ckt.add_capacitor("C1", out, Circuit::ground(), C_val, 0.0);
    ckt.add_voltage_source("V1", in, Circuit::ground(), V_in);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-6;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);

    const auto sys = sim.linearize_around(dc.newton_result.solution, /*t_op=*/0.0);
    REQUIRE(sys.ok());
    CHECK(sys.method == "piecewise_linear_segment");
    CHECK(sys.state_size == ckt.system_size());
    CHECK(sys.input_size == 1);
    CHECK(sys.output_size == ckt.system_size());

    // The state index for the `out` node is where the capacitor's `+C` lives
    // on M's diagonal (and therefore on E's diagonal).
    const auto out_idx = ckt.get_node("out");
    REQUIRE(out_idx >= 0);
    CHECK(entry(sys.E, out_idx, out_idx) == Approx(C_val).margin(1e-15));

    // Resistor stamps `+1/R` on `(in, in)` and `(out, out)` of N (since N
    // holds the conductance / KCL contribution). After the `A = -N`
    // transformation the (out, out) entry should be `-1/R`. The `out` node
    // also has the capacitor contribution but that lives in M (E), not N.
    const auto in_idx = ckt.get_node("in");
    REQUIRE(in_idx >= 0);
    CHECK(entry(sys.A, out_idx, out_idx) == Approx(-1.0 / R).margin(1e-12));
    CHECK(entry(sys.A, out_idx, in_idx)  == Approx( 1.0 / R).margin(1e-12));

    // B is a single column carrying b(t_op). At t = 0 the voltage source's
    // branch equation injects `+V_in` at the source's branch row.
    REQUIRE(sys.B.cols() == 1);
    REQUIRE(sys.B.rows() == ckt.system_size());

    // C should be identity.
    const Eigen::MatrixXd C_dense = dense(sys.C);
    for (Index i = 0; i < ckt.system_size(); ++i) {
        for (Index j = 0; j < ckt.system_size(); ++j) {
            CHECK(C_dense(i, j) == Approx(i == j ? 1.0 : 0.0).margin(1e-15));
        }
    }
    CHECK(sys.D.nonZeros() == 0);
}

// -----------------------------------------------------------------------------
// RLC: reduced pencil eigenvalues match analytical roots
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1.3: series RLC linearization reproduces analytical poles",
          "[v1][frequency_analysis][phase1][rlc]") {
    constexpr Real R = 10.0;     // Ω
    constexpr Real L = 1e-3;     // H
    constexpr Real C_val = 1e-6; // F
    // ω₀ = 1/√(LC), α = R/(2L)
    const Real omega_0 = 1.0 / std::sqrt(L * C_val);
    const Real alpha   = R / (2.0 * L);
    REQUIRE(alpha < omega_0);  // underdamped → complex-conjugate poles

    Circuit ckt;
    auto vin  = ckt.add_node("vin");
    auto mid  = ckt.add_node("mid");
    auto cap  = ckt.add_node("cap");
    ckt.add_voltage_source("V1", vin, Circuit::ground(), 0.0);
    ckt.add_resistor("R1", vin, mid, R);
    ckt.add_inductor("L1", mid, cap, L, 0.0);
    ckt.add_capacitor("C1", cap, Circuit::ground(), C_val, 0.0);

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-6;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    const auto dc = sim.dc_operating_point();
    REQUIRE(dc.success);

    const auto sys = sim.linearize_around(dc.newton_result.solution, /*t_op=*/0.0);
    REQUIRE(sys.ok());

    // Solve the generalized eigenvalue problem `A·v = s·E·v`. Because the
    // MNA system has algebraic constraints (V-source branch equation) E is
    // singular; we expect `state_size − num_algebraic` finite eigenvalues
    // matching the underdamped analytical poles and the rest reported as
    // ±∞ (descriptor-form artifacts).
    const Eigen::MatrixXd E_dense = dense(sys.E);
    const Eigen::MatrixXd A_dense = dense(sys.A);
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
    ges.compute(A_dense, E_dense);
    REQUIRE(ges.info() == Eigen::Success);

    // Extract finite eigenvalues `s = α / β` where β is non-zero.
    const auto& alphas = ges.alphas();
    const auto& betas  = ges.betas();
    std::vector<std::complex<Real>> finite_poles;
    for (Index i = 0; i < alphas.size(); ++i) {
        if (std::abs(betas(i)) > 1e-12) {
            finite_poles.push_back(alphas(i) / betas(i));
        }
    }
    REQUIRE(finite_poles.size() >= 2);

    // Pick the two finite poles with the largest negative real part
    // closest in magnitude — that's the resonant pair.
    auto closest_to_analytical = [&](std::complex<Real> target) {
        std::complex<Real> best = finite_poles.front();
        Real best_dist = std::abs(best - target);
        for (const auto& p : finite_poles) {
            const Real d = std::abs(p - target);
            if (d < best_dist) {
                best = p;
                best_dist = d;
            }
        }
        return best;
    };
    const Real omega_d = std::sqrt(omega_0 * omega_0 - alpha * alpha);
    const std::complex<Real> s_pos{-alpha,  omega_d};
    const std::complex<Real> s_neg{-alpha, -omega_d};
    const auto p_pos = closest_to_analytical(s_pos);
    const auto p_neg = closest_to_analytical(s_neg);

    // Tolerance: the analytical roots are exact for the continuous-time
    // RLC, and our linearization is the same continuous-time MNA pencil
    // (no discretization yet — that's what AC sweep will introduce per
    // frequency). So we expect tight agreement modulo numerical
    // floating-point round-off in the GES decomposition.
    CHECK(p_pos.real() == Approx(s_pos.real()).margin(1e-3 * omega_0));
    CHECK(std::abs(p_pos.imag()) == Approx(omega_d).epsilon(1e-3));
    CHECK(p_neg.real() == Approx(s_neg.real()).margin(1e-3 * omega_0));
    CHECK(std::abs(p_neg.imag()) == Approx(omega_d).epsilon(1e-3));
}

// -----------------------------------------------------------------------------
// Failure: Behavioral-mode device produces a typed failure_reason
// -----------------------------------------------------------------------------

TEST_CASE("Phase 1.4: linearize fails with typed reason on Behavioral devices",
          "[v1][frequency_analysis][phase1][failure]") {
    Circuit ckt;
    auto in   = ckt.add_node("in");
    auto sw_n = ckt.add_node("sw");
    auto out  = ckt.add_node("out");

    ckt.add_voltage_source("Vdc", in, Circuit::ground(), 12.0);
    ckt.add_diode("D1", sw_n, out, /*g_on=*/1e3, /*g_off=*/1e-9);
    ckt.add_resistor("R1", out, Circuit::ground(), 100.0);
    ckt.add_capacitor("C1", out, Circuit::ground(), 1e-6, 0.0);
    // Note: leave the diode in default Auto → Behavioral mode (no
    // `set_switching_mode_for_all`). Linearization should bail out.

    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop  = 1e-6;
    opts.dt     = 1e-7;
    opts.dt_min = 1e-12;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes    = ckt.num_nodes();
    opts.newton_options.num_branches = ckt.num_branches();

    Simulator sim(ckt, opts);
    Vector x_op = Vector::Zero(ckt.system_size());
    const auto sys = sim.linearize_around(x_op, /*t_op=*/0.0);

    CHECK_FALSE(sys.ok());
    CHECK(sys.method == "non_admissible");
    CHECK(sys.failure_reason.find("non_admissible") != std::string::npos);
}
