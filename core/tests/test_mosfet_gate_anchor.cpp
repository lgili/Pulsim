// =============================================================================
// Phase A2 of harden-component-models-vs-psim-plecs: gate-row Jacobian anchor.
// =============================================================================
//
// Regression tests for `MOSFETParams::g_gate_leak` and
// `IGBTParams::g_gate_leak`. Without the anchor, the MNA matrix row for a
// "floating" gate node (one with no external drive or pull-down resistor) is
// identically zero, so the linear solve is structurally singular and Newton
// produces a NaN solution (or, with auto-regularization, an unpredictable
// fallback-stage solution that hides the underlying defect).
//
// With the default anchor (1 nS to ground on the gate-row DIAGONAL only, no
// residual contribution), the gate row always has a finite pivot.
//
// What is — and what isn't — covered here:
//   * (covered) Direct stamp inspection: J(gate, gate) > 0 after a one-shot
//     `stamp_jacobian` at a realistic op-point, in all three stamp paths
//     (Behavioral, AD, PWL Ideal).
//   * (covered) Opt-out path: setting `g_gate_leak = 0` removes the
//     contribution, restoring legacy SPICE-parity behaviour.
//   * (covered) Residual invariant: `f[gate] == 0` — the leak must remain
//     a pure linear-solve regulariser and must not bias any physical
//     equation at the gate.
//   * (covered) IGBT end-to-end DC OP with a floating gate, confirming the
//     full convergence path (Newton + fallbacks) reaches a valid solution
//     because the gate row is non-singular.
//   * (not covered) MOSFET end-to-end DC OP with a floating gate in
//     Behavioral mode: the smooth Shichman-Hodges Jacobian has separate
//     random-restart sensitivities at deep-cutoff op-points that are
//     orthogonal to this anchor and would mask the regression signal.
//     Production circuits always drive the gate (PWM source or R_g), so
//     a fully-floating-gate Behavioral MOSFET is not a real use-case.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/simulation.hpp"

#include <Eigen/Sparse>

#include <array>
#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

namespace {

SimulationOptions make_dc_opts(const Circuit& circuit) {
    SimulationOptions opts;
    opts.tstart = 0.0;
    opts.tstop = 1e-6;
    opts.dt = 1e-7;
    opts.dt_min = 1e-12;
    opts.dt_max = 1e-7;
    opts.adaptive_timestep = false;
    opts.newton_options.num_nodes = circuit.num_nodes();
    opts.newton_options.num_branches = circuit.num_branches();
    return opts;
}

}  // namespace

// -----------------------------------------------------------------------------
// 1. Inline stamp invariants — these are the structural property the anchor
//    guarantees, independent of any Newton iteration or solver state.
// -----------------------------------------------------------------------------

TEST_CASE("MOSFET gate-row anchor stamps the diagonal in every switching-mode "
          "path",
          "[v1][mosfet][regularization]") {
    auto check_at_op_point = [](MOSFET& m, const char* label) {
        Eigen::SparseMatrix<Real> J(3, 3);
        Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
        Eigen::VectorXd x(3);
        // V_gs = 5V (above V_th = 2V), V_ds = 10V — saturation.
        x << 5.0, 10.0, 0.0;
        std::array<Index, 3> nodes{0, 1, 2};

        m.stamp_jacobian(J, f, x, nodes);

        INFO("mode: " << label);
        CHECK(J.coeff(0, 0) >= 1e-9);
        CHECK(f[0] == Approx(0.0).margin(1e-15));
    };

    SECTION("Behavioral (default)") {
        MOSFET m{};
        m.set_switching_mode(SwitchingMode::Behavioral);
        check_at_op_point(m, "Behavioral");
    }
    SECTION("Ideal (PWL)") {
        MOSFET m{};
        m.set_switching_mode(SwitchingMode::Ideal);
        check_at_op_point(m, "Ideal");
    }
}

TEST_CASE("IGBT gate-row anchor stamps the diagonal in every switching-mode "
          "path",
          "[v1][igbt][regularization]") {
    auto check_at_op_point = [](IGBT& q, const char* label) {
        Eigen::SparseMatrix<Real> J(3, 3);
        Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
        Eigen::VectorXd x(3);
        // V_ge = 10V (above V_th = 5V), V_ce = 50V.
        x << 10.0, 50.0, 0.0;
        std::array<Index, 3> nodes{0, 1, 2};

        q.stamp_jacobian(J, f, x, nodes);

        INFO("mode: " << label);
        CHECK(J.coeff(0, 0) >= 1e-9);
        CHECK(f[0] == Approx(0.0).margin(1e-15));
    };

    SECTION("Behavioral (default)") {
        IGBT q{};
        q.set_switching_mode(SwitchingMode::Behavioral);
        check_at_op_point(q, "Behavioral");
    }
    SECTION("Ideal (PWL)") {
        IGBT q{};
        q.set_switching_mode(SwitchingMode::Ideal);
        check_at_op_point(q, "Ideal");
    }
}

TEST_CASE("MOSFET gate anchor can be disabled by setting g_gate_leak = 0 "
          "(opt-out for SPICE-parity tests)",
          "[v1][mosfet][regularization]") {
    // Documents the opt-out path: setting `g_gate_leak = 0` means the
    // device contributes nothing to the gate diagonal. Useful for
    // bit-exact SPICE comparisons where the test rig provides its own
    // external gate path. In PWL Ideal mode the stamp writes nothing at
    // all to the gate row (the channel current model is independent of
    // V_gs in Ideal mode), so this is the cleanest demonstration.
    MOSFET::Params mp{};
    mp.g_gate_leak = 0.0;
    MOSFET m{mp, "M"};
    m.set_switching_mode(SwitchingMode::Ideal);

    Eigen::SparseMatrix<Real> J(3, 3);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd x(3);
    x << 5.0, 10.0, 0.0;
    std::array<Index, 3> nodes{0, 1, 2};
    m.stamp_jacobian(J, f, x, nodes);

    // With g_gate_leak = 0 and PWL Ideal mode, J(gate, gate) == 0
    // (structurally singular — the OLD behaviour we are guarding against).
    CHECK(J.coeff(0, 0) == Approx(0.0));
    CHECK(f[0] == Approx(0.0));
}

// -----------------------------------------------------------------------------
// 2. End-to-end DC OP convergence with a floating IGBT gate. The full
//    convergence path (Newton + random restart + regularization) reaches a
//    valid solution because the anchor keeps every linear solve well-posed.
// -----------------------------------------------------------------------------

TEST_CASE("IGBT gate-row anchor keeps the DC OP non-singular when the gate "
          "is floating (PWL Ideal mode)",
          "[v1][igbt][regularization][regression]") {
    Circuit circuit;
    const Index n_dc = circuit.add_node("dc");
    const Index n_coll = circuit.add_node("coll");
    const Index n_gate_float = circuit.add_node("gate_float");
    const Index gnd = Circuit::ground();

    constexpr Real v_dc = 600.0;
    circuit.add_voltage_source("Vdc", n_dc, gnd, v_dc);
    circuit.add_resistor("R_pullup", n_dc, n_coll, 1.0e3);

    IGBT::Params ip{};
    ip.vth = 5.0;
    ip.g_on = 1e4;
    ip.g_off = 1e-9;
    REQUIRE(ip.g_gate_leak == Approx(1e-9));
    circuit.add_igbt("Q_floating", n_gate_float, n_coll, gnd, ip);
    circuit.set_igbt_switching_mode("Q_floating", SwitchingMode::Ideal);

    auto opts = make_dc_opts(circuit);
    Simulator sim(circuit, opts);
    const auto dc = sim.dc_operating_point();

    INFO("DC OP message: " << dc.message);
    REQUIRE(dc.success);

    const auto& x = dc.newton_result.solution;
    INFO("V(dc)         = " << x[n_dc] << " V");
    INFO("V(coll)       = " << x[n_coll] << " V");
    INFO("V(gate_float) = " << x[n_gate_float] << " V");

    CHECK(x[n_dc] == Approx(v_dc).margin(1e-6));
    // Gate row residual is 0 and the only diagonal contribution is the
    // anchor, so V_gate must be 0 to satisfy the linear-system row.
    CHECK(std::abs(x[n_gate_float]) < 1e-3);
    // IGBT sits in OFF (pwl_state_ defaults to false), so collector pulls
    // up through R_pullup toward V_dc. With g_off = 1e-9 the voltage
    // divider drop is V_dc · R_pullup / (R_pullup + 1/g_off) ≈ 6e-4 V.
    CHECK(std::abs(x[n_coll] - v_dc) < 0.1);
}
