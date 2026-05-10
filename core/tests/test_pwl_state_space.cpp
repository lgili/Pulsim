// =============================================================================
// Test: PWL state-space assembly (Phase 2 of refactor-pwl-switching-engine)
// =============================================================================
//
// Validates `Circuit::assemble_state_space(M, N, b, t)`, the topology-bitmask
// signature, and the admissibility predicates that gate the PWL segment path.
//
// We exercise canonical small circuits (RC, RL, voltage source, switch, PWM
// source) and assert the M / N / b matrices match the analytical DAE form
// `M·ẋ + N·x = b(t)`. These tests cover the building block consumed by
// `DefaultSegmentModelService::build_model` to construct Tustin matrices.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/runtime_circuit.hpp"

using namespace pulsim::v1;
using Catch::Approx;

namespace {

// Convenience helper: Approx-equality with a relaxed tolerance suitable for
// floating-point sparse-matrix accumulations.
[[nodiscard]] Approx approx(double v) { return Approx(v).margin(1e-12); }

// Branch index of the most recently registered branch device (V-source,
// inductor, transformer). Add such devices last (after all nodes) for
// stable arithmetic against `num_nodes()`.
[[nodiscard]] Index last_branch_index(const Circuit& ckt) {
    return ckt.num_nodes() + ckt.num_branches() - 1;
}

}  // namespace

// -----------------------------------------------------------------------------
// Topology bitmask
// -----------------------------------------------------------------------------

TEST_CASE("pwl_topology_bitmask: empty circuit returns 0",
          "[pwl_state_space][topology]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    ckt.add_resistor("R1", a, -1, 1.0);
    REQUIRE(ckt.pwl_topology_bitmask() == 0);
    REQUIRE(ckt.pwl_switching_device_count() == 0);
}

TEST_CASE("pwl_topology_bitmask: bit per switching device in registration order",
          "[pwl_state_space][topology]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    auto c = ckt.add_node("c");

    // bit 0 = SW1, bit 1 = SW2, bit 2 = D1.
    ckt.add_switch("SW1", a, b);
    ckt.add_switch("SW2", b, c);
    ckt.add_diode("D1", a, c);

    REQUIRE(ckt.pwl_switching_device_count() == 3);
    REQUIRE(ckt.pwl_topology_bitmask() == 0);

    ckt.set_pwl_state("SW1", true);
    REQUIRE(ckt.pwl_topology_bitmask() == 0b001);

    ckt.set_pwl_state("SW2", true);
    REQUIRE(ckt.pwl_topology_bitmask() == 0b011);

    ckt.set_pwl_state("D1", true);
    REQUIRE(ckt.pwl_topology_bitmask() == 0b111);

    ckt.set_pwl_state("SW1", false);
    REQUIRE(ckt.pwl_topology_bitmask() == 0b110);
}

// -----------------------------------------------------------------------------
// Admissibility predicates
// -----------------------------------------------------------------------------

TEST_CASE("all_switching_devices_in_ideal_mode: empty circuit is trivially eligible",
          "[pwl_state_space][admissibility]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    ckt.add_resistor("R1", a, b, 1.0);
    REQUIRE(ckt.all_switching_devices_in_ideal_mode());
    REQUIRE(ckt.pwl_state_space_supports_all_devices());
}

TEST_CASE("all_switching_devices_in_ideal_mode: Auto resolves with circuit default",
          "[pwl_state_space][admissibility]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    ckt.add_diode("D1", a, b);

    // Default mode is Auto. With circuit_default = Behavioral (the conservative
    // production setting), Auto resolves down to Behavioral and admissibility
    // fails.
    REQUIRE_FALSE(ckt.all_switching_devices_in_ideal_mode(SwitchingMode::Behavioral));
    // With circuit_default = Ideal, Auto resolves up.
    REQUIRE(ckt.all_switching_devices_in_ideal_mode(SwitchingMode::Ideal));
}

// -----------------------------------------------------------------------------
// State-space assembly: RC low-pass
// -----------------------------------------------------------------------------

TEST_CASE("assemble_state_space: RC low-pass produces correct M, N, b",
          "[pwl_state_space][rc]") {
    Circuit ckt;
    auto in = ckt.add_node("in");
    auto out = ckt.add_node("out");

    constexpr Real V_src = 5.0;
    constexpr Real R = 1e3;
    constexpr Real C = 1e-6;

    ckt.add_resistor("R1", in, out, R);
    ckt.add_capacitor("C1", out, -1, C);
    ckt.add_voltage_source("V1", in, -1, V_src);
    const Index v_idx = last_branch_index(ckt);

    SparseMatrix M, N;
    Vector b;
    ckt.assemble_state_space(M, N, b, /*time=*/0.0);

    const Index n = ckt.system_size();  // 2 nodes + 1 V-source branch = 3
    REQUIRE(n == 3);
    REQUIRE(M.rows() == n);
    REQUIRE(N.rows() == n);
    REQUIRE(b.size() == n);

    // M should have only C entries on the `out` node (C1 is grounded).
    REQUIRE(M.coeff(out, out) == approx(C));
    REQUIRE(M.coeff(in, in) == approx(0.0));
    REQUIRE(M.coeff(v_idx, v_idx) == approx(0.0));

    // N should have:
    //   - resistor conductance G = 1/R between in and out
    //   - voltage-source branch coupling (in, br) and (br, in) = +1
    const Real G = 1.0 / R;
    REQUIRE(N.coeff(in, in) == approx(G));
    REQUIRE(N.coeff(in, out) == approx(-G));
    REQUIRE(N.coeff(out, in) == approx(-G));
    REQUIRE(N.coeff(out, out) == approx(G));
    REQUIRE(N.coeff(in, v_idx) == approx(1.0));
    REQUIRE(N.coeff(v_idx, in) == approx(1.0));

    // b should have V_src on the branch row.
    REQUIRE(b[v_idx] == approx(V_src));
    REQUIRE(b[in] == approx(0.0));
    REQUIRE(b[out] == approx(0.0));
}

// -----------------------------------------------------------------------------
// State-space assembly: RL with grounded source
// -----------------------------------------------------------------------------

TEST_CASE("assemble_state_space: RL series produces correct M, N",
          "[pwl_state_space][rl]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b_node = ckt.add_node("b");

    constexpr Real V_src = 12.0;
    constexpr Real R = 5.0;
    constexpr Real L = 1e-3;

    ckt.add_resistor("R1", a, b_node, R);
    ckt.add_inductor("L1", b_node, -1, L);
    const Index l_idx = last_branch_index(ckt);
    ckt.add_voltage_source("V1", a, -1, V_src);
    const Index v_idx = last_branch_index(ckt);

    SparseMatrix M, N;
    Vector b;
    ckt.assemble_state_space(M, N, b, /*time=*/0.0);

    // Inductor branch row: M[L_branch, L_branch] = -L.
    REQUIRE(M.coeff(l_idx, l_idx) == approx(-L));

    // N contributions.
    const Real G = 1.0 / R;
    REQUIRE(N.coeff(a, a) == approx(G));
    REQUIRE(N.coeff(a, b_node) == approx(-G));
    REQUIRE(N.coeff(b_node, b_node) == approx(G));

    REQUIRE(N.coeff(b_node, l_idx) == approx(1.0));
    REQUIRE(N.coeff(l_idx, b_node) == approx(1.0));

    REQUIRE(N.coeff(a, v_idx) == approx(1.0));
    REQUIRE(N.coeff(v_idx, a) == approx(1.0));

    REQUIRE(b[v_idx] == approx(V_src));
}

// -----------------------------------------------------------------------------
// State-space assembly: switch state changes the conductance stamp
// -----------------------------------------------------------------------------

TEST_CASE("assemble_state_space: switch stamp follows committed PWL state",
          "[pwl_state_space][switch]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    constexpr Real g_on = 1e6;
    constexpr Real g_off = 1e-12;

    ckt.add_switch("SW1", a, b, /*closed=*/false, g_on, g_off);

    SparseMatrix M, N;
    Vector bb;

    SECTION("open switch stamps g_off") {
        ckt.set_pwl_state("SW1", false);
        ckt.assemble_state_space(M, N, bb, 0.0);
        REQUIRE(N.coeff(a, a) == approx(g_off));
        REQUIRE(N.coeff(a, b) == approx(-g_off));
    }

    SECTION("closed switch stamps g_on") {
        ckt.set_pwl_state("SW1", true);
        ckt.assemble_state_space(M, N, bb, 0.0);
        REQUIRE(N.coeff(a, a) == approx(g_on));
        REQUIRE(N.coeff(a, b) == approx(-g_on));
    }
}

// -----------------------------------------------------------------------------
// State-space assembly: diode state and stamp
// -----------------------------------------------------------------------------

TEST_CASE("assemble_state_space: diode stamp follows committed pwl_state",
          "[pwl_state_space][diode]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto c = ckt.add_node("c");
    constexpr Real g_on = 1e3;
    constexpr Real g_off = 1e-9;

    ckt.add_diode("D1", a, c, g_on, g_off);

    SparseMatrix M, N;
    Vector bb;

    SECTION("default off-state") {
        ckt.assemble_state_space(M, N, bb, 0.0);
        REQUIRE(N.coeff(a, a) == approx(g_off));
    }

    SECTION("conducting") {
        ckt.set_pwl_state("D1", true);
        ckt.assemble_state_space(M, N, bb, 0.0);
        REQUIRE(N.coeff(a, a) == approx(g_on));
        REQUIRE(N.coeff(a, c) == approx(-g_on));
    }
}

// -----------------------------------------------------------------------------
// State-space assembly: time-varying source evaluated at requested time
// -----------------------------------------------------------------------------

TEST_CASE("assemble_state_space: PWM source evaluates b at requested time",
          "[pwl_state_space][time_varying]") {
    Circuit ckt;
    auto p = ckt.add_node("p");

    PWMParams pwm;
    pwm.frequency = 1e3;  // 1 kHz period = 1 ms
    pwm.duty = 0.5;       // half on
    pwm.v_high = 5.0;
    pwm.v_low = 0.0;
    pwm.phase = 0.0;
    ckt.add_pwm_voltage_source("Vp", p, -1, pwm);
    const Index v_idx = last_branch_index(ckt);

    SparseMatrix M, N;
    Vector b_low, b_high;

    // Mid-on time (0.25 ms): source should be high.
    ckt.assemble_state_space(M, N, b_high, 0.25e-3);
    // Mid-off time (0.75 ms): source should be low.
    ckt.assemble_state_space(M, N, b_low, 0.75e-3);

    REQUIRE(b_high[v_idx] == approx(5.0));
    REQUIRE(b_low[v_idx] == approx(0.0));
}

// -----------------------------------------------------------------------------
// Sanity: M is symmetric for capacitor-only networks
// -----------------------------------------------------------------------------

TEST_CASE("assemble_state_space: M symmetric for capacitor-only networks",
          "[pwl_state_space][symmetry]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto b = ckt.add_node("b");
    auto c = ckt.add_node("c");

    ckt.add_capacitor("C12", a, b, 1e-6);
    ckt.add_capacitor("C23", b, c, 2e-6);

    SparseMatrix M, N;
    Vector bb;
    ckt.assemble_state_space(M, N, bb, 0.0);

    REQUIRE(M.coeff(a, b) == approx(M.coeff(b, a)));
    REQUIRE(M.coeff(b, c) == approx(M.coeff(c, b)));
}

// -----------------------------------------------------------------------------
// Tustin recipe sanity: the discrete matrices satisfy E + A = 2M ; E − A = dt·N
// -----------------------------------------------------------------------------

TEST_CASE("Tustin recipe: E + A = 2·M ; E − A = dt·N (RC low-pass)",
          "[pwl_state_space][tustin]") {
    Circuit ckt;
    auto in = ckt.add_node("in");
    auto out = ckt.add_node("out");

    ckt.add_resistor("R1", in, out, 1e3);
    ckt.add_capacitor("C1", out, -1, 1e-6);
    ckt.add_voltage_source("V1", in, -1, 5.0);

    SparseMatrix M, N;
    Vector b;
    ckt.assemble_state_space(M, N, b, 0.0);

    constexpr Real dt = 1e-6;
    const Real half_dt = 0.5 * dt;
    SparseMatrix E = M + half_dt * N;
    SparseMatrix A = M - half_dt * N;
    E.makeCompressed();
    A.makeCompressed();

    SparseMatrix sum = E + A;          // expected: 2·M
    SparseMatrix diff = E - A;         // expected: dt·N
    sum.makeCompressed();
    diff.makeCompressed();

    REQUIRE(sum.coeff(out, out) == approx(2.0 * M.coeff(out, out)));
    REQUIRE(diff.coeff(in, in) == approx(dt * N.coeff(in, in)));
    REQUIRE(diff.coeff(in, out) == approx(dt * N.coeff(in, out)));
}

// -----------------------------------------------------------------------------
// PWL event scanning (Phase 4)
// -----------------------------------------------------------------------------

TEST_CASE("scan_pwl_commutations: diode flips when voltage reverses",
          "[pwl_state_space][events]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto c = ckt.add_node("c");
    ckt.add_diode("D1", a, c, /*g_on=*/1e3, /*g_off=*/1e-9);
    ckt.set_switching_mode_for_all(SwitchingMode::Ideal);

    Vector x = Vector::Zero(ckt.system_size());

    SECTION("off diode commutes when forward voltage exceeds hysteresis") {
        ckt.set_pwl_state("D1", false);
        x[a] = 1.0;
        x[c] = 0.0;
        const auto events = ckt.scan_pwl_commutations(x);
        REQUIRE(events.size() == 1);
        REQUIRE(events[0].device_name == "D1");
        REQUIRE(events[0].new_state == true);
    }

    SECTION("on diode commutes when current reverses") {
        ckt.set_pwl_state("D1", true);
        x[a] = -1.0;
        x[c] = 0.0;
        const auto events = ckt.scan_pwl_commutations(x);
        REQUIRE(events.size() == 1);
        REQUIRE(events[0].new_state == false);
    }

    SECTION("no commutation when state is consistent") {
        ckt.set_pwl_state("D1", true);
        x[a] = 1.0;
        x[c] = 0.0;
        const auto events = ckt.scan_pwl_commutations(x);
        REQUIRE(events.empty());
    }
}

TEST_CASE("scan_pwl_commutations: skips devices not in Ideal mode",
          "[pwl_state_space][events]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto c = ckt.add_node("c");
    ckt.add_diode("D1", a, c);
    // Default switching_mode = Auto. With circuit_default = Behavioral the
    // device is filtered out; with Ideal it participates.

    Vector x = Vector::Zero(ckt.system_size());
    x[a] = 1.0;
    x[c] = 0.0;

    const auto behavioral_scan = ckt.scan_pwl_commutations(x, SwitchingMode::Behavioral);
    REQUIRE(behavioral_scan.empty());

    const auto ideal_scan = ckt.scan_pwl_commutations(x, SwitchingMode::Ideal);
    REQUIRE(ideal_scan.size() == 1);
}

TEST_CASE("scan_pwl_commutations: VCSwitch follows control-voltage threshold",
          "[pwl_state_space][events]") {
    Circuit ckt;
    auto ctrl = ckt.add_node("ctrl");
    auto t1 = ckt.add_node("t1");
    auto t2 = ckt.add_node("t2");
    ckt.add_vcswitch("S1", ctrl, t1, t2, /*v_threshold=*/2.5);
    ckt.set_switching_mode_for_all(SwitchingMode::Ideal);
    ckt.set_pwl_state("S1", false);

    Vector x = Vector::Zero(ckt.system_size());

    SECTION("control rises above threshold → commute closed") {
        x[ctrl] = 5.0;
        const auto events = ckt.scan_pwl_commutations(x);
        REQUIRE(events.size() == 1);
        REQUIRE(events[0].new_state == true);
    }

    SECTION("control below threshold → no event for off switch") {
        x[ctrl] = 1.0;
        const auto events = ckt.scan_pwl_commutations(x);
        REQUIRE(events.empty());
    }
}

TEST_CASE("commit_pwl_commutations: writes new state and is idempotent",
          "[pwl_state_space][events]") {
    Circuit ckt;
    auto a = ckt.add_node("a");
    auto c = ckt.add_node("c");
    ckt.add_diode("D1", a, c);
    ckt.set_switching_mode_for_all(SwitchingMode::Ideal);
    REQUIRE(ckt.pwl_topology_bitmask() == 0b0);

    std::vector<Circuit::PwlCommutation> events;
    events.push_back({0, "D1", true});

    ckt.commit_pwl_commutations(events);
    REQUIRE(ckt.pwl_topology_bitmask() == 0b1);

    // Repeat — idempotent (writes same state).
    ckt.commit_pwl_commutations(events);
    REQUIRE(ckt.pwl_topology_bitmask() == 0b1);
}

// -----------------------------------------------------------------------------
// End-to-end: simulate one Tustin step on RC and compare to analytical
// -----------------------------------------------------------------------------

TEST_CASE("Tustin step: RC step response matches analytical to 4th order",
          "[pwl_state_space][tustin][analytical]") {
    // Consistency note: the DAE algebraic constraints (`N x = b` for branch
    // rows) require a consistent initial state. In production the Simulator
    // calls dc_operating_point() before transient stepping for this reason.
    // For a purely linear RC at rest with capacitor at 0V, the consistent IC
    // is `V_in = V_src, V_out = 0, i_br = −V_src/R`. Tustin then preserves
    // the algebraic constraint at every subsequent step exactly.
    Circuit ckt;
    auto in = ckt.add_node("in");
    auto out = ckt.add_node("out");

    constexpr Real V_src = 5.0;
    constexpr Real R = 1e3;
    constexpr Real C = 1e-6;
    constexpr Real tau = R * C;

    ckt.add_resistor("R1", in, out, R);
    ckt.add_capacitor("C1", out, -1, C);
    ckt.add_voltage_source("V1", in, -1, V_src);
    const Index v_idx = last_branch_index(ckt);

    SparseMatrix M, N;
    Vector b_now, b_next;
    ckt.assemble_state_space(M, N, b_now, 0.0);
    ckt.assemble_state_space(M, N, b_next, 1e-6);

    constexpr Real dt = 1e-7;  // 100 ns step << τ = 1 ms
    const Real half_dt = 0.5 * dt;
    SparseMatrix E = M + half_dt * N;
    SparseMatrix A = M - half_dt * N;
    E.makeCompressed();

    // Consistent initial state.
    Vector x_now = Vector::Zero(ckt.system_size());
    x_now[in] = V_src;
    x_now[out] = 0.0;
    x_now[v_idx] = -V_src / R;

    // Sanity: x_now satisfies the algebraic branch row `N x = b`.
    const Vector residual_alg = N * x_now - b_now;
    REQUIRE(residual_alg[v_idx] == approx(0.0));

    Vector rhs = A * x_now + half_dt * (b_now + b_next);
    Eigen::SparseLU<SparseMatrix> solver;
    solver.compute(E);
    REQUIRE(solver.info() == Eigen::Success);
    Vector x_next = solver.solve(rhs);

    // Algebraic constraint preserved: V_in stays at V_src.
    REQUIRE(x_next[in] == Approx(V_src).epsilon(1e-9));

    // Analytical V_out(dt) for first-order RC with V_out(0) = 0:
    //   V_out(t) = V_src · (1 − exp(−t/τ))
    const Real expected = V_src * (1.0 - std::exp(-dt / tau));
    REQUIRE(x_next[out] == Approx(expected).epsilon(1e-2));

    // Branch current equals the capacitor-charging current through R.
    // i_R flows from in to out so leaves `in` ⇒ branch current convention
    // gives i_br = −i_R (matches `f[in] = g(v_in − v_out) + i_br = 0`).
    const Real i_R_next = (V_src - x_next[out]) / R;
    REQUIRE(x_next[v_idx] == Approx(-i_R_next).epsilon(1e-2));
}
