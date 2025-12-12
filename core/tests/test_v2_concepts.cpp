// =============================================================================
// Test: C++23 v2 API Concepts and CRTP Devices
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/core.hpp"

using namespace pulsim::v2;
using Catch::Approx;

TEST_CASE("v2 concepts compilation", "[v2][concepts]") {
    SECTION("Device traits for Resistor") {
        REQUIRE(device_traits<Resistor>::type == DeviceType::Resistor);
        REQUIRE(device_traits<Resistor>::num_pins == 2);
        REQUIRE(device_traits<Resistor>::is_linear == true);
        REQUIRE(device_traits<Resistor>::is_dynamic == false);
    }

    SECTION("Device traits for Capacitor") {
        REQUIRE(device_traits<Capacitor>::type == DeviceType::Capacitor);
        REQUIRE(device_traits<Capacitor>::num_pins == 2);
        REQUIRE(device_traits<Capacitor>::is_linear == true);
        REQUIRE(device_traits<Capacitor>::is_dynamic == true);
    }

    SECTION("Compile-time type queries") {
        REQUIRE(is_linear_device_v<Resistor> == true);
        REQUIRE(is_dynamic_device_v<Resistor> == false);
        REQUIRE(is_linear_device_v<Capacitor> == true);
        REQUIRE(is_dynamic_device_v<Capacitor> == true);
    }
}

TEST_CASE("v2 CRTP Resistor", "[v2][crtp][resistor]") {
    SECTION("Resistor creation") {
        Resistor r(1000.0, "R1");
        REQUIRE(r.resistance() == 1000.0);
        REQUIRE(r.name() == "R1");
    }

    SECTION("Resistor stamping") {
        Resistor r(1000.0);

        // Create a 3x3 sparse matrix (2 nodes + ground)
        SparseMatrix G(2, 2);
        G.setZero();
        Vector b = Vector::Zero(2);

        // Node indices: 0 = node1, 1 = node2 (-1 = ground)
        std::array<Index, 2> nodes = {0, 1};

        r.stamp(G, b, nodes);

        // Check conductance stamps
        G.makeCompressed();
        Real g = 1.0 / 1000.0;  // 0.001 S

        REQUIRE(G.coeff(0, 0) == Approx(g));
        REQUIRE(G.coeff(0, 1) == Approx(-g));
        REQUIRE(G.coeff(1, 0) == Approx(-g));
        REQUIRE(G.coeff(1, 1) == Approx(g));
    }

    SECTION("Resistor to ground") {
        Resistor r(500.0);

        SparseMatrix G(1, 1);
        G.setZero();
        Vector b = Vector::Zero(1);

        // Node 0 to ground (-1)
        std::array<Index, 2> nodes = {0, -1};

        r.stamp(G, b, nodes);
        G.makeCompressed();

        Real g = 1.0 / 500.0;
        REQUIRE(G.coeff(0, 0) == Approx(g));
    }
}

TEST_CASE("v2 CRTP Capacitor", "[v2][crtp][capacitor]") {
    SECTION("Capacitor creation") {
        Capacitor c(1e-6, 0.0, "C1");
        REQUIRE(c.capacitance() == 1e-6);
        REQUIRE(c.name() == "C1");
    }

    SECTION("Capacitor companion model (Trapezoidal)") {
        // 1uF capacitor, dt = 1e-6s, initial voltage = 5V, initial current = 0A
        Capacitor c(1e-6, 5.0, "C1");
        c.set_timestep(1e-6);

        SparseMatrix G(2, 2);
        G.setZero();
        Vector b = Vector::Zero(2);

        std::array<Index, 2> nodes = {0, 1};
        c.stamp(G, b, nodes);
        G.makeCompressed();

        // G_eq = 2C/dt = 2 * 1e-6 / 1e-6 = 2.0 S
        Real g_eq = 2.0;

        REQUIRE(G.coeff(0, 0) == Approx(g_eq));
        REQUIRE(G.coeff(0, 1) == Approx(-g_eq));
        REQUIRE(G.coeff(1, 0) == Approx(-g_eq));
        REQUIRE(G.coeff(1, 1) == Approx(g_eq));

        // I_eq = G_eq * V_prev + I_prev = 2.0 * 5.0 + 0.0 = 10.0 A
        REQUIRE(b[0] == Approx(10.0));
        REQUIRE(b[1] == Approx(-10.0));
    }

    SECTION("Capacitor history update") {
        Capacitor c(1e-6, 0.0);
        c.set_timestep(1e-6);

        // Simulate: set current state
        c.set_current_state(5.0, 0.01);  // V = 5V, I = 10mA

        // Update history
        c.update_history();

        // Check that history was updated
        REQUIRE(c.voltage_prev() == 5.0);
        REQUIRE(c.current_prev() == 0.01);
    }
}

TEST_CASE("v2 Jacobian sparsity pattern", "[v2][sparsity]") {
    SECTION("Resistor pattern") {
        auto pattern = Resistor::jacobian_pattern();
        REQUIRE(pattern.size() == 4);
    }

    SECTION("Capacitor pattern") {
        auto pattern = Capacitor::jacobian_pattern();
        REQUIRE(pattern.size() == 4);
    }
}

TEST_CASE("v2 Result type (std::expected)", "[v2][expected]") {
    SECTION("Success result") {
        Result<int> result = 42;
        REQUIRE(result.has_value());
        REQUIRE(*result == 42);
    }

    SECTION("Error result") {
        Result<int> result = std::unexpected(SolverError::SingularMatrix);
        REQUIRE(!result.has_value());
        REQUIRE(result.error() == SolverError::SingularMatrix);
    }
}

TEST_CASE("v2 Integration method traits", "[v2][integration]") {
    SECTION("Backward Euler") {
        REQUIRE(method_traits<MethodType::BackwardEuler>::order == 1);
        REQUIRE(method_traits<MethodType::BackwardEuler>::is_A_stable == true);
        REQUIRE(method_traits<MethodType::BackwardEuler>::is_L_stable == true);
    }

    SECTION("Trapezoidal") {
        REQUIRE(method_traits<MethodType::Trapezoidal>::order == 2);
        REQUIRE(method_traits<MethodType::Trapezoidal>::is_A_stable == true);
        REQUIRE(method_traits<MethodType::Trapezoidal>::is_L_stable == false);
    }

    SECTION("BDF2") {
        REQUIRE(method_traits<MethodType::BDF2>::order == 2);
        REQUIRE(method_traits<MethodType::BDF2>::is_A_stable == true);
        REQUIRE(method_traits<MethodType::BDF2>::is_L_stable == true);
    }
}

// =============================================================================
// Tests for Inductor CRTP device
// =============================================================================

TEST_CASE("v2 CRTP Inductor", "[v2][crtp][inductor]") {
    SECTION("Inductor creation") {
        Inductor l(1e-3, 0.0, "L1");
        REQUIRE(l.inductance() == 1e-3);
        REQUIRE(l.name() == "L1");
    }

    SECTION("Inductor device traits") {
        REQUIRE(device_traits<Inductor>::type == DeviceType::Inductor);
        REQUIRE(device_traits<Inductor>::num_pins == 2);
        REQUIRE(device_traits<Inductor>::is_linear == true);
        REQUIRE(device_traits<Inductor>::is_dynamic == true);
    }

    SECTION("Inductor companion model (Trapezoidal)") {
        // 1mH inductor, dt = 1e-6s, initial current = 1A
        Inductor l(1e-3, 1.0, "L1");
        l.set_timestep(1e-6);

        SparseMatrix G(2, 2);
        G.setZero();
        Vector b = Vector::Zero(2);

        std::array<Index, 2> nodes = {0, 1};
        l.stamp(G, b, nodes);
        G.makeCompressed();

        // G_eq = dt / (2L) = 1e-6 / (2 * 1e-3) = 0.0005 S
        Real g_eq = 1e-6 / (2.0 * 1e-3);

        REQUIRE(G.coeff(0, 0) == Approx(g_eq));
        REQUIRE(G.coeff(0, 1) == Approx(-g_eq));
        REQUIRE(G.coeff(1, 0) == Approx(-g_eq));
        REQUIRE(G.coeff(1, 1) == Approx(g_eq));

        // V_eq = (2L/dt) * I_prev + V_prev = (2 * 1e-3 / 1e-6) * 1.0 + 0.0 = 2000 V
        // I_eq = g_eq * V_eq = 0.0005 * 2000 = 1.0 A
        Real v_eq = (2.0 * 1e-3 / 1e-6) * 1.0 + 0.0;
        Real i_eq = g_eq * v_eq;
        REQUIRE(b[0] == Approx(i_eq));
        REQUIRE(b[1] == Approx(-i_eq));
    }

    SECTION("Inductor history update") {
        Inductor l(1e-3, 0.0);
        l.set_timestep(1e-6);

        // Simulate: set current state
        l.set_current_state(10.0, 0.5);  // V = 10V, I = 500mA

        // Update history
        l.update_history();

        // Check that history was updated
        REQUIRE(l.voltage_prev() == 10.0);
        REQUIRE(l.current_prev() == 0.5);
    }

    SECTION("Inductor Jacobian pattern") {
        auto pattern = Inductor::jacobian_pattern();
        REQUIRE(pattern.size() == 4);
    }
}

// =============================================================================
// Tests for VoltageSource CRTP device
// =============================================================================

TEST_CASE("v2 CRTP VoltageSource", "[v2][crtp][vsource]") {
    SECTION("VoltageSource creation") {
        VoltageSource vs(12.0, "V1");
        REQUIRE(vs.voltage() == 12.0);
        REQUIRE(vs.name() == "V1");
    }

    SECTION("VoltageSource device traits") {
        REQUIRE(device_traits<VoltageSource>::type == DeviceType::VoltageSource);
        REQUIRE(device_traits<VoltageSource>::num_pins == 2);
        REQUIRE(device_traits<VoltageSource>::num_internal_nodes == 1);  // Branch current
        REQUIRE(device_traits<VoltageSource>::is_linear == true);
        REQUIRE(device_traits<VoltageSource>::is_dynamic == false);
    }

    SECTION("VoltageSource MNA stamping") {
        // 12V voltage source between node 0 and ground
        // MNA extends matrix with branch current variable
        VoltageSource vs(12.0, "V1");
        vs.set_branch_index(2);  // Branch current is variable index 2

        // 3x3 matrix: 2 nodes + 1 branch current
        SparseMatrix G(3, 3);
        G.setZero();
        Vector b = Vector::Zero(3);

        std::array<Index, 2> nodes = {0, 1};
        vs.stamp(G, b, nodes);
        G.makeCompressed();

        // Check MNA stamps
        // G(n+, branch) = +1, G(branch, n+) = +1
        // G(n-, branch) = -1, G(branch, n-) = -1
        REQUIRE(G.coeff(0, 2) == Approx(1.0));
        REQUIRE(G.coeff(2, 0) == Approx(1.0));
        REQUIRE(G.coeff(1, 2) == Approx(-1.0));
        REQUIRE(G.coeff(2, 1) == Approx(-1.0));

        // RHS: voltage value in branch equation row
        REQUIRE(b[2] == Approx(12.0));
    }

    SECTION("VoltageSource to ground") {
        VoltageSource vs(5.0, "V1");
        vs.set_branch_index(1);

        // 2x2 matrix: 1 node + 1 branch current
        SparseMatrix G(2, 2);
        G.setZero();
        Vector b = Vector::Zero(2);

        std::array<Index, 2> nodes = {0, -1};  // Node 0 to ground
        vs.stamp(G, b, nodes);
        G.makeCompressed();

        REQUIRE(G.coeff(0, 1) == Approx(1.0));
        REQUIRE(G.coeff(1, 0) == Approx(1.0));
        REQUIRE(b[1] == Approx(5.0));
    }

    SECTION("VoltageSource Jacobian pattern") {
        auto pattern = VoltageSource::jacobian_pattern();
        REQUIRE(pattern.size() == 5);  // 4 off-diagonal + 1 diagonal placeholder
    }
}

// =============================================================================
// Tests for CurrentSource CRTP device
// =============================================================================

TEST_CASE("v2 CRTP CurrentSource", "[v2][crtp][isource]") {
    SECTION("CurrentSource creation") {
        CurrentSource is(0.1, "I1");
        REQUIRE(is.current() == 0.1);
        REQUIRE(is.name() == "I1");
    }

    SECTION("CurrentSource device traits") {
        REQUIRE(device_traits<CurrentSource>::type == DeviceType::CurrentSource);
        REQUIRE(device_traits<CurrentSource>::num_pins == 2);
        REQUIRE(device_traits<CurrentSource>::num_internal_nodes == 0);
        REQUIRE(device_traits<CurrentSource>::is_linear == true);
        REQUIRE(device_traits<CurrentSource>::is_dynamic == false);
    }

    SECTION("CurrentSource stamping") {
        // 100mA current source from node 0 to node 1
        CurrentSource is(0.1, "I1");

        SparseMatrix G(2, 2);
        G.setZero();
        Vector b = Vector::Zero(2);

        std::array<Index, 2> nodes = {0, 1};
        is.stamp(G, b, nodes);
        G.makeCompressed();

        // Current source doesn't affect G matrix
        REQUIRE(G.nonZeros() == 0);

        // Current source affects RHS only
        // Current flows from n+ to n-: leaves n+, enters n-
        REQUIRE(b[0] == Approx(-0.1));  // Current leaves n+
        REQUIRE(b[1] == Approx(0.1));   // Current enters n-
    }

    SECTION("CurrentSource to ground") {
        CurrentSource is(0.05, "I1");

        SparseMatrix G(1, 1);
        G.setZero();
        Vector b = Vector::Zero(1);

        std::array<Index, 2> nodes = {0, -1};  // From node 0 to ground
        is.stamp(G, b, nodes);
        G.makeCompressed();

        REQUIRE(G.nonZeros() == 0);
        REQUIRE(b[0] == Approx(-0.05));
    }

    SECTION("CurrentSource Jacobian pattern") {
        auto pattern = CurrentSource::jacobian_pattern();
        REQUIRE(pattern.size() == 0);  // No matrix contribution
    }
}

// =============================================================================
// Tests for IdealDiode CRTP device
// =============================================================================

TEST_CASE("v2 CRTP IdealDiode", "[v2][crtp][diode]") {
    SECTION("IdealDiode creation") {
        IdealDiode d(1e3, 1e-9, "D1");
        REQUIRE(d.name() == "D1");
    }

    SECTION("IdealDiode device traits") {
        REQUIRE(device_traits<IdealDiode>::type == DeviceType::Diode);
        REQUIRE(device_traits<IdealDiode>::num_pins == 2);
        REQUIRE(device_traits<IdealDiode>::is_linear == false);  // Nonlinear!
        REQUIRE(device_traits<IdealDiode>::is_dynamic == false);
    }

    SECTION("IdealDiode forward bias (conducting)") {
        IdealDiode d(1e3, 1e-9);  // g_on = 1kS, g_off = 1nS

        SparseMatrix J(2, 2);
        J.setZero();
        Vector f = Vector::Zero(2);
        Vector x = Vector::Zero(2);

        // Forward bias: anode (node 0) at 5V, cathode (node 1) at 0V
        x[0] = 5.0;
        x[1] = 0.0;

        std::array<Index, 2> nodes = {0, 1};
        d.stamp_jacobian(J, f, x, nodes);
        J.makeCompressed();

        // Should be conducting with g_on = 1e3
        REQUIRE(d.is_conducting() == true);
        REQUIRE(J.coeff(0, 0) == Approx(1e3));
        REQUIRE(J.coeff(0, 1) == Approx(-1e3));
        REQUIRE(J.coeff(1, 0) == Approx(-1e3));
        REQUIRE(J.coeff(1, 1) == Approx(1e3));

        // Current = g * V = 1e3 * 5 = 5000A
        REQUIRE(f[0] == Approx(5000.0));
        REQUIRE(f[1] == Approx(-5000.0));
    }

    SECTION("IdealDiode reverse bias (blocking)") {
        IdealDiode d(1e3, 1e-9);

        SparseMatrix J(2, 2);
        J.setZero();
        Vector f = Vector::Zero(2);
        Vector x = Vector::Zero(2);

        // Reverse bias: anode (node 0) at 0V, cathode (node 1) at 5V
        x[0] = 0.0;
        x[1] = 5.0;

        std::array<Index, 2> nodes = {0, 1};
        d.stamp_jacobian(J, f, x, nodes);
        J.makeCompressed();

        // Should be blocking with g_off = 1e-9
        REQUIRE(d.is_conducting() == false);
        REQUIRE(J.coeff(0, 0) == Approx(1e-9));
        REQUIRE(J.coeff(0, 1) == Approx(-1e-9));
    }

    SECTION("IdealDiode Jacobian pattern") {
        auto pattern = IdealDiode::jacobian_pattern();
        REQUIRE(pattern.size() == 4);
    }
}

// =============================================================================
// Tests for IdealSwitch CRTP device
// =============================================================================

TEST_CASE("v2 CRTP IdealSwitch", "[v2][crtp][switch]") {
    SECTION("IdealSwitch creation") {
        IdealSwitch sw(1e6, 1e-12, true, "S1");
        REQUIRE(sw.name() == "S1");
        REQUIRE(sw.is_closed() == true);
    }

    SECTION("IdealSwitch device traits") {
        REQUIRE(device_traits<IdealSwitch>::type == DeviceType::Switch);
        REQUIRE(device_traits<IdealSwitch>::num_pins == 2);
        REQUIRE(device_traits<IdealSwitch>::is_linear == true);  // Piecewise linear
        REQUIRE(device_traits<IdealSwitch>::is_dynamic == false);
    }

    SECTION("IdealSwitch closed state") {
        IdealSwitch sw(1e6, 1e-12, true);  // g_on = 1MS, g_off = 1pS, closed

        SparseMatrix G(2, 2);
        G.setZero();
        Vector b = Vector::Zero(2);

        std::array<Index, 2> nodes = {0, 1};
        sw.stamp(G, b, nodes);
        G.makeCompressed();

        // Closed switch: g = g_on = 1e6
        REQUIRE(G.coeff(0, 0) == Approx(1e6));
        REQUIRE(G.coeff(0, 1) == Approx(-1e6));
        REQUIRE(G.coeff(1, 0) == Approx(-1e6));
        REQUIRE(G.coeff(1, 1) == Approx(1e6));
    }

    SECTION("IdealSwitch open state") {
        IdealSwitch sw(1e6, 1e-12, false);  // g_on = 1MS, g_off = 1pS, open

        SparseMatrix G(2, 2);
        G.setZero();
        Vector b = Vector::Zero(2);

        std::array<Index, 2> nodes = {0, 1};
        sw.stamp(G, b, nodes);
        G.makeCompressed();

        // Open switch: g = g_off = 1e-12
        REQUIRE(G.coeff(0, 0) == Approx(1e-12));
        REQUIRE(G.coeff(0, 1) == Approx(-1e-12));
    }

    SECTION("IdealSwitch state transitions") {
        IdealSwitch sw(1e6, 1e-12, false);

        REQUIRE(sw.is_closed() == false);

        sw.close();
        REQUIRE(sw.is_closed() == true);

        sw.open();
        REQUIRE(sw.is_closed() == false);

        sw.set_state(true);
        REQUIRE(sw.is_closed() == true);
    }

    SECTION("IdealSwitch Jacobian pattern") {
        auto pattern = IdealSwitch::jacobian_pattern();
        REQUIRE(pattern.size() == 4);
    }
}

// =============================================================================
// Static assertions for all CRTP devices
// =============================================================================

// =============================================================================
// Tests for MOSFET CRTP device
// =============================================================================

TEST_CASE("v2 CRTP MOSFET", "[v2][crtp][mosfet]") {
    SECTION("MOSFET creation") {
        MOSFET m(2.0, 0.1, true, "M1");
        REQUIRE(m.name() == "M1");
        REQUIRE(m.params().vth == 2.0);
        REQUIRE(m.params().kp == 0.1);
        REQUIRE(m.params().is_nmos == true);
    }

    SECTION("MOSFET device traits") {
        REQUIRE(device_traits<MOSFET>::type == DeviceType::MOSFET);
        REQUIRE(device_traits<MOSFET>::num_pins == 3);
        REQUIRE(device_traits<MOSFET>::is_linear == false);  // Nonlinear
        REQUIRE(device_traits<MOSFET>::is_dynamic == false);
        REQUIRE(device_traits<MOSFET>::has_thermal_model == true);
    }

    SECTION("MOSFET cutoff region") {
        MOSFET::Params params{.vth = 2.0, .kp = 0.1, .lambda = 0.01, .g_off = 1e-12, .is_nmos = true};
        MOSFET m(params, "M1");

        SparseMatrix J(3, 3);
        J.setZero();
        Vector f = Vector::Zero(3);
        Vector x = Vector::Zero(3);

        // Gate = 0V, Drain = 5V, Source = 0V -> Vgs = 0 < Vth = 2 -> cutoff
        x[0] = 0.0;  // Gate
        x[1] = 5.0;  // Drain
        x[2] = 0.0;  // Source

        std::array<Index, 3> nodes = {0, 1, 2};
        m.stamp_jacobian(J, f, x, nodes);
        J.makeCompressed();

        // In cutoff, should only have very small g_off conductance
        REQUIRE(J.coeff(1, 1) == Approx(1e-12).margin(1e-15));
    }

    SECTION("MOSFET saturation region") {
        MOSFET::Params params{.vth = 2.0, .kp = 0.1, .lambda = 0.0, .g_off = 1e-12, .is_nmos = true};
        MOSFET m(params, "M1");

        SparseMatrix J(3, 3);
        J.setZero();
        Vector f = Vector::Zero(3);
        Vector x = Vector::Zero(3);

        // Gate = 5V, Drain = 10V, Source = 0V
        // Vgs = 5 > Vth = 2, Vds = 10 > Vgs - Vth = 3 -> saturation
        x[0] = 5.0;  // Gate
        x[1] = 10.0; // Drain
        x[2] = 0.0;  // Source

        std::array<Index, 3> nodes = {0, 1, 2};
        m.stamp_jacobian(J, f, x, nodes);
        J.makeCompressed();

        // Id_sat = 0.5 * kp * (Vgs - Vth)^2 = 0.5 * 0.1 * 9 = 0.45A
        // gm = kp * (Vgs - Vth) = 0.1 * 3 = 0.3 S
        REQUIRE(J.coeff(1, 0) == Approx(0.3).margin(0.01));  // gm at drain-gate
    }

    SECTION("MOSFET Jacobian pattern") {
        auto pattern = MOSFET::jacobian_pattern();
        REQUIRE(pattern.size() == 9);  // 3x3 matrix
    }
}

// =============================================================================
// Tests for IGBT CRTP device
// =============================================================================

TEST_CASE("v2 CRTP IGBT", "[v2][crtp][igbt]") {
    SECTION("IGBT creation") {
        IGBT ig(5.0, 1e4, "Q1");
        REQUIRE(ig.name() == "Q1");
        REQUIRE(ig.params().vth == 5.0);
        REQUIRE(ig.params().g_on == 1e4);
    }

    SECTION("IGBT device traits") {
        REQUIRE(device_traits<IGBT>::type == DeviceType::IGBT);
        REQUIRE(device_traits<IGBT>::num_pins == 3);
        REQUIRE(device_traits<IGBT>::is_linear == false);  // Nonlinear
        REQUIRE(device_traits<IGBT>::is_dynamic == false);
        REQUIRE(device_traits<IGBT>::has_thermal_model == true);
    }

    SECTION("IGBT off state") {
        IGBT::Params params{.vth = 5.0, .g_on = 1e4, .g_off = 1e-12, .v_ce_sat = 1.5};
        IGBT ig(params, "Q1");

        SparseMatrix J(3, 3);
        J.setZero();
        Vector f = Vector::Zero(3);
        Vector x = Vector::Zero(3);

        // Gate = 0V, Collector = 100V, Emitter = 0V -> Vge = 0 < Vth = 5 -> off
        x[0] = 0.0;   // Gate
        x[1] = 100.0; // Collector
        x[2] = 0.0;   // Emitter

        std::array<Index, 3> nodes = {0, 1, 2};
        ig.stamp_jacobian(J, f, x, nodes);
        J.makeCompressed();

        REQUIRE(ig.is_conducting() == false);
        REQUIRE(J.coeff(1, 1) == Approx(1e-12).margin(1e-15));
    }

    SECTION("IGBT on state") {
        IGBT::Params params{.vth = 5.0, .g_on = 1e4, .g_off = 1e-12, .v_ce_sat = 1.5};
        IGBT ig(params, "Q1");

        SparseMatrix J(3, 3);
        J.setZero();
        Vector f = Vector::Zero(3);
        Vector x = Vector::Zero(3);

        // Gate = 15V, Collector = 10V, Emitter = 0V -> Vge = 15 > Vth = 5 -> on
        x[0] = 15.0;  // Gate
        x[1] = 10.0;  // Collector
        x[2] = 0.0;   // Emitter

        std::array<Index, 3> nodes = {0, 1, 2};
        ig.stamp_jacobian(J, f, x, nodes);
        J.makeCompressed();

        REQUIRE(ig.is_conducting() == true);
        REQUIRE(J.coeff(1, 1) == Approx(1e4).margin(1.0));
    }

    SECTION("IGBT Jacobian pattern") {
        auto pattern = IGBT::jacobian_pattern();
        REQUIRE(pattern.size() == 9);  // 3x3 matrix
    }
}

// =============================================================================
// Tests for Transformer CRTP device
// =============================================================================

TEST_CASE("v2 CRTP Transformer", "[v2][crtp][transformer]") {
    SECTION("Transformer creation") {
        Transformer tr(2.0, "T1");  // 2:1 turns ratio
        REQUIRE(tr.name() == "T1");
        REQUIRE(tr.turns_ratio() == 2.0);
    }

    SECTION("Transformer device traits") {
        REQUIRE(device_traits<Transformer>::type == DeviceType::Transformer);
        REQUIRE(device_traits<Transformer>::num_pins == 4);
        REQUIRE(device_traits<Transformer>::num_internal_nodes == 2);  // Two branch currents
        REQUIRE(device_traits<Transformer>::is_linear == true);
        REQUIRE(device_traits<Transformer>::is_dynamic == false);
    }

    SECTION("Transformer MNA stamping") {
        Transformer tr(2.0, "T1");  // 2:1 turns ratio
        tr.set_branch_indices(4, 5);  // Primary and secondary branch currents

        // 6x6 matrix: 4 nodes + 2 branch currents
        SparseMatrix G(6, 6);
        G.setZero();
        Vector b = Vector::Zero(6);

        std::array<Index, 4> nodes = {0, 1, 2, 3};  // P+, P-, S+, S-
        tr.stamp(G, b, nodes);
        G.makeCompressed();

        // Check voltage source stamps for primary winding
        REQUIRE(G.coeff(0, 4) == Approx(1.0));  // P+ to branch_p
        REQUIRE(G.coeff(4, 0) == Approx(1.0));  // branch_p to P+
        REQUIRE(G.coeff(1, 4) == Approx(-1.0)); // P- to branch_p

        // Check coupling: V_p = n * V_s
        REQUIRE(G.coeff(4, 2) == Approx(-2.0));  // Coupling to S+
        REQUIRE(G.coeff(4, 3) == Approx(2.0));   // Coupling to S-
    }

    SECTION("Transformer Jacobian pattern") {
        auto pattern = Transformer::jacobian_pattern();
        REQUIRE(pattern.size() == 16);
    }
}

// =============================================================================
// Complete device suite static assertions
// =============================================================================

TEST_CASE("v2 CRTP static assertions", "[v2][crtp][static]") {
    // These are compile-time checks, but we verify the concepts are satisfied
    SECTION("All devices satisfy StampableDevice concept") {
        REQUIRE(StampableDevice<Resistor>);
        REQUIRE(StampableDevice<Capacitor>);
        REQUIRE(StampableDevice<Inductor>);
        REQUIRE(StampableDevice<VoltageSource>);
        REQUIRE(StampableDevice<CurrentSource>);
        REQUIRE(StampableDevice<IdealDiode>);
        REQUIRE(StampableDevice<IdealSwitch>);
        REQUIRE(StampableDevice<MOSFET>);
        REQUIRE(StampableDevice<IGBT>);
        REQUIRE(StampableDevice<Transformer>);
    }

    SECTION("Linear device classification") {
        REQUIRE(is_linear_device_v<Resistor> == true);
        REQUIRE(is_linear_device_v<Capacitor> == true);
        REQUIRE(is_linear_device_v<Inductor> == true);
        REQUIRE(is_linear_device_v<VoltageSource> == true);
        REQUIRE(is_linear_device_v<CurrentSource> == true);
        REQUIRE(is_linear_device_v<IdealDiode> == false);  // Nonlinear
        REQUIRE(is_linear_device_v<IdealSwitch> == true);  // Piecewise linear
        REQUIRE(is_linear_device_v<MOSFET> == false);      // Nonlinear
        REQUIRE(is_linear_device_v<IGBT> == false);        // Nonlinear
        REQUIRE(is_linear_device_v<Transformer> == true);
    }

    SECTION("Dynamic device classification") {
        REQUIRE(is_dynamic_device_v<Resistor> == false);
        REQUIRE(is_dynamic_device_v<Capacitor> == true);
        REQUIRE(is_dynamic_device_v<Inductor> == true);
        REQUIRE(is_dynamic_device_v<VoltageSource> == false);
        REQUIRE(is_dynamic_device_v<CurrentSource> == false);
        REQUIRE(is_dynamic_device_v<IdealDiode> == false);
        REQUIRE(is_dynamic_device_v<IdealSwitch> == false);
        REQUIRE(is_dynamic_device_v<MOSFET> == false);
        REQUIRE(is_dynamic_device_v<IGBT> == false);
        REQUIRE(is_dynamic_device_v<Transformer> == false);
    }

    SECTION("Power electronics device features") {
        // All power devices should have loss models
        REQUIRE(has_loss_model_v<Resistor> == true);
        REQUIRE(has_loss_model_v<IdealDiode> == true);
        REQUIRE(has_loss_model_v<IdealSwitch> == true);
        REQUIRE(has_loss_model_v<MOSFET> == true);
        REQUIRE(has_loss_model_v<IGBT> == true);

        // Semiconductor devices should have thermal models
        REQUIRE(has_thermal_model_v<MOSFET> == true);
        REQUIRE(has_thermal_model_v<IGBT> == true);
    }
}

// =============================================================================
// Tests for C++23 Advanced Features (1.2.3, 1.2.4, 1.2.7)
// =============================================================================

TEST_CASE("v2 StateBuffer (mdspan-like)", "[v2][cpp23][mdspan]") {
    SECTION("Fixed-size StateBuffer creation") {
        // 3 state variables, 3 history depth (current + 2 previous)
        StateBuffer<double, 3, 3> buffer;

        // Initialize current state
        buffer(0, 0) = 1.0;  // state 0, current
        buffer(1, 0) = 2.0;  // state 1, current
        buffer(2, 0) = 3.0;  // state 2, current

        REQUIRE(buffer(0, 0) == 1.0);
        REQUIRE(buffer(1, 0) == 2.0);
        REQUIRE(buffer(2, 0) == 3.0);
    }

    SECTION("StateBuffer history shift") {
        StateBuffer<double, 2, 3> buffer;

        // Set initial state
        buffer(0, 0) = 10.0;
        buffer(1, 0) = 20.0;

        // Shift history
        buffer.shift_history();

        // Old current should now be at history index 1
        REQUIRE(buffer(0, 1) == 10.0);
        REQUIRE(buffer(1, 1) == 20.0);

        // Set new current
        buffer(0, 0) = 11.0;
        buffer(1, 0) = 21.0;

        // Verify both current and previous
        REQUIRE(buffer(0, 0) == 11.0);
        REQUIRE(buffer(0, 1) == 10.0);
    }

    SECTION("StateBuffer span access") {
        StateBuffer<double, 4, 2> buffer;

        // Set via operator()
        buffer(0, 0) = 1.0;
        buffer(1, 0) = 2.0;
        buffer(2, 0) = 3.0;
        buffer(3, 0) = 4.0;

        // Access via span
        auto current = buffer.current();
        REQUIRE(current.size() == 4);
        REQUIRE(current[0] == 1.0);
        REQUIRE(current[3] == 4.0);
    }

    SECTION("StateBuffer compile-time properties") {
        using Buffer = StateBuffer<float, 5, 4>;
        REQUIRE(Buffer::num_states == 5);
        REQUIRE(Buffer::history_depth == 4);
    }
}

#if PULSIM_HAS_MDSPAN
TEST_CASE("v2 DynamicStateBuffer", "[v2][cpp23][mdspan]") {
    SECTION("Dynamic buffer creation") {
        DynamicStateBuffer<double> buffer(10, 3);

        REQUIRE(buffer.num_states() == 10);
        REQUIRE(buffer.history_depth() == 3);
    }

    SECTION("Dynamic buffer mdspan view") {
        DynamicStateBuffer<double> buffer(4, 2);

        auto view = buffer.view();
        // View should have dimensions (history_depth, num_states)
        REQUIRE(view.extent(0) == 2);
        REQUIRE(view.extent(1) == 4);
    }
}
#endif

TEST_CASE("v2 Device Metadata (reflection prep)", "[v2][cpp23][reflection]") {
    SECTION("DeviceMetadata structure") {
        DeviceMetadata meta{
            .name = "TestDevice",
            .category = "passive",
            .pin_count = 2,
            .is_linear = true,
            .is_dynamic = false,
            .has_thermal_model = false
        };

        REQUIRE(meta.name == "TestDevice");
        REQUIRE(meta.category == "passive");
        REQUIRE(meta.pin_count == 2);
        REQUIRE(meta.is_linear == true);
    }

    SECTION("ParamDescriptor structure") {
        ParamDescriptor param{
            .name = "resistance",
            .unit = "ohm",
            .default_value = 1000.0,
            .min_value = 0.0,
            .max_value = 1e12
        };

        REQUIRE(param.name == "resistance");
        REQUIRE(param.unit == "ohm");
        REQUIRE(param.default_value == 1000.0);
    }
}

TEST_CASE("v2 FixedString (compile-time string)", "[v2][cpp23][reflection]") {
    SECTION("FixedString creation and conversion") {
        constexpr FixedString str("Hello");

        REQUIRE(str.size() == 5);
        REQUIRE(static_cast<std::string_view>(str) == "Hello");
    }

    SECTION("FixedString compile-time") {
        constexpr FixedString str("Test");
        static_assert(str.size() == 4);
    }
}

TEST_CASE("v2 Type name utilities", "[v2][cpp23][reflection]") {
    SECTION("Built-in type names") {
        REQUIRE(type_name<double>() == "double");
        REQUIRE(type_name<float>() == "float");
        REQUIRE(type_name<int>() == "int");
        REQUIRE(type_name<bool>() == "bool");
    }
}

// =============================================================================
// Test device metadata registration (reflection prep)
// =============================================================================
// Note: All devices are registered in device_base.hpp using PULSIM_REGISTER_DEVICE

TEST_CASE("v2 Device registration (reflection prep)", "[v2][cpp23][reflection]") {
    SECTION("Passive device metadata") {
        constexpr auto r_meta = get_device_metadata<Resistor>();
        REQUIRE(r_meta.name == "Resistor");
        REQUIRE(r_meta.category == "passive");
        REQUIRE(r_meta.pin_count == 2);
        REQUIRE(r_meta.is_linear == true);
        REQUIRE(r_meta.is_dynamic == false);

        constexpr auto c_meta = get_device_metadata<Capacitor>();
        REQUIRE(c_meta.name == "Capacitor");
        REQUIRE(c_meta.is_dynamic == true);

        constexpr auto l_meta = get_device_metadata<Inductor>();
        REQUIRE(l_meta.name == "Inductor");
        REQUIRE(l_meta.is_dynamic == true);
    }

    SECTION("Source device metadata") {
        constexpr auto vs_meta = get_device_metadata<VoltageSource>();
        REQUIRE(vs_meta.name == "VoltageSource");
        REQUIRE(vs_meta.category == "source");

        constexpr auto cs_meta = get_device_metadata<CurrentSource>();
        REQUIRE(cs_meta.name == "CurrentSource");
        REQUIRE(cs_meta.category == "source");
    }

    SECTION("Active device metadata") {
        constexpr auto mosfet_meta = get_device_metadata<MOSFET>();
        REQUIRE(mosfet_meta.name == "MOSFET");
        REQUIRE(mosfet_meta.category == "active");
        REQUIRE(mosfet_meta.pin_count == 3);
        REQUIRE(mosfet_meta.is_linear == false);
        REQUIRE(mosfet_meta.has_thermal_model == true);

        constexpr auto igbt_meta = get_device_metadata<IGBT>();
        REQUIRE(igbt_meta.name == "IGBT");
        REQUIRE(igbt_meta.has_thermal_model == true);
    }

    SECTION("HasDeviceMetadata concept for all devices") {
        REQUIRE(HasDeviceMetadata<Resistor>);
        REQUIRE(HasDeviceMetadata<Capacitor>);
        REQUIRE(HasDeviceMetadata<Inductor>);
        REQUIRE(HasDeviceMetadata<VoltageSource>);
        REQUIRE(HasDeviceMetadata<CurrentSource>);
        REQUIRE(HasDeviceMetadata<IdealDiode>);
        REQUIRE(HasDeviceMetadata<IdealSwitch>);
        REQUIRE(HasDeviceMetadata<MOSFET>);
        REQUIRE(HasDeviceMetadata<IGBT>);
        REQUIRE(HasDeviceMetadata<Transformer>);
    }

    SECTION("Parameter registration") {
        // Check that Resistor has parameter registration
        constexpr auto r_params = ParamRegistry<Resistor>::params;
        REQUIRE(r_params.size() == 1);
        REQUIRE(r_params[0].name == "resistance");
        REQUIRE(r_params[0].unit == "Ohm");

        // Check MOSFET parameters
        constexpr auto m_params = ParamRegistry<MOSFET>::params;
        REQUIRE(m_params.size() == 3);
        REQUIRE(m_params[0].name == "vth");
    }
}

// =============================================================================
// Tests for Constexpr Utilities (1.3.5)
// =============================================================================

TEST_CASE("v2 Constexpr math utilities", "[v2][cpp23][constexpr]") {
    using namespace pulsim::v2;

    SECTION("Basic math functions") {
        // These are evaluated at compile time!
        static_assert(cabs(-5.0) == 5.0);
        static_assert(cabs(5.0) == 5.0);
        static_assert(sign(-3.0) == -1.0);
        static_assert(sign(3.0) == 1.0);
        static_assert(sign(0.0) == 0.0);
        static_assert(cmin(3.0, 5.0) == 3.0);
        static_assert(cmax(3.0, 5.0) == 5.0);
        static_assert(cclamp(7.0, 0.0, 5.0) == 5.0);
        static_assert(cclamp(-1.0, 0.0, 5.0) == 0.0);
        static_assert(cclamp(3.0, 0.0, 5.0) == 3.0);

        // Runtime checks
        REQUIRE(cabs(-5.0) == 5.0);
        REQUIRE(sign(-3.0) == -1.0);
        REQUIRE(cmin(3.0, 5.0) == 3.0);
        REQUIRE(cmax(3.0, 5.0) == 5.0);
    }

    SECTION("Power functions") {
        static_assert(cpow(2, 0) == 1);
        static_assert(cpow(2, 3) == 8);
        static_assert(cpow(3, 2) == 9);
        static_assert(cpow_n<int, 0>(2) == 1);
        static_assert(cpow_n<int, 4>(2) == 16);
        static_assert(cpow_n<double, 3>(2.0) == 8.0);

        REQUIRE(cpow(2, 10) == 1024);
        REQUIRE(cpow(2.0, -1) == Approx(0.5));  // Use double for negative exponent
    }

    SECTION("Square root") {
        REQUIRE(csqrt(0.0) == 0.0);
        REQUIRE(csqrt(1.0) == Approx(1.0).margin(1e-10));
        REQUIRE(csqrt(4.0) == Approx(2.0).margin(1e-10));
        REQUIRE(csqrt(9.0) == Approx(3.0).margin(1e-10));
        REQUIRE(csqrt(2.0) == Approx(1.41421356).margin(1e-6));
    }

    SECTION("Exponential and logarithm") {
        REQUIRE(cexp(0.0) == Approx(1.0).margin(1e-10));
        REQUIRE(cexp(1.0) == Approx(2.71828182).margin(1e-6));
        REQUIRE(clog(1.0) == Approx(0.0).margin(1e-10));
        REQUIRE(clog(std::exp(1.0)) == Approx(1.0).margin(1e-6));
    }

    SECTION("Approximate equality") {
        REQUIRE(approx_equal(1.0, 1.0 + 1e-12));
        REQUIRE(approx_equal(1000.0, 1000.001, 1e-5));
        REQUIRE(!approx_equal(1.0, 2.0));
    }

    SECTION("Safe division") {
        REQUIRE(safe_div(10.0, 2.0) == 5.0);
        REQUIRE(safe_div(1.0, 0.0, 999.0) == 999.0);  // Fallback value
    }

    SECTION("Smooth step functions") {
        REQUIRE(smoothstep(0.0, 1.0, -0.5) == 0.0);
        REQUIRE(smoothstep(0.0, 1.0, 0.5) == Approx(0.5).margin(0.1));
        REQUIRE(smoothstep(0.0, 1.0, 1.5) == 1.0);
    }
}

TEST_CASE("v2 Physical constants", "[v2][cpp23][constexpr]") {
    using namespace pulsim::v2;

    SECTION("Constants are constexpr") {
        static_assert(constants::k_B > 0);
        static_assert(constants::q_e > 0);
        static_assert(constants::V_T_300K > 0);
        static_assert(constants::pi > 3.14);

        REQUIRE(constants::V_T_300K == Approx(0.02585).margin(0.001));
    }

    SECTION("Thermal voltage calculation") {
        // At 300K
        REQUIRE(thermal_voltage(300.0) == Approx(constants::V_T_300K).margin(1e-6));
        // At higher temperature
        REQUIRE(thermal_voltage(400.0) > thermal_voltage(300.0));
    }
}

TEST_CASE("v2 Constexpr array utilities", "[v2][cpp23][constexpr]") {
    using namespace pulsim::v2;

    SECTION("make_filled_array") {
        constexpr auto arr = make_filled_array<double, 5>(3.14);
        static_assert(arr[0] == 3.14);
        static_assert(arr[4] == 3.14);

        REQUIRE(arr[0] == 3.14);
        REQUIRE(arr[4] == 3.14);
    }

    SECTION("make_iota_array") {
        constexpr auto arr = make_iota_array<int, 5>();
        static_assert(arr[0] == 0);
        static_assert(arr[4] == 4);

        REQUIRE(arr[0] == 0);
        REQUIRE(arr[2] == 2);
        REQUIRE(arr[4] == 4);
    }

    SECTION("array_sum") {
        constexpr std::array<int, 4> arr = {1, 2, 3, 4};
        static_assert(array_sum(arr) == 10);

        REQUIRE(array_sum(arr) == 10);
    }

    SECTION("array_dot") {
        constexpr std::array<double, 3> a = {1.0, 2.0, 3.0};
        constexpr std::array<double, 3> b = {4.0, 5.0, 6.0};
        static_assert(array_dot(a, b) == 32.0);  // 1*4 + 2*5 + 3*6

        REQUIRE(array_dot(a, b) == 32.0);
    }

    SECTION("array_norm") {
        constexpr std::array<double, 2> arr = {3.0, 4.0};
        REQUIRE(array_norm(arr) == Approx(5.0).margin(1e-10));  // sqrt(9+16)
    }

    SECTION("array_max and array_min") {
        constexpr std::array<int, 5> arr = {3, 1, 4, 1, 5};
        static_assert(array_max(arr) == 5);
        static_assert(array_min(arr) == 1);

        REQUIRE(array_max(arr) == 5);
        REQUIRE(array_min(arr) == 1);
    }

    SECTION("transform_array") {
        constexpr std::array<int, 3> arr = {1, 2, 3};
        constexpr auto doubled = transform_array(arr, [](int x) { return x * 2; });
        static_assert(doubled[0] == 2);
        static_assert(doubled[1] == 4);
        static_assert(doubled[2] == 6);

        REQUIRE(doubled[0] == 2);
        REQUIRE(doubled[1] == 4);
    }

    SECTION("generate_array") {
        constexpr auto squares = generate_array<int, 5>([](std::size_t i) {
            return static_cast<int>(i * i);
        });
        static_assert(squares[0] == 0);
        static_assert(squares[1] == 1);
        static_assert(squares[2] == 4);
        static_assert(squares[4] == 16);

        REQUIRE(squares[3] == 9);
    }
}

TEST_CASE("v2 Lookup tables", "[v2][cpp23][constexpr]") {
    using namespace pulsim::v2;

    SECTION("LUT generation and interpolation") {
        // Create a lookup table for x^2 from 0 to 10
        constexpr auto lut = make_lut<double, 11>([](double x) { return x * x; }, 0.0, 10.0);

        // Check exact values at sample points
        REQUIRE(lut[0] == Approx(0.0));
        REQUIRE(lut[5] == Approx(25.0));
        REQUIRE(lut[10] == Approx(100.0));

        // Test interpolation
        REQUIRE(lut_interpolate(lut, 2.5, 0.0, 10.0) == Approx(6.25).margin(0.5));
    }
}

TEST_CASE("v2 Unit conversions", "[v2][cpp23][constexpr]") {
    using namespace pulsim::v2;

    SECTION("SI prefixes") {
        static_assert(units::nano == 1e-9);
        static_assert(units::micro == 1e-6);
        static_assert(units::milli == 1e-3);
        static_assert(units::kilo == 1e3);
        static_assert(units::mega == 1e6);

        REQUIRE(1.0 * units::micro == 1e-6);
    }

    SECTION("Frequency conversions") {
        REQUIRE(units::hz_to_rad(1.0) == Approx(2.0 * constants::pi));
        REQUIRE(units::rad_to_hz(2.0 * constants::pi) == Approx(1.0));
        REQUIRE(units::period_to_freq(0.001) == Approx(1000.0));
    }
}
