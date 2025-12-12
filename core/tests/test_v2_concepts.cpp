// =============================================================================
// Test: C++23 v2 API Concepts and CRTP Devices
// =============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v2/core.hpp"
#include "pulsim/v2/compat.hpp"

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

// =============================================================================
// Tests for Numeric Types (Phase 2.1)
// =============================================================================

TEST_CASE("v2 Real type configuration", "[v2][numeric][real]") {
    using namespace pulsim::v2;

    SECTION("Precision selection") {
        static_assert(std::is_same_v<RealT<Precision::Double>, double>);
        static_assert(std::is_same_v<RealT<Precision::Single>, float>);
        static_assert(std::is_same_v<RealD, double>);
        static_assert(std::is_same_v<RealS, float>);
        static_assert(std::is_same_v<Real, double>);  // Default alias
    }

    SECTION("RealTraits") {
        static_assert(RealTraits<double>::default_abstol == 1e-9);
        static_assert(RealTraits<float>::default_abstol == 1e-6f);
        static_assert(RealTraits<double>::default_reltol == 1e-3);

        REQUIRE(RealTraits<double>::epsilon > 0);
        REQUIRE(RealTraits<double>::digits == 53);  // IEEE 754 double
    }

    SECTION("RealType concept") {
        static_assert(RealType<double>);
        static_assert(RealType<float>);
        static_assert(!RealType<int>);
    }
}

TEST_CASE("v2 Index type configuration", "[v2][numeric][index]") {
    using namespace pulsim::v2;

    SECTION("Index width selection") {
        static_assert(std::is_same_v<IndexT<IndexWidth::Narrow>, std::int32_t>);
        static_assert(std::is_same_v<IndexT<IndexWidth::Wide>, std::int64_t>);
        static_assert(sizeof(Index32) == 4);
        static_assert(sizeof(Index64) == 8);
        static_assert(std::is_same_v<Index, std::int32_t>);  // Default alias
    }

    SECTION("Ground node constant") {
        static_assert(ground_node_v<Index32> == -1);
        static_assert(ground_node_v<Index64> == -1);
        REQUIRE(ground_node == -1);  // Default constant
    }

    SECTION("IndexType concept") {
        static_assert(IndexType<int>);
        static_assert(IndexType<std::int64_t>);
        static_assert(!IndexType<unsigned int>);
    }
}

TEST_CASE("v2 StaticVector", "[v2][numeric][vector]") {
    using namespace pulsim::v2;

    SECTION("Construction") {
        StaticVector<double, 3> v1;  // Default: zeros
        REQUIRE(v1[0] == 0.0);
        REQUIRE(v1[1] == 0.0);
        REQUIRE(v1[2] == 0.0);

        StaticVector<double, 3> v2(5.0);  // Fill
        REQUIRE(v2[0] == 5.0);
        REQUIRE(v2[1] == 5.0);

        StaticVector<int, 3> v3{1, 2, 3};  // Initializer list
        REQUIRE(v3[0] == 1);
        REQUIRE(v3[1] == 2);
        REQUIRE(v3[2] == 3);

        Vec3d v4(1.0, 2.0, 3.0);  // Variadic
        REQUIRE(v4[0] == 1.0);
        REQUIRE(v4[2] == 3.0);
    }

    SECTION("Element access") {
        Vec3d v{1.0, 2.0, 3.0};
        REQUIRE(v[0] == 1.0);
        REQUIRE(v.front() == 1.0);
        REQUIRE(v.back() == 3.0);
        REQUIRE(v.size() == 3);
    }

    SECTION("Arithmetic operations") {
        Vec3d a{1.0, 2.0, 3.0};
        Vec3d b{4.0, 5.0, 6.0};

        auto sum = a + b;
        REQUIRE(sum[0] == 5.0);
        REQUIRE(sum[1] == 7.0);
        REQUIRE(sum[2] == 9.0);

        auto diff = b - a;
        REQUIRE(diff[0] == 3.0);

        auto scaled = a * 2.0;
        REQUIRE(scaled[0] == 2.0);
        REQUIRE(scaled[2] == 6.0);

        auto neg = -a;
        REQUIRE(neg[0] == -1.0);
    }

    SECTION("Vector operations") {
        Vec3d a{1.0, 2.0, 3.0};
        Vec3d b{4.0, 5.0, 6.0};

        // Dot product: 1*4 + 2*5 + 3*6 = 32
        REQUIRE(a.dot(b) == 32.0);

        // Squared norm: 1 + 4 + 9 = 14
        REQUIRE(a.squared_norm() == 14.0);

        // Norm
        REQUIRE(a.norm() == Approx(std::sqrt(14.0)));

        // Sum
        REQUIRE(a.sum() == 6.0);

        // Min/Max
        REQUIRE(a.min_element() == 1.0);
        REQUIRE(a.max_element() == 3.0);

        Vec3d c{-5.0, 3.0, -2.0};
        REQUIRE(c.max_abs() == 5.0);
    }

    SECTION("Constexpr operations") {
        constexpr Vec3<int> v{1, 2, 3};
        static_assert(v[0] == 1);
        static_assert(v.size() == 3);
        static_assert((v + Vec3<int>{1, 1, 1})[0] == 2);
        static_assert(v.dot(Vec3<int>{1, 1, 1}) == 6);
    }

    SECTION("Iteration") {
        Vec3d v{1.0, 2.0, 3.0};
        double sum = 0;
        for (double x : v) sum += x;
        REQUIRE(sum == 6.0);
    }
}

TEST_CASE("v2 StaticMatrix", "[v2][numeric][matrix]") {
    using namespace pulsim::v2;

    SECTION("Construction") {
        Mat2d m1;  // Default: zeros
        REQUIRE(m1(0, 0) == 0.0);

        Mat2d m2(1.0);  // Fill with 1
        REQUIRE(m2(0, 0) == 1.0);
        REQUIRE(m2(1, 1) == 1.0);

        Mat2d m3{{1.0, 2.0}, {3.0, 4.0}};  // Initializer list
        REQUIRE(m3(0, 0) == 1.0);
        REQUIRE(m3(0, 1) == 2.0);
        REQUIRE(m3(1, 0) == 3.0);
        REQUIRE(m3(1, 1) == 4.0);
    }

    SECTION("Identity matrix") {
        auto I = Mat3d::identity();
        REQUIRE(I(0, 0) == 1.0);
        REQUIRE(I(1, 1) == 1.0);
        REQUIRE(I(2, 2) == 1.0);
        REQUIRE(I(0, 1) == 0.0);
        REQUIRE(I(1, 0) == 0.0);
    }

    SECTION("Row and column access") {
        Mat2d m{{1.0, 2.0}, {3.0, 4.0}};
        auto r0 = m.row(0);
        REQUIRE(r0[0] == 1.0);
        REQUIRE(r0[1] == 2.0);

        auto c1 = m.col(1);
        REQUIRE(c1[0] == 2.0);
        REQUIRE(c1[1] == 4.0);
    }

    SECTION("Arithmetic operations") {
        Mat2d a{{1.0, 2.0}, {3.0, 4.0}};
        Mat2d b{{5.0, 6.0}, {7.0, 8.0}};

        auto sum = a + b;
        REQUIRE(sum(0, 0) == 6.0);
        REQUIRE(sum(1, 1) == 12.0);

        auto scaled = a * 2.0;
        REQUIRE(scaled(0, 0) == 2.0);
        REQUIRE(scaled(1, 1) == 8.0);
    }

    SECTION("Matrix-vector multiplication") {
        Mat2d m{{1.0, 2.0}, {3.0, 4.0}};
        Vec2d v{1.0, 1.0};

        auto result = m * v;
        REQUIRE(result[0] == 3.0);   // 1*1 + 2*1
        REQUIRE(result[1] == 7.0);   // 3*1 + 4*1
    }

    SECTION("Matrix-matrix multiplication") {
        Mat2d a{{1.0, 2.0}, {3.0, 4.0}};
        Mat2d b{{5.0, 6.0}, {7.0, 8.0}};

        auto c = a * b;
        REQUIRE(c(0, 0) == 19.0);  // 1*5 + 2*7
        REQUIRE(c(0, 1) == 22.0);  // 1*6 + 2*8
        REQUIRE(c(1, 0) == 43.0);  // 3*5 + 4*7
        REQUIRE(c(1, 1) == 50.0);  // 3*6 + 4*8
    }

    SECTION("Transpose") {
        Mat2d m{{1.0, 2.0}, {3.0, 4.0}};
        auto t = m.transpose();
        REQUIRE(t(0, 0) == 1.0);
        REQUIRE(t(0, 1) == 3.0);
        REQUIRE(t(1, 0) == 2.0);
        REQUIRE(t(1, 1) == 4.0);
    }

    SECTION("Determinant and inverse") {
        Mat2d m{{4.0, 7.0}, {2.0, 6.0}};
        REQUIRE(m.determinant() == Approx(10.0));  // 4*6 - 7*2

        auto inv = m.inverse();
        auto identity = m * inv;
        REQUIRE(identity(0, 0) == Approx(1.0).margin(1e-10));
        REQUIRE(identity(0, 1) == Approx(0.0).margin(1e-10));
        REQUIRE(identity(1, 0) == Approx(0.0).margin(1e-10));
        REQUIRE(identity(1, 1) == Approx(1.0).margin(1e-10));
    }

    SECTION("Trace") {
        Mat3d m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        REQUIRE(m.trace() == 15.0);  // 1 + 5 + 9
    }

    SECTION("Constexpr operations") {
        constexpr Mat2<int> m{{1, 2}, {3, 4}};
        static_assert(m(0, 0) == 1);
        static_assert(m(1, 1) == 4);
        static_assert(Mat2<int>::identity()(0, 0) == 1);
    }
}

TEST_CASE("v2 SparsityPattern", "[v2][numeric][sparsity]") {
    using namespace pulsim::v2;

    SECTION("Construction") {
        SparsityPattern<10> p;
        REQUIRE(p.size() == 0);
        REQUIRE(p.empty());

        p.add(0, 0);
        p.add(0, 1);
        p.add(1, 0);
        p.add(1, 1);
        REQUIRE(p.size() == 4);
        REQUIRE(!p.empty());
    }

    SECTION("Predefined patterns") {
        auto p2x2 = make_2x2_pattern();
        REQUIRE(p2x2.size() == 4);
        REQUIRE(p2x2.contains(0, 0));
        REQUIRE(p2x2.contains(1, 1));

        auto p3x3 = make_3x3_pattern();
        REQUIRE(p3x3.size() == 9);
    }

    SECTION("Contains check") {
        SparsityPattern<10> p{{PatternEntry{0, 0}, PatternEntry{1, 1}}};
        REQUIRE(p.contains(0, 0));
        REQUIRE(p.contains(1, 1));
        REQUIRE(!p.contains(0, 1));
    }

    SECTION("Merge patterns") {
        SparsityPattern<4> p1{{PatternEntry{0, 0}, PatternEntry{0, 1}}};
        SparsityPattern<4> p2{{PatternEntry{1, 0}, PatternEntry{1, 1}}};

        auto merged = p1.merge(p2);
        REQUIRE(merged.size() == 4);
        REQUIRE(merged.contains(0, 0));
        REQUIRE(merged.contains(1, 1));
    }

    SECTION("Max indices") {
        SparsityPattern<4> p{{PatternEntry{0, 0}, PatternEntry{2, 3}}};
        auto [max_r, max_c] = p.max_indices();
        REQUIRE(max_r == 2);
        REQUIRE(max_c == 3);
    }

    SECTION("Iteration") {
        auto p = make_2x2_pattern();
        int count = 0;
        for (const auto& entry : p) {
            REQUIRE(entry.valid());
            ++count;
        }
        REQUIRE(count == 4);
    }
}

TEST_CASE("v2 Units and dimensional analysis", "[v2][numeric][units]") {
    using namespace pulsim::v2;
    using namespace pulsim::v2::literals;

    SECTION("Quantity creation") {
        Voltage<> v(5.0);
        REQUIRE(v.value() == 5.0);

        Current<> i(2.0);
        REQUIRE(i.value() == 2.0);
    }

    SECTION("User-defined literals") {
        auto v = 5.0_V;
        REQUIRE(v.value() == 5.0);

        auto v_mV = 100.0_mV;
        REQUIRE(v_mV.value() == Approx(0.1));

        auto i = 1.0_A;
        REQUIRE(i.value() == 1.0);

        auto i_mA = 500.0_mA;
        REQUIRE(i_mA.value() == Approx(0.5));

        auto c = 100.0_uF;
        REQUIRE(c.value() == Approx(100e-6));

        auto l = 10.0_mH;
        REQUIRE(l.value() == Approx(10e-3));

        auto t = 1.0_ms;
        REQUIRE(t.value() == Approx(1e-3));

        auto f = 1.0_kHz;
        REQUIRE(f.value() == Approx(1e3));
    }

    SECTION("Same-dimension arithmetic") {
        auto v1 = 5.0_V;
        auto v2 = 3.0_V;

        auto sum = v1 + v2;
        REQUIRE(sum.value() == 8.0);

        auto diff = v1 - v2;
        REQUIRE(diff.value() == 2.0);

        auto neg = -v1;
        REQUIRE(neg.value() == -5.0);

        auto scaled = v1 * 2.0;
        REQUIRE(scaled.value() == 10.0);
    }

    SECTION("Dimensional analysis: V = I * R") {
        Current<> i(2.0);
        Resistance<> r(100.0);

        auto v = i * r;  // Should be Voltage
        static_assert(std::is_same_v<decltype(v), Voltage<>>);
        REQUIRE(v.value() == 200.0);
    }

    SECTION("Dimensional analysis: P = V * I") {
        Voltage<> v(10.0);
        Current<> i(2.0);

        auto p = v * i;  // Should be Power
        static_assert(std::is_same_v<decltype(p), Power<>>);
        REQUIRE(p.value() == 20.0);
    }

    SECTION("Dimensional analysis: R = V / I") {
        Voltage<> v(10.0);
        Current<> i(2.0);

        auto r = v / i;  // Should be Resistance
        static_assert(std::is_same_v<decltype(r), Resistance<>>);
        REQUIRE(r.value() == 5.0);
    }

    SECTION("Comparison") {
        auto v1 = 5.0_V;
        auto v2 = 3.0_V;
        auto v3 = 5.0_V;

        REQUIRE(v1 > v2);
        REQUIRE(v2 < v1);
        REQUIRE(v1 == v3);
        REQUIRE(v1 >= v3);
        REQUIRE(v2 <= v1);
    }
}

// =============================================================================
// Tests for Normalization/Scaling Helpers (Phase 2.1.7)
// =============================================================================

TEST_CASE("v2 ScalingFactors", "[v2][numeric][scaling]") {
    using namespace pulsim::v2;

    SECTION("Default scaling") {
        ScalingFactors<> sf;
        REQUIRE(sf.voltage_scale == 1.0);
        REQUIRE(sf.current_scale == 1.0);
        REQUIRE(sf.time_scale == 1.0);
        REQUIRE(sf.conductance_scale == 1.0);
    }

    SECTION("Custom scaling") {
        ScalingFactors<> sf(100.0, 10.0, 1e-6);
        REQUIRE(sf.voltage_scale == 100.0);
        REQUIRE(sf.current_scale == 10.0);
        REQUIRE(sf.time_scale == 1e-6);
        REQUIRE(sf.conductance_scale == Approx(0.1));  // 10/100
    }

    SECTION("Preset scaling factors") {
        auto pe = ScalingFactors<>::power_electronics();
        REQUIRE(pe.voltage_scale == 100.0);
        REQUIRE(pe.current_scale == 1.0);

        auto sig = ScalingFactors<>::signal_level();
        REQUIRE(sig.voltage_scale == 1.0);
        REQUIRE(sig.current_scale == 1e-3);

        auto hp = ScalingFactors<>::high_power();
        REQUIRE(hp.voltage_scale == 1000.0);
        REQUIRE(hp.current_scale == 100.0);
    }

    SECTION("From circuit characteristics") {
        auto sf = ScalingFactors<>::from_circuit(400.0, 50.0, 1e-6);
        REQUIRE(sf.voltage_scale == 400.0);
        REQUIRE(sf.current_scale == 50.0);
        REQUIRE(sf.time_scale == 1e-6);
    }
}

TEST_CASE("v2 VariableNormalizer", "[v2][numeric][scaling]") {
    using namespace pulsim::v2;

    SECTION("Identity normalization") {
        VariableNormalizer<> norm;
        REQUIRE_FALSE(norm.is_active());
        REQUIRE(norm.normalize_voltage(100.0) == 100.0);
        REQUIRE(norm.denormalize_voltage(100.0) == 100.0);
    }

    SECTION("Voltage/current normalization") {
        ScalingFactors<> sf(100.0, 10.0, 1e-6);
        VariableNormalizer<> norm(sf);

        REQUIRE(norm.is_active());

        // Normalize
        REQUIRE(norm.normalize_voltage(200.0) == Approx(2.0));   // 200/100
        REQUIRE(norm.normalize_current(5.0) == Approx(0.5));     // 5/10

        // Denormalize
        REQUIRE(norm.denormalize_voltage(2.0) == Approx(200.0));
        REQUIRE(norm.denormalize_current(0.5) == Approx(5.0));
    }

    SECTION("Conductance/resistance normalization") {
        ScalingFactors<> sf(100.0, 10.0, 1e-6);
        VariableNormalizer<> norm(sf);

        // G = I/V, so G_base = 10/100 = 0.1 S
        REQUIRE(norm.normalize_conductance(0.05) == Approx(0.5));   // 0.05/0.1
        REQUIRE(norm.normalize_resistance(200.0) == Approx(20.0));   // 200*0.1

        REQUIRE(norm.denormalize_conductance(0.5) == Approx(0.05));
        REQUIRE(norm.denormalize_resistance(20.0) == Approx(200.0));
    }

    SECTION("Capacitance/inductance normalization") {
        ScalingFactors<> sf(100.0, 10.0, 1e-6);  // V=100, I=10, t=1μs
        VariableNormalizer<> norm(sf);

        // C_norm = C * V / (I * t) = C * 100 / (10 * 1e-6) = C * 1e7
        double c_phys = 100e-6;  // 100μF
        double c_norm = norm.normalize_capacitance(c_phys);
        REQUIRE(c_norm == Approx(100e-6 * 100.0 / (10.0 * 1e-6)));

        // L_norm = L * I / (V * t) = L * 10 / (100 * 1e-6) = L * 1e5
        double l_phys = 10e-3;  // 10mH
        double l_norm = norm.normalize_inductance(l_phys);
        REQUIRE(l_norm == Approx(10e-3 * 10.0 / (100.0 * 1e-6)));
    }
}

TEST_CASE("v2 WeightedNorm", "[v2][numeric][scaling]") {
    using namespace pulsim::v2;

    SECTION("Default tolerances") {
        WeightedNorm<> wnorm;
        auto tol = wnorm.tolerances();
        REQUIRE(tol.abstol_v == 1e-9);
        REQUIRE(tol.reltol_v == 1e-3);
        REQUIRE(tol.abstol_i == 1e-12);
        REQUIRE(tol.reltol_i == 1e-3);
    }

    SECTION("Convergence check") {
        WeightedNorm<>::Tolerances tol{1e-6, 1e-3, 1e-9, 1e-3};
        WeightedNorm<> wnorm(tol);

        // Simple 3-node, 1-branch system
        Eigen::VectorXd delta(4);
        Eigen::VectorXd solution(4);

        // Solution: V1=10V, V2=5V, V3=3V, I1=0.1A
        solution << 10.0, 5.0, 3.0, 0.1;

        // Small delta - should converge
        delta << 1e-9, 1e-9, 1e-9, 1e-12;
        REQUIRE(wnorm.has_converged(delta, solution, 3, 1));

        // Large delta - should not converge
        delta << 0.1, 0.1, 0.1, 0.001;
        REQUIRE_FALSE(wnorm.has_converged(delta, solution, 3, 1));
    }
}

TEST_CASE("v2 PerUnitSystem", "[v2][numeric][scaling]") {
    using namespace pulsim::v2;

    SECTION("Base quantity calculations") {
        PerUnitSystem<>::BaseQuantities base{1000.0, 400.0, 50.0};  // 1kVA, 400V, 50Hz

        REQUIRE(base.I_base() == Approx(2.5));     // 1000/400
        REQUIRE(base.Z_base() == Approx(160.0));   // 400²/1000
        REQUIRE(base.Y_base() == Approx(0.00625)); // 1000/400²
    }

    SECTION("Per-unit conversions") {
        PerUnitSystem<>::BaseQuantities base{1000.0, 400.0, 50.0};
        PerUnitSystem<> pu(base);

        // Voltage: 800V should be 2 p.u.
        REQUIRE(pu.to_pu_voltage(800.0) == Approx(2.0));
        REQUIRE(pu.from_pu_voltage(2.0) == Approx(800.0));

        // Current: 5A should be 2 p.u.
        REQUIRE(pu.to_pu_current(5.0) == Approx(2.0));
        REQUIRE(pu.from_pu_current(2.0) == Approx(5.0));

        // Power: 2000W should be 2 p.u.
        REQUIRE(pu.to_pu_power(2000.0) == Approx(2.0));
        REQUIRE(pu.from_pu_power(2.0) == Approx(2000.0));

        // Impedance: 320Ω should be 2 p.u.
        REQUIRE(pu.to_pu_impedance(320.0) == Approx(2.0));
        REQUIRE(pu.from_pu_impedance(2.0) == Approx(320.0));
    }
}

// =============================================================================
// Tests for Backward Compatibility Shim (Phase 1.3.6/1.3.7)
// =============================================================================

TEST_CASE("v2 Compat header", "[v2][compat]") {
    using namespace pulsim::compat;

    SECTION("Type aliases") {
        static_assert(std::is_same_v<Real, double>);
        static_assert(std::is_same_v<Index, std::int32_t>);
        REQUIRE(ground == -1);
    }

    SECTION("Device type mapping") {
        // Check that compat devices are v2 devices
        static_assert(std::is_same_v<Resistor, pulsim::v2::Resistor>);
        static_assert(std::is_same_v<Capacitor, pulsim::v2::Capacitor>);
        static_assert(std::is_same_v<Inductor, pulsim::v2::Inductor>);
    }

    SECTION("Version detection") {
        // PULSIM_USE_V2 is 0 by default
        REQUIRE_FALSE(is_v2_enabled());
        REQUIRE(std::string(api_version_string()) == "v1");
        REQUIRE(std::string(version_string()) == "2.0.0");
    }
}

TEST_CASE("v2 Migration helpers", "[v2][compat][migration]") {
    using namespace pulsim::migration;
    using namespace pulsim::v2;

    SECTION("Component to device type mapping") {
        REQUIRE(component_to_device_type(0) == DeviceType::Resistor);
        REQUIRE(component_to_device_type(1) == DeviceType::Capacitor);
        REQUIRE(component_to_device_type(2) == DeviceType::Inductor);
        REQUIRE(component_to_device_type(3) == DeviceType::VoltageSource);
        REQUIRE(component_to_device_type(4) == DeviceType::CurrentSource);
        REQUIRE(component_to_device_type(9) == DeviceType::Diode);
        REQUIRE(component_to_device_type(10) == DeviceType::Switch);
        REQUIRE(component_to_device_type(11) == DeviceType::MOSFET);
        REQUIRE(component_to_device_type(12) == DeviceType::IGBT);
        REQUIRE(component_to_device_type(13) == DeviceType::Transformer);
        REQUIRE(component_to_device_type(99) == DeviceType::Unknown);
    }

    SECTION("Device support check") {
        REQUIRE(is_device_supported_v2(DeviceType::Resistor));
        REQUIRE(is_device_supported_v2(DeviceType::MOSFET));
        REQUIRE_FALSE(is_device_supported_v2(DeviceType::Unknown));
    }
}

// =============================================================================
// Tests for Expression Templates (Phase 2.4)
// =============================================================================

TEST_CASE("v2 Expression templates - basic operations", "[v2][expression]") {
    using namespace pulsim::v2;

    SECTION("Vector addition expression") {
        StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
        StaticVector<double, 4> b{5.0, 6.0, 7.0, 8.0};

        auto expr = a + b;
        REQUIRE(expr[0] == 6.0);
        REQUIRE(expr[1] == 8.0);
        REQUIRE(expr[2] == 10.0);
        REQUIRE(expr[3] == 12.0);
        REQUIRE(expr.size() == 4);
    }

    SECTION("Vector subtraction expression") {
        StaticVector<double, 4> a{10.0, 20.0, 30.0, 40.0};
        StaticVector<double, 4> b{1.0, 2.0, 3.0, 4.0};

        auto expr = a - b;
        REQUIRE(expr[0] == 9.0);
        REQUIRE(expr[1] == 18.0);
        REQUIRE(expr[2] == 27.0);
        REQUIRE(expr[3] == 36.0);
    }

    SECTION("Scalar multiplication expression") {
        StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};

        auto expr1 = 2.0 * a;
        REQUIRE(expr1[0] == 2.0);
        REQUIRE(expr1[3] == 8.0);

        auto expr2 = a * 3.0;
        REQUIRE(expr2[0] == 3.0);
        REQUIRE(expr2[3] == 12.0);
    }

    SECTION("Scalar division expression") {
        StaticVector<double, 4> a{10.0, 20.0, 30.0, 40.0};

        auto expr = a / 10.0;
        REQUIRE(expr[0] == 1.0);
        REQUIRE(expr[3] == 4.0);
    }

    SECTION("Negation expression") {
        StaticVector<double, 4> a{1.0, -2.0, 3.0, -4.0};

        auto expr = -a;
        REQUIRE(expr[0] == -1.0);
        REQUIRE(expr[1] == 2.0);
        REQUIRE(expr[2] == -3.0);
        REQUIRE(expr[3] == 4.0);
    }
}

TEST_CASE("v2 Expression templates - nested expressions", "[v2][expression]") {
    using namespace pulsim::v2;

    SECTION("Chained addition") {
        StaticVector<double, 3> a{1.0, 2.0, 3.0};
        StaticVector<double, 3> b{4.0, 5.0, 6.0};
        StaticVector<double, 3> c{7.0, 8.0, 9.0};

        auto expr = a + b + c;
        REQUIRE(expr[0] == 12.0);
        REQUIRE(expr[1] == 15.0);
        REQUIRE(expr[2] == 18.0);
    }

    SECTION("Mixed operations") {
        StaticVector<double, 3> a{1.0, 2.0, 3.0};
        StaticVector<double, 3> b{1.0, 1.0, 1.0};

        // 2 * (a + b)
        auto expr = 2.0 * (a + b);
        REQUIRE(expr[0] == 4.0);
        REQUIRE(expr[1] == 6.0);
        REQUIRE(expr[2] == 8.0);
    }

    SECTION("Complex expression") {
        StaticVector<double, 3> a{1.0, 2.0, 3.0};
        StaticVector<double, 3> b{4.0, 5.0, 6.0};

        // (a + b) - 2.0 * a
        auto expr = (a + b) - (2.0 * a);
        REQUIRE(expr[0] == Approx(3.0));  // (1+4) - 2*1 = 3
        REQUIRE(expr[1] == Approx(3.0));  // (2+5) - 2*2 = 3
        REQUIRE(expr[2] == Approx(3.0));  // (3+6) - 2*3 = 3
    }
}

TEST_CASE("v2 Expression templates - lazy evaluation", "[v2][expression]") {
    using namespace pulsim::v2;

    SECTION("eval function") {
        StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
        StaticVector<double, 4> b{5.0, 6.0, 7.0, 8.0};

        auto expr = a + b;
        auto result = eval<decltype(expr), 4>(expr);

        REQUIRE(result[0] == 6.0);
        REQUIRE(result[3] == 12.0);
    }

    SECTION("eval_to function") {
        StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
        StaticVector<double, 4> b{5.0, 6.0, 7.0, 8.0};
        StaticVector<double, 4> result;

        auto expr = a + b;
        eval_to(expr, result);

        REQUIRE(result[0] == 6.0);
        REQUIRE(result[3] == 12.0);
    }

    SECTION("Expression sum") {
        StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
        StaticVector<double, 4> b{1.0, 1.0, 1.0, 1.0};

        auto expr = a + b;
        REQUIRE(expr.sum() == 14.0);  // (1+1) + (2+1) + (3+1) + (4+1)
    }

    SECTION("Expression dot product") {
        StaticVector<double, 3> a{1.0, 2.0, 3.0};
        StaticVector<double, 3> b{1.0, 1.0, 1.0};
        StaticVector<double, 3> c{4.0, 5.0, 6.0};

        auto expr = a + b;  // {2, 3, 4}
        REQUIRE(expr.dot(c) == 47.0);  // 2*4 + 3*5 + 4*6 = 47
    }

    SECTION("Expression norms") {
        StaticVector<double, 3> a{3.0, 0.0, 4.0};
        StaticVector<double, 3> b{0.0, 0.0, 0.0};

        auto expr = a + b;
        REQUIRE(expr.squared_norm() == 25.0);  // 9 + 0 + 16
        REQUIRE(expr.norm() == Approx(5.0));
    }
}

TEST_CASE("v2 Expression templates - VectorRef", "[v2][expression]") {
    using namespace pulsim::v2;

    SECTION("VectorRef from raw array") {
        double data[] = {1.0, 2.0, 3.0, 4.0};
        VectorRef<double> ref(data, 4);

        REQUIRE(ref.size() == 4);
        REQUIRE(ref[0] == 1.0);
        REQUIRE(ref[3] == 4.0);
    }

    SECTION("VectorRef from StaticVector") {
        StaticVector<double, 4> vec{1.0, 2.0, 3.0, 4.0};
        VectorRef ref(vec);

        REQUIRE(ref.size() == 4);
        REQUIRE(ref[0] == 1.0);
    }

    SECTION("VectorRef in expressions") {
        StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
        double b_data[] = {5.0, 6.0, 7.0, 8.0};
        VectorRef<double> b(b_data, 4);

        auto expr = a + b;
        REQUIRE(expr[0] == 6.0);
        REQUIRE(expr[3] == 12.0);
    }
}

TEST_CASE("v2 Expression templates - constexpr", "[v2][expression]") {
    using namespace pulsim::v2;

    SECTION("Constexpr expression evaluation") {
        constexpr StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
        constexpr StaticVector<double, 4> b{5.0, 6.0, 7.0, 8.0};

        // Expression can be evaluated at compile time
        static_assert((a + b)[0] == 6.0);
        static_assert((a + b)[3] == 12.0);
        static_assert((2.0 * a)[0] == 2.0);
        static_assert((a - b)[0] == -4.0);
    }
}

// =============================================================================
// Tests for Compile-Time Circuit Analysis (Phase 2.5)
// =============================================================================

TEST_CASE("v2 Circuit device traits", "[v2][circuit][traits]") {
    using namespace pulsim::v2;

    SECTION("Terminal counts") {
        REQUIRE(circuit_device_traits<Resistor>::terminal_count == 2);
        REQUIRE(circuit_device_traits<Capacitor>::terminal_count == 2);
        REQUIRE(circuit_device_traits<Inductor>::terminal_count == 2);
        REQUIRE(circuit_device_traits<VoltageSource>::terminal_count == 2);
        REQUIRE(circuit_device_traits<MOSFET>::terminal_count == 3);
        REQUIRE(circuit_device_traits<Transformer>::terminal_count == 4);
    }

    SECTION("Branch counts") {
        REQUIRE(circuit_device_traits<Resistor>::branch_count == 0);
        REQUIRE(circuit_device_traits<Capacitor>::branch_count == 0);
        REQUIRE(circuit_device_traits<Inductor>::branch_count == 1);
        REQUIRE(circuit_device_traits<VoltageSource>::branch_count == 1);
        REQUIRE(circuit_device_traits<CurrentSource>::branch_count == 0);
        REQUIRE(circuit_device_traits<Transformer>::branch_count == 2);
    }

    SECTION("Linear/nonlinear classification") {
        REQUIRE(circuit_device_traits<Resistor>::is_linear == true);
        REQUIRE(circuit_device_traits<Capacitor>::is_linear == true);
        REQUIRE(circuit_device_traits<Inductor>::is_linear == true);
        REQUIRE(circuit_device_traits<VoltageSource>::is_linear == true);
        REQUIRE(circuit_device_traits<IdealDiode>::is_linear == false);
        REQUIRE(circuit_device_traits<MOSFET>::is_linear == false);
    }

    SECTION("History requirements") {
        REQUIRE(circuit_device_traits<Resistor>::needs_history == false);
        REQUIRE(circuit_device_traits<Capacitor>::needs_history == true);
        REQUIRE(circuit_device_traits<Inductor>::needs_history == true);
        REQUIRE(circuit_device_traits<VoltageSource>::needs_history == false);
    }

    SECTION("Switching classification") {
        REQUIRE(circuit_device_traits<Resistor>::is_switching == false);
        REQUIRE(circuit_device_traits<IdealSwitch>::is_switching == true);
        REQUIRE(circuit_device_traits<IdealDiode>::is_switching == true);
        REQUIRE(circuit_device_traits<MOSFET>::is_switching == true);
        REQUIRE(circuit_device_traits<IGBT>::is_switching == true);
    }
}

TEST_CASE("v2 Compile-time counting utilities", "[v2][circuit][counting]") {
    using namespace pulsim::v2;

    SECTION("Node counting") {
        static_assert(count_max_nodes_v<Resistor> == 2);
        static_assert(count_max_nodes_v<Resistor, Capacitor> == 4);
        static_assert(count_max_nodes_v<Resistor, Capacitor, VoltageSource> == 6);
        static_assert(count_max_nodes_v<MOSFET> == 3);
    }

    SECTION("Branch counting") {
        static_assert(count_branches_v<Resistor> == 0);
        static_assert(count_branches_v<VoltageSource> == 1);
        static_assert(count_branches_v<Inductor> == 1);
        static_assert(count_branches_v<Resistor, VoltageSource> == 1);
        static_assert(count_branches_v<VoltageSource, Inductor> == 2);
        static_assert(count_branches_v<Transformer> == 2);
    }

    SECTION("Linear device counting") {
        static_assert(count_linear_devices_v<Resistor> == 1);
        static_assert(count_linear_devices_v<Resistor, Capacitor> == 2);
        static_assert(count_linear_devices_v<Resistor, IdealDiode> == 1);
        static_assert(count_nonlinear_devices_v<Resistor, IdealDiode> == 1);
    }
}

TEST_CASE("v2 Circuit type queries", "[v2][circuit][queries]") {
    using namespace pulsim::v2;

    SECTION("Linear circuit check") {
        static_assert(is_linear_circuit<Resistor, Capacitor, VoltageSource>());
        static_assert(is_linear_circuit<Resistor, Inductor>());
        static_assert(!is_linear_circuit<Resistor, IdealDiode>());
        static_assert(!is_linear_circuit<Resistor, MOSFET>());
    }

    SECTION("Reactive elements check") {
        static_assert(!has_reactive_elements<Resistor, VoltageSource>());
        static_assert(has_reactive_elements<Resistor, Capacitor>());
        static_assert(has_reactive_elements<Resistor, Inductor>());
        static_assert(has_reactive_elements<Capacitor, Inductor>());
    }

    SECTION("Switching elements check") {
        static_assert(!has_switching_elements<Resistor, Capacitor>());
        static_assert(has_switching_elements<Resistor, IdealSwitch>());
        static_assert(has_switching_elements<IdealDiode, Resistor>());
        static_assert(has_switching_elements<MOSFET, Resistor>());
    }
}

TEST_CASE("v2 Topology validation", "[v2][circuit][validation]") {
    using namespace pulsim::v2;

    SECTION("RC circuit validation") {
        constexpr auto val = validate_circuit_topology<Resistor, Capacitor, VoltageSource>();
        REQUIRE(val.is_valid);
        REQUIRE(val.has_voltage_reference);
        REQUIRE(val.node_count == 6);  // Max nodes
        REQUIRE(val.branch_count == 1);  // VoltageSource branch
    }

    SECTION("RLC circuit validation") {
        constexpr auto val = validate_circuit_topology<Resistor, Inductor, Capacitor, VoltageSource>();
        REQUIRE(val.is_valid);
        REQUIRE(val.branch_count == 2);  // VoltageSource + Inductor branches
    }

    SECTION("Circuit with no voltage source") {
        constexpr auto val = validate_circuit_topology<Resistor, Capacitor>();
        REQUIRE(val.is_valid);  // Still valid, just no voltage reference
        REQUIRE_FALSE(val.has_voltage_reference);
    }
}

TEST_CASE("v2 CircuitGraph basic usage", "[v2][circuit][graph]") {
    using namespace pulsim::v2;

    SECTION("RC circuit graph creation") {
        Resistor r(1000.0);
        Capacitor c(1e-6, 0.0);
        VoltageSource vs(10.0);

        auto circuit = make_circuit(r, c, vs);

        REQUIRE(circuit.device_type_count == 3);
        REQUIRE(circuit.total_branches == 1);  // Only VoltageSource has branch
        REQUIRE(circuit.all_linear == true);
        REQUIRE(circuit.needs_history == true);  // Capacitor needs history
        REQUIRE(circuit.has_switching == false);
        REQUIRE(circuit.is_valid());
    }

    SECTION("Circuit with nonlinear device") {
        Resistor r(1000.0);
        IdealDiode d(1e3, 1e-9);
        VoltageSource vs(10.0);

        auto circuit = make_circuit(r, d, vs);

        REQUIRE(circuit.all_linear == false);
        REQUIRE(circuit.has_switching == true);
    }

    SECTION("Circuit with MOSFET") {
        Resistor r(1000.0);
        MOSFET m(2.0, 0.1, true);
        VoltageSource vdd(12.0);
        VoltageSource vgs(5.0);

        auto circuit = make_circuit(r, m, vdd, vgs);

        REQUIRE(circuit.device_type_count == 4);
        REQUIRE(circuit.total_branches == 2);  // Two voltage sources
        REQUIRE(circuit.all_linear == false);
        REQUIRE(circuit.has_switching == true);
    }
}

TEST_CASE("v2 CircuitGraph compile-time properties", "[v2][circuit][graph]") {
    using namespace pulsim::v2;

    SECTION("Static assertions on circuit properties") {
        // These are evaluated at compile time
        using RCCircuit = CircuitGraph<Resistor, Capacitor, VoltageSource>;
        static_assert(RCCircuit::device_type_count == 3);
        static_assert(RCCircuit::total_branches == 1);
        static_assert(RCCircuit::all_linear == true);
        static_assert(RCCircuit::needs_history == true);
        static_assert(RCCircuit::has_switching == false);

        using DiodeCircuit = CircuitGraph<Resistor, IdealDiode, VoltageSource>;
        static_assert(DiodeCircuit::all_linear == false);
        static_assert(DiodeCircuit::has_switching == true);

        using InductorCircuit = CircuitGraph<Resistor, Inductor, VoltageSource>;
        static_assert(InductorCircuit::total_branches == 2);  // VSource + Inductor
        static_assert(InductorCircuit::needs_history == true);
    }
}

TEST_CASE("v2 CircuitGraph sparsity pattern", "[v2][circuit][sparsity]") {
    using namespace pulsim::v2;

    SECTION("Circuit type-level sparsity") {
        // The sparsity pattern at type level gives capacity, not actual used entries
        using RVCircuit = CircuitGraph<Resistor, VoltageSource>;
        static_assert(RVCircuit::max_jacobian_nnz > 0);

        using RCVCircuit = CircuitGraph<Resistor, Capacitor, VoltageSource>;
        static_assert(RCVCircuit::max_jacobian_nnz > 0);
    }
}

TEST_CASE("v2 NodeCollector", "[v2][circuit][nodes]") {
    using namespace pulsim::v2;

    SECTION("Collect unique nodes") {
        NodeCollector<Resistor, Capacitor, VoltageSource> collector;

        // Add nodes manually (simulating device node collection)
        collector.add_node(0);
        collector.add_node(1);
        collector.add_node(0);  // Duplicate
        collector.add_node(2);

        REQUIRE(collector.node_count == 3);
        REQUIRE(collector.has_node(0));
        REQUIRE(collector.has_node(1));
        REQUIRE(collector.has_node(2));
        REQUIRE(!collector.has_node(3));
    }

    SECTION("Ground node not collected") {
        NodeCollector<Resistor> collector;

        collector.add_node(-1);  // Ground
        collector.add_node(0);

        REQUIRE(collector.node_count == 1);
        REQUIRE(!collector.has_node(-1));
        REQUIRE(collector.has_node(0));
    }

    SECTION("Node index lookup") {
        NodeCollector<Resistor, Capacitor> collector;

        collector.add_node(5);
        collector.add_node(10);
        collector.add_node(15);

        REQUIRE(collector.node_index(5) == 0);
        REQUIRE(collector.node_index(10) == 1);
        REQUIRE(collector.node_index(15) == 2);
    }
}

TEST_CASE("v2 StaticSparsityBuilder", "[v2][circuit][sparsity]") {
    using namespace pulsim::v2;

    SECTION("Conductance stamp pattern") {
        StaticSparsityBuilder<4, 2> builder;

        // Add conductance stamp between nodes 0 and 1
        builder.add_conductance(0, 1);

        const auto& pattern = builder.pattern();
        REQUIRE(pattern.contains(0, 0));
        REQUIRE(pattern.contains(0, 1));
        REQUIRE(pattern.contains(1, 0));
        REQUIRE(pattern.contains(1, 1));
    }

    SECTION("Voltage source stamp pattern") {
        StaticSparsityBuilder<4, 2> builder;

        // Add voltage source between node 0 and ground, branch index 4
        builder.add_voltage_source(0, -1, 4);

        const auto& pattern = builder.pattern();
        REQUIRE(pattern.contains(0, 4));
        REQUIRE(pattern.contains(4, 0));
    }

    SECTION("Inductor stamp pattern") {
        StaticSparsityBuilder<4, 2> builder;

        // Add inductor between nodes 0 and 1, branch index 4
        builder.add_inductor(0, 1, 4);

        const auto& pattern = builder.pattern();
        REQUIRE(pattern.contains(0, 4));
        REQUIRE(pattern.contains(4, 0));
        REQUIRE(pattern.contains(1, 4));
        REQUIRE(pattern.contains(4, 1));
        REQUIRE(pattern.contains(4, 4));  // Companion resistance
    }
}

// =============================================================================
// Static assertions for Phase 2.4 and 2.5 (compile-time verification)
// =============================================================================

namespace static_tests {
using namespace pulsim::v2;

// Expression templates compile-time tests
static_assert([]() constexpr {
    StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    StaticVector<double, 4> b{5.0, 6.0, 7.0, 8.0};
    auto expr = a + b;
    return expr[0] == 6.0 && expr[3] == 12.0;
}());

static_assert([]() constexpr {
    StaticVector<double, 4> a{1.0, 2.0, 3.0, 4.0};
    auto expr = 2.0 * a;
    return expr[0] == 2.0 && expr[3] == 8.0;
}());

// Circuit graph compile-time tests
static_assert(count_max_nodes_v<Resistor, Capacitor> == 4);
static_assert(count_branches_v<VoltageSource, Inductor> == 2);
static_assert(is_linear_circuit<Resistor, Capacitor>());
static_assert(!is_linear_circuit<Resistor, IdealDiode>());
static_assert(has_reactive_elements<Capacitor>());
static_assert(!has_switching_elements<Resistor>());

// Validation static tests
static_assert(validate_circuit_topology<Resistor, VoltageSource>().is_valid);
static_assert(validate_circuit_topology<Resistor, Capacitor, VoltageSource>().has_voltage_reference);

} // namespace static_tests
