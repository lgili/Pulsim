// =============================================================================
// Phase B3 of harden-component-models-vs-psim-plecs: motor winding
// steady-state thermal accessors.
// =============================================================================
//
// Adds API-surface tests for the per-motor `*_steady_state_winding_temperature`
// accessors. They solve the implicit thermal balance
//
//     T_w = T_amb + n_phases · I_rms² · R_s(T_w) · R_th
//     R_s(T_w) = R_s · (1 + R_s_tc · (T_w − T_ref))
//
// in closed form (linear-in-x where x = T_w − T_ref). The accessors return
// NaN when:
//   * the motor is not found
//   * the thermal model is disabled (R_th_winding_to_ambient = 0)
//   * the model becomes unstable (I·I·R·R_tc·R_th ≥ 1 — physical runaway)
//
// These are pure-formula tests; they don't run a transient. The actual
// R_s scaling in the stamp path is deferred to a follow-up — for now,
// users sample I_rms post-simulation and query the steady-state T_w.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "pulsim/v1/runtime_circuit.hpp"

#include <cmath>

using namespace pulsim::v1;
using Catch::Approx;

TEST_CASE("Motor winding thermal: DC motor steady-state T_w matches "
          "closed-form solution",
          "[v1][motor][thermal][b3]") {
    Circuit circuit;
    const Index n_pos = circuit.add_node("a+");
    motors::DcMotorParams p;
    p.R_a = 1.0;
    p.K_e = 0.05;
    p.K_t = 0.05;
    p.R_th_winding_to_ambient = 5.0;    // K/W
    p.T_amb = 25.0;
    p.R_s_tc = 3.93e-3;
    p.T_ref_winding = 20.0;
    circuit.add_dc_motor("M1", n_pos, Circuit::ground(), p);

    // Cold case: I_rms = 0 → T_w = T_amb.
    CHECK(circuit.dc_motor_steady_state_winding_temperature("M1", 0.0)
          == Approx(25.0).margin(1e-9));

    // Loaded case: I_rms = 1 A. Closed-form:
    //   power_factor = 1 · 1² · 1.0 · 5.0 = 5.0
    //   denom        = 1 − 5.0 · 3.93e-3 ≈ 0.98035
    //   T_w − T_ref  = (25 − 20 + 5) / 0.98035 ≈ 10.2002
    //   T_w          ≈ 30.20 °C
    CHECK(circuit.dc_motor_steady_state_winding_temperature("M1", 1.0)
          == Approx(30.2).margin(0.01));
}

TEST_CASE("Motor winding thermal: PMSM uses 1.5× phase factor",
          "[v1][motor][thermal][b3]") {
    Circuit circuit;
    const Index n_a = circuit.add_node("a");
    const Index n_b = circuit.add_node("b");
    const Index n_c = circuit.add_node("c");
    motors::PmsmParams p;
    p.Rs = 0.5;
    p.Ld = p.Lq = 5e-3;
    p.psi_pm = 0.1;
    p.pole_pairs = 2;
    p.R_th_winding_to_ambient = 2.0;
    p.T_amb = 25.0;
    p.R_s_tc = 3.93e-3;
    p.T_ref_winding = 20.0;
    circuit.add_pmsm("M_pmsm", n_a, n_b, n_c, Circuit::ground(), p);

    // I_s_rms = 10 A:
    //   power_factor = 1.5 · 100 · 0.5 · 2.0 = 150 W·K/W
    //   denom        = 1 − 150 · 3.93e-3 ≈ 0.4105
    //   T_w          = 20 + (25 − 20 + 150) / 0.4105 ≈ 397.5 °C (very hot!)
    const Real T_w = circuit.pmsm_steady_state_winding_temperature("M_pmsm", 10.0);
    INFO("PMSM T_w = " << T_w << " °C (at 10 A_rms)");
    CHECK(std::isfinite(T_w));
    CHECK(T_w > 300.0);   // hot
    CHECK(T_w < 500.0);   // not runaway
}

TEST_CASE("Motor winding thermal: thermal model disabled returns NaN",
          "[v1][motor][thermal][b3]") {
    Circuit circuit;
    const Index n_pos = circuit.add_node("a+");
    motors::DcMotorParams p;
    p.R_a = 1.0;
    p.K_e = 0.05;
    p.K_t = 0.05;
    // R_th_winding_to_ambient stays at default 0 → model disabled.
    circuit.add_dc_motor("M_no_thermal", n_pos, Circuit::ground(), p);

    const Real T_w = circuit.dc_motor_steady_state_winding_temperature(
        "M_no_thermal", 1.0);
    CHECK(std::isnan(T_w));
}

TEST_CASE("Motor winding thermal: missing motor returns NaN",
          "[v1][motor][thermal][b3]") {
    Circuit circuit;
    const Real T_w = circuit.dc_motor_steady_state_winding_temperature(
        "does_not_exist", 1.0);
    CHECK(std::isnan(T_w));
}

TEST_CASE("Motor winding thermal: physical runaway (denom ≤ 0) returns NaN",
          "[v1][motor][thermal][b3]") {
    Circuit circuit;
    const Index n_pos = circuit.add_node("a+");
    motors::DcMotorParams p;
    p.R_a = 1.0;
    p.K_e = 0.05;
    p.K_t = 0.05;
    // Push the linear thermal model past stability: R_th huge so that
    // I²·R_s·R_s_tc·R_th > 1. With I=10, R_a=1, R_s_tc=3.93e-3, this
    // requires R_th > 1/(100·1·3.93e-3) ≈ 2.54 K/W. Use 10.
    p.R_th_winding_to_ambient = 10.0;
    p.T_amb = 25.0;
    p.R_s_tc = 3.93e-3;
    p.T_ref_winding = 20.0;
    circuit.add_dc_motor("M_runaway", n_pos, Circuit::ground(), p);

    const Real T_w = circuit.dc_motor_steady_state_winding_temperature(
        "M_runaway", 100.0);
    INFO("T_w = " << T_w);
    CHECK(std::isnan(T_w));
}
