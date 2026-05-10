// =============================================================================
// DC motor step response — separately excited.
//
// Build:
//   cmake -S examples/cpp -B build/examples
//   cmake --build build/examples --target pulsim_example_dc_motor_step
//   ./build/examples/pulsim_example_dc_motor_step
//
// Demonstrates:
//   - DcMotor first-order electrical + mechanical dynamics.
//   - Step response under fixed armature voltage and zero load torque.
//   - Closed-form steady-state speed (`steady_state_omega`) as the
//     analytical reference the simulation should converge to.
//   - Mechanical time constant — used to size the simulated horizon
//     (~ 5·τ_m settles within 1 % of asymptote).
//
// See also: docs/motor-models.md
// =============================================================================
#include "pulsim/v1/motors/dc_motor.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace pulsim::v1;
using motors::DcMotor;
using motors::DcMotorParams;

int main() {
    DcMotorParams p;
    p.name = "DC_24V_servo";
    p.R_a = 1.0;     // Ω
    p.L_a = 1e-3;    // H
    p.K_e = 0.05;    // V·s/rad   (back-EMF)
    p.K_t = 0.05;    // N·m/A     (torque)
    p.J   = 1e-4;    // kg·m²
    p.b   = 1e-5;    // N·m·s
    DcMotor motor(p);

    constexpr double Va       = 24.0;     // step input
    constexpr double tau_load = 0.0;      // no-load
    constexpr double dt       = 1e-4;     // 10 kHz
    constexpr int    N        = 5000;     // 0.5 s

    const double omega_ss_pred = motor.steady_state_omega(Va, tau_load);
    const double tau_m         = motor.mechanical_time_constant();
    std::cout << "DC motor parameters:\n"
              << "  R_a=" << p.R_a   << " Ω    L_a="   << p.L_a*1e3 << " mH\n"
              << "  K_e=" << p.K_e   << " V·s/rad     K_t=" << p.K_t << " N·m/A\n"
              << "  J  =" << p.J*1e3 << " g·m²        b   =" << p.b*1e6 << " µN·m·s\n";
    std::cout << "Step input: Va = " << Va << " V, tau_load = " << tau_load << "\n";
    std::cout << "Predicted ω_ss = " << omega_ss_pred << " rad/s ("
              << omega_ss_pred * 60.0 / (2 * 3.14159265) << " rpm)\n";
    std::cout << "Predicted τ_m  = " << tau_m * 1e3 << " ms\n\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "    t (s)   |   ω (rad/s)   |   i_a (A)   |  ω/ω_ss\n";
    std::cout << "------------+---------------+-------------+--------\n";
    for (int k = 0; k <= N; ++k) {
        motor.step(Va, tau_load, dt);
        if (k % 250 == 0) {
            std::cout << std::setw(10) << k * dt << "  | "
                      << std::setw(13) << motor.omega() << " | "
                      << std::setw(11) << motor.i_a() << " | "
                      << std::setw(7) << motor.omega() / omega_ss_pred << "\n";
        }
    }
    std::cout << "\nFinal ω = " << motor.omega() << " rad/s vs predicted "
              << omega_ss_pred << " rad/s    "
              << "(error " << 100.0 * (motor.omega() - omega_ss_pred) / omega_ss_pred
              << " %)\n";
    return 0;
}
