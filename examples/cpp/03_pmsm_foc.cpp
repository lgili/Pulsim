// =============================================================================
// PMSM under field-oriented control — id/iq current loop.
//
// Build:
//   cmake -S examples/cpp -B build/examples
//   cmake --build build/examples --target pulsim_example_pmsm_foc
//   ./build/examples/pulsim_example_pmsm_foc
//
// Demonstrates:
//   - PMSM electrical model in the dq frame.
//   - PmsmFocCurrentLoop with pole-zero-cancellation tuning
//     (Kp = ω_c · L_axis,  Ki = Kp · R_s / L_axis).
//   - Step response on i_q_ref while i_d_ref = 0 (SPM convention).
//   - Verify both loops decouple and settle within 1 / bandwidth_hz.
//
// See also: docs/motor-models.md
// =============================================================================
#include "pulsim/v1/motors/pmsm.hpp"
#include "pulsim/v1/motors/pmsm_foc.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace pulsim::v1;
using motors::PmsmParams;
using motors::Pmsm;
using motors::PmsmFocCurrentLoop;
using motors::PmsmFocCurrentLoopParams;

int main() {
    // Surface-mounted PMSM, balanced 3φ.
    PmsmParams motor;
    motor.Rs = 0.5;                 // Ω/phase
    motor.Ld = 4e-3;                // H
    motor.Lq = 4e-3;                // H — SPM ⇒ Ld = Lq
    motor.psi_pm = 0.08;            // V·s — permanent-magnet flux linkage
    motor.pole_pairs = 4;
    motor.J = 1e-4;                 // kg·m²
    motor.b_friction = 1e-2;        // strong viscous friction so the rotor
                                    // settles at moderate speed and Vq
                                    // doesn't saturate against back-EMF.
    Pmsm machine(motor);

    PmsmFocCurrentLoopParams foc;
    foc.bandwidth_hz = 1500.0;      // current-loop crossover
    foc.Vd_min = -200.0;
    foc.Vd_max =  200.0;
    foc.Vq_min = -200.0;
    foc.Vq_max =  200.0;
    PmsmFocCurrentLoop loop(motor, foc);

    constexpr double dt    = 5e-5;     // 20 kHz control loop
    constexpr int    N     = 4000;     // 0.2 s
    constexpr double iq_ref = 2.0;     // step torque demand (small enough that
                                       // Vq doesn't saturate at speed)
    constexpr double id_ref = 0.0;     // SPM: id_ref ≡ 0
    constexpr double tau_load = 0.01;  // light load — keeps iq tracking

    std::cout << "PMSM-FOC current-loop step (id_ref=0, iq_ref=" << iq_ref << " A):\n"
              << "  motor: Rs=" << motor.Rs << " Ω    Ld=Lq=" << motor.Ld*1e3 << " mH\n"
              << "  loop:  bandwidth = " << foc.bandwidth_hz << " Hz  "
              << "(expected rise time ≈ " << 1e3/foc.bandwidth_hz << " ms)\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "    t (ms)  |   id (A)   |   iq (A)   |   Vd (V)   |   Vq (V)   |  ω (rad/s)\n";
    std::cout << "------------+------------+------------+------------+------------+-----------\n";

    for (int k = 0; k <= N; ++k) {
        const Real id_meas = machine.i_d();
        const Real iq_meas = machine.i_q();
        const auto [Vd, Vq] = loop.step(id_ref, iq_ref, id_meas, iq_meas, dt);
        // PMSM step advances both electrical and mechanical states.
        machine.step(Vd, Vq, tau_load, dt);

        if (k % 200 == 0) {
            std::cout << std::setw(10) << k * dt * 1e3 << "  | "
                      << std::setw(10) << machine.i_d() << " | "
                      << std::setw(10) << machine.i_q() << " | "
                      << std::setw(10) << Vd << " | "
                      << std::setw(10) << Vq << " | "
                      << std::setw(9)  << machine.omega_m() << "\n";
        }
    }

    std::cout << "\nFinal:  id = " << machine.i_d() << " A (target " << id_ref << ")    "
              << "iq = " << machine.i_q() << " A (target " << iq_ref << ")\n";
    std::cout << "        ω_m = " << machine.omega_m() << " rad/s "
              << "(" << machine.omega_m()*60/(2*3.14159265) << " rpm)\n";
    return 0;
}
