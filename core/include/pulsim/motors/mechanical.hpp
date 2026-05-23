// SPDX-License-Identifier: MIT
//
// Pulsim — Phase D / C++ port: shared rotational-mechanical base
// for all motor models.
//
// Forward-Euler integration of J·dω/dt = T_em − T_load − B·ω, plus
// θ = ∫ω dt. Both ω and θ are mutable members updated per simulation
// step by the motor's chain block.

#pragma once

#include "pulsim/numeric/types.hpp"

#include <functional>

namespace pulsim::motors {

struct Mechanical {
    Real J_kgm2 = Real{1e-6};               // rotor inertia
    Real B_Nms_per_rad = Real{0};           // viscous friction
    Real T_load_Nm = Real{0};               // constant load
    // Optional callback for speed-dependent load (fans, pumps).
    // Signature: (t, omega) -> Nm.
    std::function<Real(Real, Real)> T_load_fn{};

    // Live state.
    Real omega_rad_s = Real{0};
    Real theta_rad   = Real{0};

    void reset() noexcept {
        omega_rad_s = Real{0};
        theta_rad   = Real{0};
    }

    void integrate(Real t, Real T_em_Nm, Real dt) {
        Real T_load = T_load_Nm;
        if (T_load_fn) {
            T_load = T_load_fn(t, omega_rad_s);
        }
        const Real denom = (J_kgm2 > Real{1e-30}) ? J_kgm2 : Real{1e-30};
        const Real domega = (T_em_Nm - T_load -
                                B_Nms_per_rad * omega_rad_s) /
                              denom * dt;
        omega_rad_s += domega;
        theta_rad   += omega_rad_s * dt;
    }
};

}  // namespace pulsim::motors
