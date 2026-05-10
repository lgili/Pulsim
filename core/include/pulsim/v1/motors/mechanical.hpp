#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::motors {

// =============================================================================
// add-motor-models — mechanical primitives (Phase 1)
// =============================================================================
//
// Motor models couple electrical state (currents, voltages) to a
// mechanical port (torque τ, angular velocity ω). Phase 1 ships the
// pure-mechanical pieces — shaft, gearbox, load profiles — so the
// motor models in Phases 3 / 5 just plug a torque source into a shaft
// and let the math compose.
//
// All primitives are header-only math objects (no Circuit-side
// integration today; that lands once `Circuit::DeviceVariant` accepts
// motor / mechanical entries in the next change). State advances
// happen through `step(dt)` calls — the time-domain simulator doing
// transient analysis can drive these in a callback.

/// Newton's second law on a rotating shaft:
///   J · dω/dt = τ_input - τ_load - b·ω - τ_coulomb·sign(ω)
struct Shaft {
    Real J = 1e-3;                ///< kg·m² — rotor + load inertia
    Real b_friction = 0.0;        ///< N·m·s — viscous-friction coefficient
    Real friction_coulomb = 0.0;  ///< N·m — Coulomb friction (constant, sign(ω))
    Real omega = 0.0;             ///< current angular velocity (rad/s)

    /// Forward-Euler advance under net torque τ_net = τ_input - τ_load.
    /// (Trapezoidal would be marginally more accurate but the explicit
    /// step is what closed-loop FOC simulations expect at typical
    /// control-loop dt's of 50–100 µs.)
    void advance(Real tau_net, Real dt) {
        const Real tau_friction =
            b_friction * omega +
            friction_coulomb * (omega > Real{0} ? Real{1} :
                                omega < Real{0} ? Real{-1} : Real{0});
        omega += dt * (tau_net - tau_friction) / J;
    }
};

/// Ideal gearbox between two shafts. `ratio` = ω_in / ω_out.
/// `efficiency` ∈ (0, 1] — torque transfer scaled by η on the output side.
struct GearBox {
    Real ratio      = 1.0;     ///< speed ratio (ω_in / ω_out)
    Real efficiency = 1.0;     ///< 0 < η ≤ 1

    /// Speed conversion: input ω → output ω.
    [[nodiscard]] Real omega_out(Real omega_in) const {
        return omega_in / ratio;
    }
    /// Torque conversion: input τ → output τ (after losses).
    [[nodiscard]] Real torque_out(Real tau_in) const {
        return tau_in * ratio * efficiency;
    }
    /// Reflect a load torque from the output side back to the input side.
    [[nodiscard]] Real reflect_load(Real tau_load_out) const {
        if (efficiency <= Real{0}) return Real{0};
        return tau_load_out / (ratio * efficiency);
    }
};

/// Constant-torque load — the textbook Newton-cradle load.
struct ConstantTorqueLoad {
    Real torque = 0.0;            ///< N·m, opposes motion (positive ω → +τ_load)
    [[nodiscard]] Real load_torque(Real /*omega*/) const noexcept {
        return torque;
    }
};

/// Quadratic / fan / propeller load: τ_load = k · ω · |ω|.
/// Sign convention: load opposes motion (positive ω → positive τ_load).
struct FanLoad {
    Real k = 1e-4;                ///< N·m / (rad/s)²
    [[nodiscard]] Real load_torque(Real omega) const noexcept {
        return k * omega * std::abs(omega);
    }
};

/// Pure-inertia load (no torque draw, just adds to shaft inertia
/// when summed). The shaft's `J` already covers the motor side; this
/// is for downstream mechanical chains where the load adds inertia
/// distinct from friction.
struct FlywheelLoad {
    Real J_extra = 0.0;           ///< kg·m²
    [[nodiscard]] Real load_torque(Real /*omega*/) const noexcept {
        return Real{0};            // pure inertia, no resistive torque
    }
};

}  // namespace pulsim::v1::motors
