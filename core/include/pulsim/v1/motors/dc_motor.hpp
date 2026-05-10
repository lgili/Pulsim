#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::motors {

// =============================================================================
// add-motor-models — separately-excited DC motor (Phase 5.2)
// =============================================================================
//
// First-principles separately-excited DC motor:
//
//   v_a = R_a · i_a + L_a · di_a/dt + K_e · ω
//   τ_em = K_t · i_a
//   J · dω/dt = τ_em - τ_load - b · ω
//
// `K_e` (V·s/rad) and `K_t` (N·m/A) are equal in SI; the model accepts
// them as separate fields so users can dial small empirical mismatches
// without fighting the convention.

struct DcMotorParams {
    std::string name;
    Real R_a   = 1.0;        ///< Ω — armature resistance
    Real L_a   = 1e-3;       ///< H — armature inductance
    Real K_e   = 0.05;       ///< V·s/rad — back-EMF constant
    Real K_t   = 0.05;       ///< N·m/A — torque constant
    Real J     = 1e-4;       ///< kg·m² — rotor inertia
    Real b     = 1e-5;       ///< N·m·s — viscous friction

    Real i_a_init   = 0.0;
    Real omega_init = 0.0;
    Real theta_init = 0.0;
};

class DcMotor {
public:
    DcMotor() = default;
    explicit DcMotor(DcMotorParams params)
        : params_(std::move(params)),
          i_a_(params_.i_a_init),
          omega_(params_.omega_init),
          theta_(params_.theta_init) {}

    /// One forward-Euler step under armature voltage `Va` and load
    /// torque `tau_load`. Returns the electromagnetic torque.
    Real step(Real Va, Real tau_load, Real dt) {
        const Real didt =
            (Va - params_.R_a * i_a_ - params_.K_e * omega_) / params_.L_a;
        i_a_ += dt * didt;
        const Real tau_em = params_.K_t * i_a_;
        omega_ += dt * (tau_em - tau_load - params_.b * omega_) / params_.J;
        theta_ += dt * omega_;
        return tau_em;
    }

    /// First-order steady-state speed under constant voltage / load:
    ///   ω_ss = (V·K_t - τ_load·R_a) / (K_t·K_e + b·R_a)
    /// Closed-form prediction the no-load step-response test compares
    /// against.
    [[nodiscard]] Real steady_state_omega(Real Va, Real tau_load) const {
        const Real num   = Va * params_.K_t - tau_load * params_.R_a;
        const Real denom = params_.K_t * params_.K_e + params_.b * params_.R_a;
        return num / denom;
    }

    /// Mechanical time constant τ_m ≈ J · R_a / (K_t · K_e). Used to
    /// validate the speed step-response shape.
    [[nodiscard]] Real mechanical_time_constant() const {
        return params_.J * params_.R_a / (params_.K_t * params_.K_e);
    }

    [[nodiscard]] Real i_a()    const noexcept { return i_a_; }
    [[nodiscard]] Real omega()  const noexcept { return omega_; }
    [[nodiscard]] Real theta()  const noexcept { return theta_; }
    [[nodiscard]] const DcMotorParams& params() const noexcept { return params_; }

private:
    DcMotorParams params_{};
    Real i_a_   = 0.0;
    Real omega_ = 0.0;
    Real theta_ = 0.0;
};

}  // namespace pulsim::v1::motors
