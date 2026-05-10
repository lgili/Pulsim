#pragma once

#include "pulsim/v1/motors/mechanical.hpp"
#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::motors {

// =============================================================================
// add-motor-models — PMSM in dq frame (Phase 3)
// =============================================================================
//
// Permanent-magnet synchronous machine (PMSM) modeled in the rotor
// reference frame (dq). Stator electrical equations:
//
//   v_d = R_s · i_d + L_d · di_d/dt - ω_e · L_q · i_q
//   v_q = R_s · i_q + L_q · di_q/dt + ω_e · (L_d · i_d + ψ_PM)
//
// Electromagnetic torque (with `pole_pairs` p):
//
//   τ_em = (3/2) · p · (ψ_PM · i_q + (L_d - L_q) · i_d · i_q)
//
// Mechanical port from `Shaft`:
//   J · dω_m/dt = τ_em - τ_load - b · ω_m
//   ω_e = p · ω_m
//
// State variables: (i_d, i_q, ω_m, θ_m).

struct PmsmParams {
    std::string name;

    // Electrical parameters
    Real Rs           = 0.5;       ///< Ω — stator resistance
    Real Ld           = 5e-3;      ///< H — d-axis inductance
    Real Lq           = 5e-3;      ///< H — q-axis inductance (Lq > Ld for IPM)
    Real psi_pm       = 0.1;       ///< Wb — flux linkage from permanent magnet
    int  pole_pairs   = 2;         ///< number of pole pairs (poles / 2)

    // Mechanical parameters
    Real J            = 1e-3;      ///< kg·m² — rotor inertia (excluding load)
    Real b_friction   = 1e-4;      ///< N·m·s — viscous friction
    Real friction_coulomb = 0.0;   ///< N·m

    // Initial state
    Real i_d_init     = 0.0;
    Real i_q_init     = 0.0;
    Real omega_init   = 0.0;       ///< mechanical (rad/s)
    Real theta_init   = 0.0;       ///< mechanical (rad)
};

class Pmsm {
public:
    Pmsm() = default;
    explicit Pmsm(PmsmParams params)
        : params_(std::move(params)),
          i_d_(params_.i_d_init),
          i_q_(params_.i_q_init),
          omega_m_(params_.omega_init),
          theta_m_(params_.theta_init) {}

    /// Single-step advance under (V_d, V_q) electrical inputs and
    /// `tau_load` mechanical load. Forward-Euler discretization of the
    /// stator currents and shaft equation.
    ///
    /// Returns the electromagnetic torque produced this step (useful
    /// for telemetry / control-loop feedback).
    Real step(Real Vd, Real Vq, Real tau_load, Real dt) {
        // Electrical angular velocity
        const Real omega_e = static_cast<Real>(params_.pole_pairs) * omega_m_;

        // Stator current derivatives (decoupled-Park PMSM model)
        const Real didt =
            (Vd - params_.Rs * i_d_ + omega_e * params_.Lq * i_q_) / params_.Ld;
        const Real diqt =
            (Vq - params_.Rs * i_q_ -
             omega_e * (params_.Ld * i_d_ + params_.psi_pm)) / params_.Lq;

        i_d_ += dt * didt;
        i_q_ += dt * diqt;

        // Electromagnetic torque (3/2 · p · [ψ_PM·iq + (Ld-Lq)·id·iq])
        const Real tau_em =
            Real{1.5} * static_cast<Real>(params_.pole_pairs) *
            (params_.psi_pm * i_q_ +
             (params_.Ld - params_.Lq) * i_d_ * i_q_);

        // Shaft acceleration
        const Real tau_friction =
            params_.b_friction * omega_m_ +
            params_.friction_coulomb *
                (omega_m_ > Real{0} ? Real{1} :
                 omega_m_ < Real{0} ? Real{-1} : Real{0});
        omega_m_ += dt * (tau_em - tau_load - tau_friction) / params_.J;
        theta_m_ += dt * omega_m_;

        return tau_em;
    }

    /// Open-circuit voltage (back-EMF) at the current speed. Equal to
    /// `psi_pm · ω_e` per phase (peak), used for the no-load gate.
    [[nodiscard]] Real back_emf_peak() const noexcept {
        const Real omega_e = static_cast<Real>(params_.pole_pairs) * omega_m_;
        return params_.psi_pm * omega_e;
    }

    /// Electrical angle θ_e = p · θ_m, used by Park transforms when
    /// converting between (a, b, c) phase and (d, q) rotor frames.
    [[nodiscard]] Real theta_electrical() const noexcept {
        return static_cast<Real>(params_.pole_pairs) * theta_m_;
    }

    /// Direct accessors.
    [[nodiscard]] Real i_d()      const noexcept { return i_d_; }
    [[nodiscard]] Real i_q()      const noexcept { return i_q_; }
    [[nodiscard]] Real omega_m()  const noexcept { return omega_m_; }
    [[nodiscard]] Real theta_m()  const noexcept { return theta_m_; }
    [[nodiscard]] const PmsmParams& params() const noexcept { return params_; }

    void set_state(Real i_d, Real i_q, Real omega_m, Real theta_m) {
        i_d_ = i_d; i_q_ = i_q; omega_m_ = omega_m; theta_m_ = theta_m;
    }

private:
    PmsmParams params_{};
    Real i_d_     = 0.0;
    Real i_q_     = 0.0;
    Real omega_m_ = 0.0;          // mechanical angular velocity (rad/s)
    Real theta_m_ = 0.0;          // mechanical angle (rad)
};

}  // namespace pulsim::v1::motors
