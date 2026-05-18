#pragma once

#include "pulsim/v1/motors/mechanical.hpp"
#include "pulsim/v1/numeric_types.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::motors {

// =============================================================================
// consolidate-motors-and-three-phase — BLDC (Phase C.1)
// =============================================================================
//
// Brushless DC machine with trapezoidal back-EMF per phase (120° flat-top).
// The non-salient (L_d = L_q = L_s) abc-frame model is:
//
//   v_k − v_N = R_s · i_k + L_s · di_k/dt + e_k(θ_e, ω_m)         (k = a,b,c)
//
// where the back-EMF for phase k is the textbook "180° trapezoidal" shape:
//
//                        f_a(θ_e)            (per-unit, ∈ [-1, +1])
//                          +1 ─┐
//                              │
//   ┌──[0°, 120°]──┐           │           ┌──[300°, 360°]──┐
//   │   flat +1    │           ╲           │  ramp -1 → +1  │
//   └──────────────┘            ╲          └────────────────┘
//                                ╲
//                       ramp +1 → -1
//                       on [120°, 180°]
//                                  ╲
//                                   ╲
//                          ┌──[180°, 300°]──┐
//                          │    flat -1     │
//                          └────────────────┘
//                          -1 ─────────────────
//
//   e_k(θ_e, ω_m) = K_e_peak · f_k(θ_e) · ω_e          (V)
//   f_b(θ_e)      = f_a(θ_e - 2π/3)
//   f_c(θ_e)      = f_a(θ_e - 4π/3)
//   θ_e           = p · θ_m,         ω_e = p · ω_m
//
// Electromagnetic torque. Using the energy balance ∑ e_k · i_k = T_em · ω_m
// (mechanical power = electrical-back-EMF power), we get:
//
//   T_em = (∑_k e_k · i_k) / ω_m
//        = K_e_peak · ω_e · (∑_k f_k(θ_e) · i_k) / ω_m
//        = K_e_peak · p   · (∑_k f_k(θ_e) · i_k)
//
// The trailing `p` (pole_pairs) factor makes T_em independent of ω_m and
// finite at standstill, which is what the simulator needs. For the canonical
// six-step commutation pattern with two phases conducting at flat-top
// (i_k = ±I on the +1 / −1 plateaus, third phase floating), this collapses
// to the well-known T_em = 2 · K_e_peak · I · p result.
//
// Mechanical port: standard Newton-Euler shaft model identical to
// `motors::Shaft` (used by the PMSM as well):
//
//   J · dω_m/dt = T_em − τ_load_total − b · ω_m − sign(ω_m) · τ_C
//   θ_m_new    = θ_m + dt · ω_m                            (forward-Euler)
//
// `τ_load_total` aggregates the externally-set `tau_load_external_` and the
// configured constant / quadratic load terms from the params struct, so
// fan-like and propeller-like loads compose cleanly:
//
//   τ_load_total = tau_load_external_ + tau_load_const
//                + tau_load_quad_coeff · ω_m · |ω_m|

struct BldcMotorParams {
    std::string name;

    // Electrical parameters (per phase, balanced 3φ)
    Real R_s             = 0.5;        ///< Ω — stator phase resistance
    Real L_s             = 1e-3;       ///< H — stator phase inductance (non-salient)
    Real K_e_peak        = 0.05;       ///< V·s/rad — peak back-EMF constant per phase
                                       ///<  (e_k_peak = K_e_peak · ω_e at flat-top)
    int  pole_pairs      = 2;          ///< electrical-to-mechanical ratio

    // Mechanical parameters
    Real J               = 1e-4;       ///< kg·m² — rotor inertia
    Real b_friction      = 1e-5;       ///< N·m·s — viscous friction
    Real friction_coulomb = 0.0;       ///< N·m — Coulomb / static friction

    // Aggregated mechanical load profile (composes with set_load_torque()).
    Real tau_load_const     = 0.0;     ///< N·m — constant opposing torque
    Real tau_load_quad_coeff = 0.0;    ///< N·m / (rad/s)² — fan/propeller load

    // Initial state
    Real omega_init      = 0.0;        ///< mechanical (rad/s)
    Real theta_init      = 0.0;        ///< mechanical (rad)
};

class BldcMotor {
public:
    BldcMotor() = default;

    explicit BldcMotor(BldcMotorParams params, std::string name = "")
        : params_(std::move(params)),
          name_(std::move(name)),
          omega_m_(params_.omega_init),
          theta_m_(params_.theta_init) {}

    // ---------------------------------------------------------------------
    // Back-EMF / torque queries
    // ---------------------------------------------------------------------

    /// Per-phase back-EMF profile (per-unit, ∈ [-1, +1]). Index 0 = phase A,
    /// 1 = phase B (shifted -2π/3), 2 = phase C (shifted -4π/3 ≡ +2π/3).
    [[nodiscard]] Real back_emf_profile(int phase_index, Real theta_m) const {
        const Real theta_e = electrical_angle(theta_m);
        const Real shift = phase_shift_for(phase_index);
        return trapezoidal_profile(theta_e - shift);
    }

    /// Per-phase back-EMF voltage [V] at angle `theta_m` with the motor's
    /// current `omega_m_`. The result is `K_e_peak · f_k(θ_e) · ω_e`.
    [[nodiscard]] Real back_emf_phase(int phase_index, Real theta_m) const {
        const Real omega_e = static_cast<Real>(params_.pole_pairs) * omega_m_;
        return params_.K_e_peak * back_emf_profile(phase_index, theta_m) * omega_e;
    }

    /// Electromagnetic torque [N·m] at the given phase currents and rotor
    /// angle. Independent of ω_m by construction (see header math notes).
    [[nodiscard]] Real electromagnetic_torque(Real i_a, Real i_b, Real i_c,
                                              Real theta_m) const {
        const Real f_a = back_emf_profile(0, theta_m);
        const Real f_b = back_emf_profile(1, theta_m);
        const Real f_c = back_emf_profile(2, theta_m);
        const Real p = static_cast<Real>(params_.pole_pairs);
        return params_.K_e_peak * p * (f_a * i_a + f_b * i_b + f_c * i_c);
    }

    // ---------------------------------------------------------------------
    // Mechanical advance
    // ---------------------------------------------------------------------

    /// Advance ω_m and θ_m by `dt` under the phase currents (i_a, i_b, i_c).
    /// Forward-Euler integration of the Newton-Euler shaft equation. Mirrors
    /// the `motors::Shaft::advance` semantics. The external load torque is
    /// the sum of `set_load_torque(...)` plus the configured profile terms.
    void advance(Real i_a, Real i_b, Real i_c, Real dt) {
        const Real tau_em = electromagnetic_torque(i_a, i_b, i_c, theta_m_);
        const Real tau_load_total = total_load_torque();

        const Real sign_omega = (omega_m_ > Real{0}) ? Real{1.0}
                              : (omega_m_ < Real{0}) ? Real{-1.0}
                              :                        Real{0.0};
        const Real tau_friction =
            params_.b_friction * omega_m_ +
            params_.friction_coulomb * sign_omega;

        const Real J_safe = std::max<Real>(params_.J, Real{1e-30});
        const Real domega = dt * (tau_em - tau_load_total - tau_friction) / J_safe;

        // Note: θ_m advances with the OLD ω_m (semi-implicit Euler is fine
        // here — same convention as motors::Shaft and DcMotorDevice).
        theta_m_ += dt * omega_m_;
        omega_m_ += domega;
    }

    // ---------------------------------------------------------------------
    // Accessors / setters
    // ---------------------------------------------------------------------

    [[nodiscard]] Real omega_m()      const noexcept { return omega_m_; }
    [[nodiscard]] Real theta_m()      const noexcept { return theta_m_; }
    [[nodiscard]] Real omega_electrical() const noexcept {
        return static_cast<Real>(params_.pole_pairs) * omega_m_;
    }
    [[nodiscard]] Real theta_electrical(Real theta_m) const noexcept {
        return electrical_angle(theta_m);
    }
    [[nodiscard]] const BldcMotorParams& params() const noexcept { return params_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    void set_omega(Real w) noexcept { omega_m_ = w; }
    void set_theta(Real t) noexcept { theta_m_ = t; }
    void set_state(Real omega_m, Real theta_m) noexcept {
        omega_m_ = omega_m;
        theta_m_ = theta_m;
    }

    /// External load torque [N·m]. Composes with `tau_load_const` and
    /// `tau_load_quad_coeff · ω_m · |ω_m|` from the params struct.
    void set_load_torque(Real tau) noexcept { tau_load_external_ = tau; }
    [[nodiscard]] Real load_torque() const noexcept { return tau_load_external_; }

    /// Aggregated load torque (external + constant + quadratic profile).
    [[nodiscard]] Real total_load_torque() const noexcept {
        return tau_load_external_
             + params_.tau_load_const
             + params_.tau_load_quad_coeff * omega_m_ * std::abs(omega_m_);
    }

private:
    // -- Helpers --

    [[nodiscard]] Real electrical_angle(Real theta_m) const noexcept {
        return static_cast<Real>(params_.pole_pairs) * theta_m;
    }

    [[nodiscard]] static Real phase_shift_for(int phase_index) noexcept {
        // Standard 120° per-phase offset. Phase A = 0, B = +2π/3, C = +4π/3
        // (so f_b(θ_e) = f_a(θ_e − 2π/3) and f_c(θ_e) = f_a(θ_e − 4π/3)).
        constexpr Real two_pi_over_3  = Real{2.0943951023931953};
        constexpr Real four_pi_over_3 = Real{4.1887902047863905};
        switch (phase_index) {
            case 0:  return Real{0};
            case 1:  return two_pi_over_3;
            case 2:  return four_pi_over_3;
            default: return Real{0};  // out-of-range guards against UB
        }
    }

    /// Trapezoidal back-EMF profile, normalized to ±1. Input may be ANY
    /// real (positive or negative); the result is periodic with 2π.
    [[nodiscard]] static Real trapezoidal_profile(Real theta_e) noexcept {
        constexpr Real two_pi = Real{6.283185307179586};

        // Wrap θ_e into [0, 2π).
        Real x = std::fmod(theta_e, two_pi);
        if (x < Real{0}) x += two_pi;

        // Six 60° segments. Angles in degrees for clarity:
        //   [  0, 120) : +1                        flat-top
        //   [120, 180) : +1 − 2·(x−120)/60        linear ramp +1 → -1
        //   [180, 300) : -1                        flat-bottom
        //   [300, 360) : -1 + 2·(x−300)/60        linear ramp -1 → +1
        //
        // Using radians: 60° = π/3, 120° = 2π/3, 180° = π,
        //                300° = 5π/3,  360° = 2π.
        constexpr Real pi_over_3      = Real{1.0471975511965976};   // 60°
        constexpr Real two_pi_over_3  = Real{2.0943951023931953};   // 120°
        constexpr Real pi             = Real{3.141592653589793};    // 180°
        constexpr Real five_pi_over_3 = Real{5.235987755982989};    // 300°

        if (x < two_pi_over_3) {
            return Real{1};
        }
        if (x < pi) {
            // Ramp +1 at 120° down to -1 at 180°.
            const Real u = (x - two_pi_over_3) / pi_over_3;  // 0..1
            return Real{1} - Real{2} * u;
        }
        if (x < five_pi_over_3) {
            return Real{-1};
        }
        // x ∈ [300°, 360°): ramp -1 at 300° up to +1 at 360°.
        const Real u = (x - five_pi_over_3) / pi_over_3;     // 0..1
        return Real{-1} + Real{2} * u;
    }

    // -- State --

    BldcMotorParams params_{};
    std::string     name_{};
    Real            omega_m_           = 0.0;
    Real            theta_m_           = 0.0;
    Real            tau_load_external_ = 0.0;
};

}  // namespace pulsim::v1::motors
