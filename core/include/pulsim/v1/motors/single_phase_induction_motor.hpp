#pragma once

#include "pulsim/v1/numeric_types.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::motors {

// =============================================================================
// Single-phase induction motor — Permanent Split Capacitor (PSC) topology
// =============================================================================
//
// This is the "compressor convencional" motor used in pre-inverter
// (fixed-frequency on-off) domestic refrigerators / freezers — also
// commonly called PSC, RSIR, or CSIR depending on the start-aid
// configuration. PSC is the dominant configuration for hermetic
// refrigeration compressors: main winding + auxiliary winding shifted
// 90° spatially, with a run capacitor permanently in series with the
// auxiliary winding to keep it active throughout normal operation.
//
// Modelled as a two-winding stator equivalent (αβ frame, main on α,
// auxiliary on β) with a squirrel-cage rotor whose state is the
// rotor flux linkage ψ_r in the same αβ frame. The capacitor is
// internal to the device (not a separate Circuit primitive) so the
// user just sees a 2-pin black-box motor — line and neutral.
//
// State vector: [i_main, i_aux, ψ_r_alpha, ψ_r_beta, V_cap, ω_m, θ_m]
//
// Stator equations (stationary frame):
//   v_main = R_s_main · i_main + L_s_main · di_main/dt
//                              + (L_m / L_r) · dψ_r_alpha/dt
//   v_aux − V_cap = R_s_aux · i_aux + L_s_aux · di_aux/dt
//                                   + (L_m / L_r) · dψ_r_beta/dt
//
// Capacitor:
//   dV_cap/dt = i_aux / C_run
//
// Rotor flux dynamics (squirrel cage, short-circuited windings):
//   dψ_rα/dt = −R_r/L_r · ψ_rα + R_r·L_m/L_r · i_main − ω_e · ψ_rβ
//   dψ_rβ/dt = −R_r/L_r · ψ_rβ + R_r·L_m/L_r · i_aux  + ω_e · ψ_rα
//
// Electromagnetic torque (Krause αβ form for 2-phase stator):
//   T_em = p · (L_m / L_r) · (ψ_rα · i_aux − ψ_rβ · i_main)
//
// Mechanical:
//   J · dω_m/dt = T_em − τ_load − b · ω_m − τ_C · sign(ω_m)
//
// Notes:
//   - Main and auxiliary windings see the SAME line voltage v_main =
//     v_aux = v_line; the difference is that the aux branch has the
//     capacitor in series, so its effective driving voltage is
//     v_aux_effective = v_line − V_cap.
//   - This is a fixed-frequency model — at 60 Hz (or 50 Hz) line
//     supply the motor runs at near-synchronous speed (slip ~ 3-5%
//     at rated load) regardless of internal load.
//   - For modern inverter-driven (BLDC/PMSM) compressors, use
//     `BldcMotorDevice` or `PmsmDevice` instead.

struct SinglePhaseInductionMotorParams {
    // ---- Main winding (α-axis) ----
    Real R_s_main = 10.0;        ///< Ω — main stator resistance
    Real L_s_main = 50e-3;       ///< H — main stator self-inductance

    // ---- Auxiliary winding (β-axis) + run capacitor (PSC) ----
    Real R_s_aux = 20.0;         ///< Ω — auxiliary stator resistance
    Real L_s_aux = 80e-3;        ///< H — auxiliary stator self-inductance
    Real C_run = 4e-6;           ///< F — permanent run capacitor

    // ---- Rotor (squirrel cage, referred to stator) ----
    Real R_r = 8.0;
    Real L_r = 55e-3;            ///< self-inductance
    Real L_m = 50e-3;            ///< mutual inductance

    // ---- Mechanical ----
    int  pole_pairs = 2;          ///< 4-pole = 2 pole pairs, typical fridge
    Real J = 1e-4;                ///< kg·m² — rotor + shaft inertia
    Real b_friction = 1e-4;       ///< N·m·s — viscous friction
    Real friction_coulomb = 0.05; ///< N·m — Coulomb friction
};

class SinglePhaseInductionMotor {
public:
    SinglePhaseInductionMotor() = default;

    explicit SinglePhaseInductionMotor(SinglePhaseInductionMotorParams params,
                                       std::string name = "")
        : params_(std::move(params)), name_(std::move(name)) {}

    /// Advance state under the imposed instantaneous line voltage.
    /// `v_line` is the voltage applied between the device's line and
    /// neutral terminals (i.e., v_main = v_aux = v_line). The model
    /// integrates one step of dt seconds using a semi-implicit Euler
    /// scheme on the αβ state-space form.
    void advance(Real v_line, Real dt) {
        if (!(dt > Real{0})) return;
        const Real omega_e =
            static_cast<Real>(params_.pole_pairs) * omega_m_;

        // Auxiliary branch effective driving voltage (line minus the
        // run capacitor's terminal voltage).
        const Real v_aux_eff = v_line - V_cap_;

        // Build the explicit state-space update. Stator current
        // dynamics from the trapezoidal-companion-style integration:
        //   L_s · di/dt = v − R_s · i − (L_m/L_r) · dψ_r/dt
        // We compute dψ_r/dt below first, then substitute.
        const Real Lr = params_.L_r;
        const Real Lm = params_.L_m;
        const Real Rr = params_.R_r;
        const Real Lm_over_Lr = Lm / Lr;

        const Real dpsi_ra_dt = -Rr / Lr * psi_ra_ +
                                Rr * Lm / Lr * i_main_ -
                                omega_e * psi_rb_;
        const Real dpsi_rb_dt = -Rr / Lr * psi_rb_ +
                                Rr * Lm / Lr * i_aux_ +
                                omega_e * psi_ra_;

        // Stator current derivatives.
        const Real di_main_dt = (v_line - params_.R_s_main * i_main_ -
                                  Lm_over_Lr * dpsi_ra_dt) /
                                 params_.L_s_main;
        const Real di_aux_dt = (v_aux_eff - params_.R_s_aux * i_aux_ -
                                Lm_over_Lr * dpsi_rb_dt) /
                                params_.L_s_aux;

        // Forward-Euler advance of stator currents and rotor flux. The
        // capacitor voltage gets trapezoidal treatment below
        // (harden-component-models-vs-psim-plecs Phase A5) — see the
        // device-wrapper analogue in
        // `components/single_phase_induction_motor_device.hpp`.
        const Real i_aux_prev = i_aux_;
        psi_ra_ += dt * dpsi_ra_dt;
        psi_rb_ += dt * dpsi_rb_dt;
        i_main_ += dt * di_main_dt;
        i_aux_  += dt * di_aux_dt;

        // Trapezoidal cap update: V_cap_new = V_cap + 0.5·dt·(i_aux_old +
        // i_aux_new)/C_run. Removes the half-step lag that the previous
        // forward-Euler form (`V_cap += dt·i_aux/C_run`) introduced in
        // the run-cap starting transient at dt = 100 µs.
        V_cap_ += Real{0.5} * dt * (i_aux_prev + i_aux_) / params_.C_run;

        // Torque + mechanical advance.
        const Real p = static_cast<Real>(params_.pole_pairs);
        const Real T_em = p * Lm_over_Lr *
                          (psi_ra_ * i_aux_ - psi_rb_ * i_main_);
        const Real tau_friction =
            params_.b_friction * omega_m_ +
            params_.friction_coulomb *
                (omega_m_ > Real{0} ? Real{1} :
                 omega_m_ < Real{0} ? Real{-1} : Real{0});
        const Real domega_dt = (T_em - tau_load_ - tau_friction) /
                                params_.J;
        omega_m_ += dt * domega_dt;
        theta_m_ += dt * omega_m_;
    }

    // ---- Inverse Clarke (αβ → ab) of the stator currents.
    // In a two-phase representation, the "phase A" current at the
    // motor's line terminal is i_main + i_aux (both go through the
    // single line input). We expose the breakdown for telemetry.
    [[nodiscard]] Real i_main() const noexcept { return i_main_; }
    [[nodiscard]] Real i_aux() const noexcept { return i_aux_; }
    [[nodiscard]] Real i_line() const noexcept { return i_main_ + i_aux_; }

    [[nodiscard]] Real V_cap() const noexcept { return V_cap_; }
    [[nodiscard]] Real psi_r_alpha() const noexcept { return psi_ra_; }
    [[nodiscard]] Real psi_r_beta() const noexcept { return psi_rb_; }
    [[nodiscard]] Real omega_m() const noexcept { return omega_m_; }
    [[nodiscard]] Real theta_m() const noexcept { return theta_m_; }

    /// Electromagnetic torque at the current state (N·m).
    [[nodiscard]] Real electromagnetic_torque() const noexcept {
        const Real Lm_over_Lr = params_.L_m / params_.L_r;
        const Real p = static_cast<Real>(params_.pole_pairs);
        return p * Lm_over_Lr *
               (psi_ra_ * i_aux_ - psi_rb_ * i_main_);
    }

    /// Slip referenced to synchronous speed (s = 1 − ω_m / ω_sync).
    /// For 60 Hz / 4-pole: ω_sync_mech = 2π·60 / 2 = 188.5 rad/s.
    [[nodiscard]] Real slip(Real omega_sync_mechanical) const noexcept {
        if (omega_sync_mechanical == Real{0}) return Real{1};
        return Real{1} - omega_m_ / omega_sync_mechanical;
    }

    void set_omega(Real w) noexcept { omega_m_ = w; }
    void set_theta(Real t) noexcept { theta_m_ = t; }
    void set_load_torque(Real tau) noexcept { tau_load_ = tau; }
    [[nodiscard]] Real load_torque() const noexcept { return tau_load_; }

    /// External-state advance used by `SinglePhaseInductionMotorDevice`.
    /// The Circuit's MNA already solved for the new stator currents;
    /// we receive them along with the rotor-flux derivatives that the
    /// device pre-computed, and we just integrate the rotor flux,
    /// mechanical state, and the capacitor voltage forward by `dt`.
    /// This bypasses the internal `advance(v_line, dt)` path, which
    /// derives currents from a single line voltage and would conflict
    /// with the device's stamped MNA solution.
    void advance_state_external_(Real i_main_new, Real i_aux_new,
                                 Real dpsi_ra_dt, Real dpsi_rb_dt,
                                 Real dt) {
        if (!(dt > Real{0})) return;
        i_main_ = i_main_new;
        i_aux_  = i_aux_new;
        psi_ra_ += dt * dpsi_ra_dt;
        psi_rb_ += dt * dpsi_rb_dt;
        const Real Lm_over_Lr = params_.L_m / params_.L_r;
        const Real p = static_cast<Real>(params_.pole_pairs);
        const Real T_em = p * Lm_over_Lr *
                          (psi_ra_ * i_aux_ - psi_rb_ * i_main_);
        const Real tau_friction =
            params_.b_friction * omega_m_ +
            params_.friction_coulomb *
                (omega_m_ > Real{0} ? Real{1} :
                 omega_m_ < Real{0} ? Real{-1} : Real{0});
        const Real domega_dt = (T_em - tau_load_ - tau_friction) /
                                params_.J;
        omega_m_ += dt * domega_dt;
        theta_m_ += dt * omega_m_;
        // Run-capacitor voltage is owned by the device, not the math
        // object — the device handles V_cap update separately.
    }

    [[nodiscard]] const SinglePhaseInductionMotorParams& params() const noexcept {
        return params_;
    }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    SinglePhaseInductionMotorParams params_{};
    std::string name_;
    // Stator currents
    Real i_main_ = 0.0;
    Real i_aux_  = 0.0;
    // Rotor flux linkage (αβ)
    Real psi_ra_ = 0.0;
    Real psi_rb_ = 0.0;
    // Run-capacitor voltage
    Real V_cap_  = 0.0;
    // Mechanical
    Real omega_m_ = 0.0;
    Real theta_m_ = 0.0;
    // External load torque (compressor's mechanical demand)
    Real tau_load_ = 0.0;
};

}  // namespace pulsim::v1::motors
