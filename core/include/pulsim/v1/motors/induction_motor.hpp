#pragma once

#include "pulsim/v1/motors/frame_transforms.hpp"
#include "pulsim/v1/motors/mechanical.hpp"
#include "pulsim/v1/numeric_types.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1::motors {

// =============================================================================
// consolidate-motors-and-three-phase — Squirrel-cage induction motor (Phase C.2)
// =============================================================================
//
// Three-phase squirrel-cage induction machine modeled in the stationary αβ
// reference frame following the standard Krause / Bose textbook formulation,
// with rotor flux linkage chosen as the state (rotor currents are referred
// quantities that are reconstructed from i_s and ψ_r whenever they are
// needed). The rotor windings are short-circuited (v_r = 0).
//
// Stator electrical equations (αβ frame, written before substitution):
//
//   v_sα = R_s · i_sα + L_s · di_sα/dt + L_m · di_rα/dt
//   v_sβ = R_s · i_sβ + L_s · di_sβ/dt + L_m · di_rβ/dt
//
// Rotor electrical equations (squirrel cage → v_r = 0):
//
//   0 = R_r · i_rα + L_r · di_rα/dt + L_m · di_sα/dt + ω_e · (L_r·i_rβ + L_m·i_sβ)
//   0 = R_r · i_rβ + L_r · di_rβ/dt + L_m · di_sβ/dt - ω_e · (L_r·i_rα + L_m·i_sα)
//
// where ω_e = pole_pairs · ω_m is the electrical rotor speed (the αβ frame
// is stationary, but the squirrel-cage rotor windings move with the rotor
// — hence the speed-voltage cross-coupling on the rotor side).
//
// Substituting i_rα = (ψ_rα − L_m·i_sα) / L_r (and similarly for β), the
// rotor equations reduce to a pair of first-order ODEs in the rotor flux:
//
//   dψ_rα/dt = −(R_r/L_r) · ψ_rα + (R_r · L_m / L_r) · i_sα − ω_e · ψ_rβ
//   dψ_rβ/dt = −(R_r/L_r) · ψ_rβ + (R_r · L_m / L_r) · i_sβ + ω_e · ψ_rα
//
// and the stator equations become (after eliminating di_r/dt and grouping
// — σ = 1 − L_m²/(L_s·L_r) is the total leakage coefficient):
//
//   σ·L_s · di_sα/dt =
//        v_sα − (R_s + R_r · L_m²/L_r²) · i_sα
//             + (L_m · R_r / L_r²) · ψ_rα
//             + (L_m / L_r) · ω_e · ψ_rβ
//   σ·L_s · di_sβ/dt =
//        v_sβ − (R_s + R_r · L_m²/L_r²) · i_sβ
//             + (L_m · R_r / L_r²) · ψ_rβ
//             - (L_m / L_r) · ω_e · ψ_rα
//
// Electromagnetic torque (Krause / amplitude-invariant form):
//
//   τ_em = (3/2) · p · (L_m / L_r) · (ψ_rα · i_sβ − ψ_rβ · i_sα)
//
// Shaft equation (mirrors PMSM / DC-motor convention):
//
//   J · dω_m/dt = τ_em − τ_load_external − b · ω_m − sign(ω_m) · τ_coulomb
//                       − τ_load_const − τ_load_quad_coeff · ω_m · |ω_m|
//   dθ_m/dt = ω_m
//
// State vector exposed to callers / the device wrapper:
//   x = (i_sα, i_sβ, ψ_rα, ψ_rβ, ω_m, θ_m)
//
// The class is *header-only math*. It does not stamp into MNA — that is
// `InductionMotorDevice`'s job. The math here uses an explicit forward-Euler
// step internally because (a) it mirrors the PMSM `step()` shape and
// (b) for typical control-loop dt's (50-100 µs) the time constants of the
// squirrel cage are well above the step. The device wrapper layers a
// trapezoidal companion model on top for MNA stability.

struct InductionMotorParams {
    Real R_s = 1.0;             ///< Ω — stator phase resistance
    Real R_r = 1.5;             ///< Ω — rotor resistance referred to stator
    Real L_s = 0.15;            ///< H — stator self-inductance
    Real L_r = 0.15;            ///< H — rotor self-inductance (≈ L_s typically)
    Real L_m = 0.14;            ///< H — mutual / magnetizing inductance (< L_s, L_r)
    int  pole_pairs = 2;
    Real J = 0.01;              ///< kg·m² — rotor inertia
    Real b_friction = 1e-3;     ///< N·m·s — viscous friction
    Real friction_coulomb = 0.0;       ///< N·m — Coulomb friction
    Real tau_load_const     = 0.0;     ///< N·m — speed-independent load
    Real tau_load_quad_coeff = 0.0;    ///< N·m / (rad/s)² — fan-style load
};

class InductionMotor {
public:
    /// State variables: (i_sα, i_sβ, ψ_rα, ψ_rβ, ω_m, θ_m).
    InductionMotor() = default;

    explicit InductionMotor(InductionMotorParams params, std::string name = "")
        : params_(std::move(params))
        , name_(std::move(name)) {}

    /// Electromagnetic torque computed from current state (Krause form):
    ///   τ_em = (3/2) · p · (L_m / L_r) · (ψ_rα · i_sβ − ψ_rβ · i_sα)
    [[nodiscard]] Real electromagnetic_torque() const noexcept {
        const Real p = static_cast<Real>(params_.pole_pairs);
        const Real l_r = std::max<Real>(params_.L_r, Real{1e-30});
        return Real{1.5} * p * (params_.L_m / l_r)
             * (psi_ra_ * i_sb_ - psi_rb_ * i_sa_);
    }

    /// Single-step semi-implicit advance under stationary-frame stator
    /// voltages (v_sα, v_sβ). Order:
    ///   1. Compute torque from current state.
    ///   2. Advance i_sα, i_sβ via forward Euler on the stator ODE.
    ///   3. Advance ψ_rα, ψ_rβ via forward Euler on the rotor flux ODE.
    ///      (rotor flux uses the *old* i_s so the two updates are
    ///      decoupled — this matches the explicit Euler convention.)
    ///   4. Advance ω_m (forward Euler) and θ_m using the new ω_m.
    void advance(Real v_sa, Real v_sb, Real dt) noexcept {
        // Numerical safety: clamp denominators away from zero.
        const Real l_r = std::max<Real>(params_.L_r, Real{1e-30});
        const Real l_s = std::max<Real>(params_.L_s, Real{1e-30});
        const Real j   = std::max<Real>(params_.J,   Real{1e-30});

        const Real l_m   = params_.L_m;
        const Real r_s   = params_.R_s;
        const Real r_r   = params_.R_r;
        const Real p     = static_cast<Real>(params_.pole_pairs);
        const Real omega_e = p * omega_m_;

        // Total leakage coefficient σ = 1 − L_m²/(L_s·L_r).
        const Real sigma = std::max<Real>(
            Real{1} - (l_m * l_m) / (l_s * l_r), Real{1e-9});
        const Real sigma_ls = sigma * l_s;

        // Useful subexpressions.
        const Real r_eq      = r_s + r_r * (l_m * l_m) / (l_r * l_r);
        const Real lm_over_lr = l_m / l_r;
        const Real lm_rr_over_lr2 = lm_over_lr * r_r / l_r;  // = L_m·R_r / L_r²

        // -- Step 1: torque from current state (used downstream for shaft). --
        const Real tau_em = electromagnetic_torque();

        // -- Step 2: stator current derivatives (σ·L_s · di/dt = …). --
        const Real didt_a = (v_sa
                             - r_eq * i_sa_
                             + lm_rr_over_lr2 * psi_ra_
                             + lm_over_lr * omega_e * psi_rb_) / sigma_ls;
        const Real didt_b = (v_sb
                             - r_eq * i_sb_
                             + lm_rr_over_lr2 * psi_rb_
                             - lm_over_lr * omega_e * psi_ra_) / sigma_ls;

        // -- Step 3: rotor flux derivatives at the OLD state. --
        const Real k_r = r_r / l_r;
        const Real g_r = r_r * l_m / l_r;
        const Real dpsi_a = -k_r * psi_ra_
                            + g_r * i_sa_
                            - omega_e * psi_rb_;
        const Real dpsi_b = -k_r * psi_rb_
                            + g_r * i_sb_
                            + omega_e * psi_ra_;

        // -- Step 4: advance stator currents (forward Euler) and rotor
        //            flux (one-iteration trapezoidal, Heun's predictor-
        //            corrector — harden-component-models-vs-psim-plecs
        //            Phase A5). The rotor-flux ODE is linear with the αβ
        //            cross-coupling block `[-k_r, -ω_e; +ω_e, -k_r]`;
        //            at 50 Hz DOL-start (ω_e ≈ 314 rad/s, dt ≈ 100 µs)
        //            forward Euler accumulates a perpendicular-axis
        //            phase-lag error of ω_e²·dt/2 ≈ 5 rad/s per step,
        //            visible as a slow |ψ_r| drift that destabilises
        //            high-slip starts. Trapezoidal averages dψ at the
        //            old state and at the forward-Euler predictor —
        //            4 extra mul + 2 adds per step, no implicit solve.
        i_sa_ += dt * didt_a;
        i_sb_ += dt * didt_b;

        const Real psi_ra_pred = psi_ra_ + dt * dpsi_a;
        const Real psi_rb_pred = psi_rb_ + dt * dpsi_b;
        // Use the freshly-advanced stator currents in the corrector
        // (same convention as the device wrapper's advance_state).
        const Real dpsi_a_pred = -k_r * psi_ra_pred
                                 + g_r * i_sa_
                                 - omega_e * psi_rb_pred;
        const Real dpsi_b_pred = -k_r * psi_rb_pred
                                 + g_r * i_sb_
                                 + omega_e * psi_ra_pred;
        psi_ra_ += Real{0.5} * dt * (dpsi_a + dpsi_a_pred);
        psi_rb_ += Real{0.5} * dt * (dpsi_b + dpsi_b_pred);

        // -- Step 5: shaft acceleration. --
        const Real sign_w = (omega_m_ > Real{0}) ? Real{1.0}
                          : (omega_m_ < Real{0}) ? Real{-1.0}
                          : Real{0.0};
        const Real tau_friction =
            params_.b_friction * omega_m_ +
            params_.friction_coulomb * sign_w;
        const Real tau_load_internal =
            params_.tau_load_const +
            params_.tau_load_quad_coeff * omega_m_ * std::abs(omega_m_);
        omega_m_ += dt *
            (tau_em - tau_load_external_ - tau_friction - tau_load_internal) / j;
        theta_m_ += dt * omega_m_;
    }

    // -- Inverse Clarke: αβ → abc (amplitude-invariant convention). --
    [[nodiscard]] Real i_sa() const noexcept { return i_sa_; }
    [[nodiscard]] Real i_sb_state() const noexcept { return i_sb_; }

    /// Phase A line current = i_sα.
    [[nodiscard]] Real i_phase_a() const noexcept {
        return i_sa_;
    }
    /// Phase B line current = −1/2·i_sα + √3/2·i_sβ.
    [[nodiscard]] Real i_phase_b() const noexcept {
        constexpr Real sqrt3_over_2 = Real{0.8660254037844386};
        return -Real{0.5} * i_sa_ + sqrt3_over_2 * i_sb_;
    }
    /// Phase C line current = −1/2·i_sα − √3/2·i_sβ.
    [[nodiscard]] Real i_phase_c() const noexcept {
        constexpr Real sqrt3_over_2 = Real{0.8660254037844386};
        return -Real{0.5} * i_sa_ - sqrt3_over_2 * i_sb_;
    }

    /// Rotor flux αβ components (state).
    [[nodiscard]] Real psi_ra() const noexcept { return psi_ra_; }
    [[nodiscard]] Real psi_rb() const noexcept { return psi_rb_; }

    /// Reconstructed (referred) rotor currents:
    ///   i_rα = (ψ_rα − L_m · i_sα) / L_r,  i_rβ analogously.
    [[nodiscard]] Real i_ra() const noexcept {
        const Real l_r = std::max<Real>(params_.L_r, Real{1e-30});
        return (psi_ra_ - params_.L_m * i_sa_) / l_r;
    }
    [[nodiscard]] Real i_rb() const noexcept {
        const Real l_r = std::max<Real>(params_.L_r, Real{1e-30});
        return (psi_rb_ - params_.L_m * i_sb_) / l_r;
    }

    [[nodiscard]] Real omega_m() const noexcept { return omega_m_; }
    [[nodiscard]] Real theta_m() const noexcept { return theta_m_; }

    /// Electrical rotor angle / speed (θ_e = p·θ_m, ω_e = p·ω_m).
    [[nodiscard]] Real omega_electrical() const noexcept {
        return static_cast<Real>(params_.pole_pairs) * omega_m_;
    }
    [[nodiscard]] Real theta_electrical() const noexcept {
        return static_cast<Real>(params_.pole_pairs) * theta_m_;
    }

    /// Slip s = (ω_sync − ω_e) / ω_sync = 1 − ω_e/ω_sync. Caller supplies
    /// the synchronous *electrical* angular frequency ω_sync (rad/s) of the
    /// stator excitation — it cannot be inferred from the state alone.
    /// Returns 1 (locked rotor, full slip) when ω_sync ≈ 0.
    [[nodiscard]] Real slip(Real omega_sync_electrical) const noexcept {
        if (std::abs(omega_sync_electrical) < Real{1e-12}) {
            return Real{1};
        }
        const Real omega_e = omega_electrical();
        return (omega_sync_electrical - omega_e) / omega_sync_electrical;
    }

    [[nodiscard]] const InductionMotorParams& params() const noexcept {
        return params_;
    }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    void set_omega(Real w) noexcept { omega_m_ = w; }
    void set_theta(Real t) noexcept { theta_m_ = t; }
    void set_stator_current(Real i_sa, Real i_sb) noexcept {
        i_sa_ = i_sa;
        i_sb_ = i_sb;
    }
    void set_rotor_flux(Real psi_a, Real psi_b) noexcept {
        psi_ra_ = psi_a;
        psi_rb_ = psi_b;
    }
    void set_load_torque(Real tau) noexcept { tau_load_external_ = tau; }
    [[nodiscard]] Real load_torque() const noexcept { return tau_load_external_; }

private:
    InductionMotorParams params_{};
    std::string name_;

    // State: stator currents (αβ) + rotor flux (αβ) + mechanical pair.
    Real i_sa_   = 0.0;
    Real i_sb_   = 0.0;
    Real psi_ra_ = 0.0;
    Real psi_rb_ = 0.0;
    Real omega_m_ = 0.0;
    Real theta_m_ = 0.0;

    // External shaft load torque (set via `set_load_torque`).
    Real tau_load_external_ = 0.0;
};

}  // namespace pulsim::v1::motors
