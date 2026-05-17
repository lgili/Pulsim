#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/motors/frame_transforms.hpp"
#include "pulsim/v1/motors/pmsm.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1 {

// =============================================================================
// PMSM — runtime device (Phase: integrate-3φ-motors-magnetics, Track 2.2)
// =============================================================================
//
// Permanent-magnet synchronous machine modeled as a 4-terminal device (A, B,
// C, N) with 3 reserved MNA branch rows — one per phase line current — and 4
// internal state variables:
//
//   - omega_m_  : rotor mechanical angular velocity (rad/s)
//   - theta_m_  : rotor mechanical angle (rad)
//   - i_d_      : cached d-axis stator current (computed after each step
//                  via Park transform — exposed for telemetry / closed-loop
//                  control feedback, not stamped directly)
//   - i_q_      : cached q-axis stator current (same)
//
// Per-phase electrical model (each phase k = a, b, c):
//
//   v_k − v_N = R_s · i_k + L_s · di_k/dt + e_k(θ_e)
//
// where the back-EMF e_k arises from the rotating permanent-magnet flux on
// the d-axis. With θ_e = p · θ_m and ω_e = p · ω_m, projecting the q-axis
// flux velocity λ̇_q = ω_e · ψ_PM through the inverse Park/Clarke pair gives
// the standard balanced 3-phase back-EMF set:
//
//   e_a = -ω_e · ψ_PM · sin(θ_e)
//   e_b = -ω_e · ψ_PM · sin(θ_e - 2π/3)
//   e_c = -ω_e · ψ_PM · sin(θ_e + 2π/3)
//
// L_s used in the abc-frame stamping is the non-salient mean (L_d + L_q)/2.
// Saliency (L_d ≠ L_q) is captured only in the torque expression below — a
// fully salient abc-frame model would need angle-dependent per-phase
// inductance, which is heavier and only meaningful for IPM machines. For
// SPM (L_d = L_q) the average matches both axes exactly.
//
// Trapezoidal companion (mirrors Inductor / DcMotorDevice):
//
//   f_brk = (v_k − v_N) − (R_s + 2L_s/dt) · i_k
//         + (2L_s/dt) · i_k_prev + V_Lk_prev − e_k(θ_e_prev)
//
// Back-EMF uses θ_e at the *previous* accepted step (semi-implicit), so
// solver iterations on i_a/i_b/i_c do not have to chase ω at the same
// time. ω and θ are advanced forward-Euler in `advance_state()`.
//
// Electromagnetic torque (the standard amplitude-invariant Park form):
//
//   τ_em = (3/2) · p · (ψ_PM · i_q + (L_d − L_q) · i_d · i_q)
//
// Mechanical step:
//
//   ω_m_new = ω_m + dt · (τ_em − τ_load − b · ω_m − sign(ω_m) · τ_C) / J
//   θ_m_new = θ_m + dt · ω_m
//
// DC OP: inductors are shorts, so each phase reduces to
//
//   v_k − v_N = R_s · i_k + e_k(θ_e_init)
//
// which is stamped as a voltage source in series with R_s.
//
// External torque can be set via `set_tau_load(τ)`; defaults to 0 N·m.

class PmsmDevice : public DynamicDeviceBase<PmsmDevice> {
public:
    using Base = DynamicDeviceBase<PmsmDevice>;
    using Params = motors::PmsmParams;

    // 4 pins: A, B, C, N
    static constexpr std::size_t num_pins = 4;
    // Treated as an inductive device for stamping classification.
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    PmsmDevice() = default;

    explicit PmsmDevice(Params params, std::string name = "")
        : Base(std::move(name))
        , params_(std::move(params))
        , i_a_(0.0)
        , i_b_(0.0)
        , i_c_(0.0)
        , i_a_prev_(0.0)
        , i_b_prev_(0.0)
        , i_c_prev_(0.0)
        , v_La_prev_(0.0)
        , v_Lb_prev_(0.0)
        , v_Lc_prev_(0.0)
        , omega_m_(params_.omega_init)
        , theta_m_(params_.theta_init)
        , i_d_(params_.i_d_init)
        , i_q_(params_.i_q_init)
        , tau_em_(0.0)
        , tau_load_(0.0)
        , branch_a_(-1)
        , branch_b_(-1)
        , branch_c_(-1)
        , history_initialized_(false) {}

    /// Reserved MNA branch row indices for the 3 line currents. Set by
    /// `Circuit::add_pmsm` at construction time.
    void set_branch_indices(NodeIndex a, NodeIndex b, NodeIndex c) noexcept {
        branch_a_ = a;
        branch_b_ = b;
        branch_c_ = c;
    }
    [[nodiscard]] NodeIndex branch_a() const noexcept { return branch_a_; }
    [[nodiscard]] NodeIndex branch_b() const noexcept { return branch_b_; }
    [[nodiscard]] NodeIndex branch_c() const noexcept { return branch_c_; }

    /// External shaft load torque [N·m]. Defaults to 0 — set to model loading.
    void set_tau_load(Real tau) noexcept { tau_load_ = tau; }
    [[nodiscard]] Real tau_load() const noexcept { return tau_load_; }

    [[nodiscard]] Real i_a()       const noexcept { return i_a_; }
    [[nodiscard]] Real i_b()       const noexcept { return i_b_; }
    [[nodiscard]] Real i_c()       const noexcept { return i_c_; }
    [[nodiscard]] Real i_a_prev()  const noexcept { return i_a_prev_; }
    [[nodiscard]] Real i_b_prev()  const noexcept { return i_b_prev_; }
    [[nodiscard]] Real i_c_prev()  const noexcept { return i_c_prev_; }
    [[nodiscard]] Real v_La_prev() const noexcept { return v_La_prev_; }
    [[nodiscard]] Real v_Lb_prev() const noexcept { return v_Lb_prev_; }
    [[nodiscard]] Real v_Lc_prev() const noexcept { return v_Lc_prev_; }

    [[nodiscard]] Real omega_m()  const noexcept { return omega_m_; }
    [[nodiscard]] Real theta_m()  const noexcept { return theta_m_; }
    [[nodiscard]] Real i_d()      const noexcept { return i_d_; }
    [[nodiscard]] Real i_q()      const noexcept { return i_q_; }
    [[nodiscard]] Real tau_em()   const noexcept { return tau_em_; }

    [[nodiscard]] Real omega_electrical() const noexcept {
        return static_cast<Real>(params_.pole_pairs) * omega_m_;
    }
    [[nodiscard]] Real theta_electrical() const noexcept {
        return static_cast<Real>(params_.pole_pairs) * theta_m_;
    }

    /// Non-salient inductance used by the abc-frame stamping (mean of L_d/L_q).
    [[nodiscard]] Real l_s_effective() const noexcept {
        return Real{0.5} * (params_.Ld + params_.Lq);
    }

    /// Trapezoidal companion conductance for each phase R+L branch:
    ///   g_eq = 1 / (R_s + 2·L_s/dt). Useful for diagnostics and matching
    /// the internal stamping math in tests.
    [[nodiscard]] Real companion_conductance(Real dt) const noexcept {
        return 1.0 / (params_.Rs + 2.0 * l_s_effective() / dt);
    }

    /// Per-phase back-EMF at the *current* electrical angle stored in
    /// theta_m_ (i.e. the angle that was used to stamp THIS step). The
    /// Jacobian ladder reads this value when computing the residual.
    [[nodiscard]] Real back_emf_a() const noexcept {
        const Real theta_e = theta_electrical();
        const Real omega_e = omega_electrical();
        return -omega_e * params_.psi_pm * std::sin(theta_e);
    }
    [[nodiscard]] Real back_emf_b() const noexcept {
        constexpr Real two_pi_over_3 = Real{2.0943951023931953};
        const Real theta_e = theta_electrical();
        const Real omega_e = omega_electrical();
        return -omega_e * params_.psi_pm * std::sin(theta_e - two_pi_over_3);
    }
    [[nodiscard]] Real back_emf_c() const noexcept {
        constexpr Real two_pi_over_3 = Real{2.0943951023931953};
        const Real theta_e = theta_electrical();
        const Real omega_e = omega_electrical();
        return -omega_e * params_.psi_pm * std::sin(theta_e + two_pi_over_3);
    }

    [[nodiscard]] const Params& params() const noexcept { return params_; }
    [[nodiscard]] bool history_initialized() const noexcept {
        return history_initialized_;
    }

    /// Advance the mechanical + electrical history after each accepted
    /// timestep. The MNA layer has already produced the new (i_a, i_b, i_c)
    /// branch currents and (v_a, v_b, v_c) terminal voltages — all measured
    /// relative to the neutral node. Computes the new torque from the dq
    /// projection, advances ω and θ by forward-Euler, and shifts the
    /// trapezoidal-companion history forward.
    ///
    /// IMPORTANT: this is called with the back-EMF still based on
    /// theta_m_ as it was at the START of this step (semi-implicit). The
    /// new theta_m_ is computed at the END of this call so the *next* step
    /// uses the freshly-advanced angle.
    void advance_state(Real v_a_term, Real v_b_term, Real v_c_term,
                       Real i_a, Real i_b, Real i_c, Real dt) noexcept {
        // Back-EMF at the angle that produced this solve (semi-implicit).
        const Real e_a = back_emf_a();
        const Real e_b = back_emf_b();
        const Real e_c = back_emf_c();

        // Inductor voltage carried into the next step's companion model:
        // V_Lk = v_k_term − R_s · i_k − e_k.
        const Real v_La_new = v_a_term - params_.Rs * i_a - e_a;
        const Real v_Lb_new = v_b_term - params_.Rs * i_b - e_b;
        const Real v_Lc_new = v_c_term - params_.Rs * i_c - e_c;

        // Project abc currents onto the dq frame (same θ_e used in stamping).
        const Real theta_e = theta_electrical();
        const auto [i_d_new, i_q_new] =
            motors::abc_to_dq(i_a, i_b, i_c, theta_e);

        // Electromagnetic torque (amplitude-invariant Park).
        const Real p = static_cast<Real>(params_.pole_pairs);
        const Real tau_em_new =
            Real{1.5} * p *
            (params_.psi_pm * i_q_new +
             (params_.Ld - params_.Lq) * i_d_new * i_q_new);

        // Mechanical forward-Euler step. Coulomb friction adds a sign(ω)·τ_C
        // term that opposes motion (zero at exact standstill).
        const Real omega_speed = omega_m_;
        const Real sign_omega = (omega_speed > Real{0}) ? Real{1.0}
                              : (omega_speed < Real{0}) ? Real{-1.0}
                              : Real{0.0};
        const Real tau_friction =
            params_.b_friction * omega_speed +
            params_.friction_coulomb * sign_omega;
        const Real omega_new =
            omega_m_ + dt * (tau_em_new - tau_load_ - tau_friction) /
                std::max<Real>(params_.J, Real{1e-30});
        const Real theta_new = theta_m_ + dt * omega_m_;

        i_a_prev_ = i_a;
        i_b_prev_ = i_b;
        i_c_prev_ = i_c;
        i_a_ = i_a;
        i_b_ = i_b;
        i_c_ = i_c;
        v_La_prev_ = v_La_new;
        v_Lb_prev_ = v_Lb_new;
        v_Lc_prev_ = v_Lc_new;
        i_d_ = i_d_new;
        i_q_ = i_q_new;
        tau_em_ = tau_em_new;
        omega_m_ = omega_new;
        theta_m_ = theta_new;
        history_initialized_ = true;
    }

    /// Variant for circuits that solve the DC OP at t=0 and want the
    /// transient history zeroed (no companion-model carry-over from the OP).
    void reset_history_for_transient_start(Real i_a, Real i_b, Real i_c) noexcept {
        i_a_prev_ = i_a;
        i_b_prev_ = i_b;
        i_c_prev_ = i_c;
        i_a_ = i_a;
        i_b_ = i_b;
        i_c_ = i_c;
        // No "previous" L_s · di/dt at t=0 — start with v_L_prev = 0.
        v_La_prev_ = 0.0;
        v_Lb_prev_ = 0.0;
        v_Lc_prev_ = 0.0;
        // Also seed dq currents from the OP solution for completeness.
        const Real theta_e = theta_electrical();
        const auto [i_d_op, i_q_op] =
            motors::abc_to_dq(i_a, i_b, i_c, theta_e);
        i_d_ = i_d_op;
        i_q_ = i_q_op;
        history_initialized_ = true;
    }

    /// Provided to keep the CRTP DynamicDeviceBase contract happy. The
    /// runtime path doesn't call this — it calls `advance_state` directly.
    void update_history_impl() noexcept {
        history_initialized_ = true;
    }

private:
    Params params_{};
    Real i_a_         = 0.0;
    Real i_b_         = 0.0;
    Real i_c_         = 0.0;
    Real i_a_prev_    = 0.0;
    Real i_b_prev_    = 0.0;
    Real i_c_prev_    = 0.0;
    Real v_La_prev_   = 0.0;
    Real v_Lb_prev_   = 0.0;
    Real v_Lc_prev_   = 0.0;
    Real omega_m_     = 0.0;
    Real theta_m_     = 0.0;
    Real i_d_         = 0.0;
    Real i_q_         = 0.0;
    Real tau_em_      = 0.0;
    Real tau_load_    = 0.0;
    NodeIndex branch_a_ = -1;
    NodeIndex branch_b_ = -1;
    NodeIndex branch_c_ = -1;
    bool history_initialized_ = false;
};

}  // namespace pulsim::v1
