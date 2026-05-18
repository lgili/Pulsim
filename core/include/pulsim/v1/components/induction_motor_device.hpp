#pragma once

// TODO(consolidate-motors): integration into Circuit::DeviceVariant and
// Circuit::add_induction_motor is handled by the parent agent after this
// file lands. The accessors / state-update helpers defined here mirror
// PmsmDevice and are designed to plug into the same stamp_rlc /
// advance_state / restore_state ladder that PmsmDevice already uses in
// runtime_circuit.hpp.

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/motors/frame_transforms.hpp"
#include "pulsim/v1/motors/induction_motor.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1 {

// =============================================================================
// InductionMotorDevice — runtime device (consolidate-motors-and-three-phase, C.2)
// =============================================================================
//
// Three-phase squirrel-cage induction motor modeled as a 4-terminal device
// (A, B, C, N) with three reserved MNA branch rows — one per phase line
// current — and the following internal state:
//
//   - i_sα, i_sβ     : stator currents in the αβ frame (recomputed every
//                      step from the freshly solved abc line currents).
//   - ψ_rα, ψ_rβ     : rotor flux linkage in the αβ frame.
//   - ω_m, θ_m       : mechanical angular velocity / rotor angle.
//
// Stamping model (per-phase trapezoidal companion, mirroring PmsmDevice):
//
//   v_k − v_N = R_s · i_k + L_eff · di_k/dt + e_k(t)
//
// where
//
//   L_eff = σ·L_s,   σ = 1 − L_m²/(L_s·L_r)
//
// is the transient (leakage) inductance seen from the stator after the rotor
// flux dynamics are eliminated, and e_k is the per-phase back-EMF induced by
// the rotating rotor flux. Per phase, the back-EMF is the inverse-Clarke of
// (L_m/L_r)·dψ_rα/dt and (L_m/L_r)·dψ_rβ/dt — see `back_emf_*()` below.
//
// Trapezoidal companion (per phase k ∈ {a, b, c}):
//
//   f_brk = (v_k − v_N) − (R_s + 2·L_eff/dt) · i_k
//         + (2·L_eff/dt) · i_k_prev + V_Lk_prev − e_k(state_prev)
//
// As with PmsmDevice, the back-EMF uses the rotor flux at the *previous*
// accepted step (semi-implicit), so Newton iteration on i_a/i_b/i_c does not
// have to chase the flux state simultaneously. ψ_r, ω, θ are advanced
// forward-Euler in `advance_state()` after the solve.
//
// Electromagnetic torque (Krause / amplitude-invariant form):
//
//   τ_em = (3/2) · p · (L_m / L_r) · (ψ_rα · i_sβ − ψ_rβ · i_sα)
//
// Mechanical step (matches PmsmDevice / DcMotorDevice convention):
//
//   ω_m_new = ω_m + dt · (τ_em − τ_load − b · ω_m − sign(ω_m) · τ_C) / J
//   θ_m_new = θ_m + dt · ω_m
//
// DC operating point: the leakage inductance is shorted, so each phase
// reduces to a voltage source (back-EMF from the initial rotor flux) in
// series with R_s. The OP back-EMF is built from whatever rotor flux state
// the device was initialised with — defaults to zero (motor at rest, no
// magnetisation), which is the realistic squirrel-cage starting condition.

class InductionMotorDevice : public DynamicDeviceBase<InductionMotorDevice> {
public:
    using Base = DynamicDeviceBase<InductionMotorDevice>;
    using Params = motors::InductionMotorParams;

    // 4 pins: A, B, C, N
    static constexpr std::size_t num_pins = 4;
    // Treated as an inductive device for stamping classification.
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    InductionMotorDevice() = default;

    explicit InductionMotorDevice(Params params, std::string name = "")
        : Base(std::move(name))
        , motor_(std::move(params), this->name())
        , i_a_(0.0)
        , i_b_(0.0)
        , i_c_(0.0)
        , i_a_prev_(0.0)
        , i_b_prev_(0.0)
        , i_c_prev_(0.0)
        , v_La_prev_(0.0)
        , v_Lb_prev_(0.0)
        , v_Lc_prev_(0.0)
        , tau_em_(0.0)
        , branch_a_(-1)
        , branch_b_(-1)
        , branch_c_(-1)
        , history_initialized_(false) {}

    // -------------------------------------------------------------------------
    // Reserved MNA branch rows (set by Circuit::add_induction_motor).
    // -------------------------------------------------------------------------
    void set_branch_indices(NodeIndex a, NodeIndex b, NodeIndex c) noexcept {
        branch_a_ = a;
        branch_b_ = b;
        branch_c_ = c;
    }
    [[nodiscard]] NodeIndex branch_a() const noexcept { return branch_a_; }
    [[nodiscard]] NodeIndex branch_b() const noexcept { return branch_b_; }
    [[nodiscard]] NodeIndex branch_c() const noexcept { return branch_c_; }

    // -------------------------------------------------------------------------
    // Mechanical / state I/O.
    // -------------------------------------------------------------------------
    /// External shaft load torque [N·m]. Defaults to 0.
    void set_load_torque(Real tau) noexcept { motor_.set_load_torque(tau); }
    /// Alias used by Circuit-side wrappers; mirrors `PmsmDevice::set_tau_load`.
    void set_tau_load(Real tau) noexcept { motor_.set_load_torque(tau); }
    [[nodiscard]] Real tau_load() const noexcept { return motor_.load_torque(); }

    /// Seed the rotor mechanical speed (e.g. for tests that need to start
    /// at synchronous speed). Should be called before the first stamp.
    void set_initial_omega(Real w) noexcept { motor_.set_omega(w); }
    /// Seed the rotor mechanical angle.
    void set_initial_theta(Real t) noexcept { motor_.set_theta(t); }
    /// Seed the rotor flux state (αβ). Useful for tests that start from a
    /// pre-magnetised motor (skips the initial flux build-up transient).
    void set_initial_rotor_flux(Real psi_a, Real psi_b) noexcept {
        motor_.set_rotor_flux(psi_a, psi_b);
    }

    // -------------------------------------------------------------------------
    // Read-back accessors.
    // -------------------------------------------------------------------------
    [[nodiscard]] Real i_a()       const noexcept { return i_a_; }
    [[nodiscard]] Real i_b()       const noexcept { return i_b_; }
    [[nodiscard]] Real i_c()       const noexcept { return i_c_; }
    [[nodiscard]] Real i_a_prev()  const noexcept { return i_a_prev_; }
    [[nodiscard]] Real i_b_prev()  const noexcept { return i_b_prev_; }
    [[nodiscard]] Real i_c_prev()  const noexcept { return i_c_prev_; }
    [[nodiscard]] Real v_La_prev() const noexcept { return v_La_prev_; }
    [[nodiscard]] Real v_Lb_prev() const noexcept { return v_Lb_prev_; }
    [[nodiscard]] Real v_Lc_prev() const noexcept { return v_Lc_prev_; }

    [[nodiscard]] Real omega_m()  const noexcept { return motor_.omega_m(); }
    [[nodiscard]] Real theta_m()  const noexcept { return motor_.theta_m(); }
    [[nodiscard]] Real i_sa()     const noexcept { return motor_.i_sa(); }
    [[nodiscard]] Real i_sb()     const noexcept { return motor_.i_sb_state(); }
    [[nodiscard]] Real psi_ra()   const noexcept { return motor_.psi_ra(); }
    [[nodiscard]] Real psi_rb()   const noexcept { return motor_.psi_rb(); }
    [[nodiscard]] Real tau_em()   const noexcept { return tau_em_; }

    [[nodiscard]] Real omega_electrical() const noexcept {
        return motor_.omega_electrical();
    }
    [[nodiscard]] Real theta_electrical() const noexcept {
        return motor_.theta_electrical();
    }

    /// Slip s = (ω_sync − ω_e) / ω_sync. Caller supplies the synchronous
    /// electrical angular frequency (rad/s).
    [[nodiscard]] Real slip(Real omega_sync_electrical) const noexcept {
        return motor_.slip(omega_sync_electrical);
    }

    /// Same as above, but the synchronous speed is supplied as a stator
    /// frequency in Hz (typical user convenience: slip(50.0) for 50 Hz grid).
    [[nodiscard]] Real slip_from_hz(Real f_sync_hz) const noexcept {
        constexpr Real two_pi = Real{6.283185307179586};
        return motor_.slip(two_pi * f_sync_hz);
    }

    /// Transient / leakage inductance σ·L_s seen from the stator after the
    /// rotor-flux dynamics are eliminated. Used by the trapezoidal companion
    /// stamping in place of L_s.
    [[nodiscard]] Real l_s_effective() const noexcept {
        const auto& p = motor_.params();
        const Real l_s = std::max<Real>(p.L_s, Real{1e-30});
        const Real l_r = std::max<Real>(p.L_r, Real{1e-30});
        const Real sigma = std::max<Real>(
            Real{1} - (p.L_m * p.L_m) / (l_s * l_r), Real{1e-9});
        return sigma * l_s;
    }

    /// Trapezoidal companion conductance per phase: 1 / (R_s + 2·L_eff/dt).
    /// Useful for tests / diagnostics that need to match the stamping math.
    [[nodiscard]] Real companion_conductance(Real dt) const noexcept {
        return Real{1.0} / (motor_.params().R_s + Real{2.0} * l_s_effective() / dt);
    }

    // -------------------------------------------------------------------------
    // Per-phase back-EMF used by the trapezoidal companion stamping.
    //
    // The induction-motor back-EMF on the stator side comes from the
    // changing rotor flux. After expressing rotor currents in terms of ψ_r,
    // the stator voltage equation factors out as:
    //
    //   v_sα = R_s·i_sα + σ·L_s · di_sα/dt + (L_m/L_r) · dψ_rα/dt
    //
    // so the "back-EMF" seen by the stator α-component is
    //
    //   e_sα = (L_m/L_r) · dψ_rα/dt
    //        = (L_m/L_r) · [−(R_r/L_r)·ψ_rα + (R_r·L_m/L_r)·i_sα
    //                        − ω_e·ψ_rβ]
    //
    // The αβ → abc inverse-Clarke maps it to per-phase back-EMF values.
    //
    // i_sα is the *previous-step* stator current (semi-implicit) so the
    // solver does not have to iterate on flux during Newton.
    // -------------------------------------------------------------------------

    [[nodiscard]] Real back_emf_alpha() const noexcept {
        const auto& p = motor_.params();
        const Real l_r = std::max<Real>(p.L_r, Real{1e-30});
        const Real omega_e = motor_.omega_electrical();
        const Real dpsi_a = -(p.R_r / l_r) * motor_.psi_ra()
                            + (p.R_r * p.L_m / l_r) * motor_.i_sa()
                            - omega_e * motor_.psi_rb();
        return (p.L_m / l_r) * dpsi_a;
    }
    [[nodiscard]] Real back_emf_beta() const noexcept {
        const auto& p = motor_.params();
        const Real l_r = std::max<Real>(p.L_r, Real{1e-30});
        const Real omega_e = motor_.omega_electrical();
        const Real dpsi_b = -(p.R_r / l_r) * motor_.psi_rb()
                            + (p.R_r * p.L_m / l_r) * motor_.i_sb_state()
                            + omega_e * motor_.psi_ra();
        return (p.L_m / l_r) * dpsi_b;
    }
    [[nodiscard]] Real back_emf_a() const noexcept {
        return back_emf_alpha();
    }
    [[nodiscard]] Real back_emf_b() const noexcept {
        constexpr Real sqrt3_over_2 = Real{0.8660254037844386};
        return -Real{0.5} * back_emf_alpha() + sqrt3_over_2 * back_emf_beta();
    }
    [[nodiscard]] Real back_emf_c() const noexcept {
        constexpr Real sqrt3_over_2 = Real{0.8660254037844386};
        return -Real{0.5} * back_emf_alpha() - sqrt3_over_2 * back_emf_beta();
    }

    [[nodiscard]] const Params& params() const noexcept {
        return motor_.params();
    }
    [[nodiscard]] bool history_initialized() const noexcept {
        return history_initialized_;
    }

    // -------------------------------------------------------------------------
    // History / state advance after each accepted MNA step.
    // -------------------------------------------------------------------------
    /// Advance internal state given freshly solved terminal voltages /
    /// branch currents (all relative to neutral). Mirrors the
    /// `PmsmDevice::advance_state` shape — runtime_circuit.hpp will call
    /// this from its `advance_state` ladder.
    void advance_state(Real v_a_term, Real v_b_term, Real v_c_term,
                       Real i_a, Real i_b, Real i_c, Real dt) noexcept {
        // -- 1. Back-EMF at the rotor-flux state used to stamp THIS step. --
        const Real e_a = back_emf_a();
        const Real e_b = back_emf_b();
        const Real e_c = back_emf_c();

        // -- 2. Inductor voltage carried into the next companion step. --
        //    V_Lk = v_k_term − R_s · i_k − e_k (mirrors PMSM convention).
        const auto& p = motor_.params();
        const Real v_La_new = v_a_term - p.R_s * i_a - e_a;
        const Real v_Lb_new = v_b_term - p.R_s * i_b - e_b;
        const Real v_Lc_new = v_c_term - p.R_s * i_c - e_c;

        // -- 3. Project freshly solved abc currents onto αβ. --
        const auto [i_sa_new, i_sb_new] = motors::clarke(i_a, i_b, i_c);
        motor_.set_stator_current(i_sa_new, i_sb_new);

        // -- 4. Compute torque from updated stator currents + rotor flux. --
        tau_em_ = motor_.electromagnetic_torque();

        // -- 5. Advance rotor flux (forward Euler) using NEW i_s. --
        const Real l_r = std::max<Real>(p.L_r, Real{1e-30});
        const Real omega_e = motor_.omega_electrical();
        const Real dpsi_a = -(p.R_r / l_r) * motor_.psi_ra()
                            + (p.R_r * p.L_m / l_r) * i_sa_new
                            - omega_e * motor_.psi_rb();
        const Real dpsi_b = -(p.R_r / l_r) * motor_.psi_rb()
                            + (p.R_r * p.L_m / l_r) * i_sb_new
                            + omega_e * motor_.psi_ra();
        motor_.set_rotor_flux(motor_.psi_ra() + dt * dpsi_a,
                              motor_.psi_rb() + dt * dpsi_b);

        // -- 6. Mechanical forward-Euler step. --
        const Real omega_speed = motor_.omega_m();
        const Real sign_w = (omega_speed > Real{0}) ? Real{1.0}
                          : (omega_speed < Real{0}) ? Real{-1.0}
                          : Real{0.0};
        const Real tau_friction =
            p.b_friction * omega_speed +
            p.friction_coulomb * sign_w;
        const Real tau_load_internal =
            p.tau_load_const +
            p.tau_load_quad_coeff * omega_speed * std::abs(omega_speed);
        const Real j = std::max<Real>(p.J, Real{1e-30});
        const Real omega_new =
            omega_speed + dt *
                (tau_em_ - motor_.load_torque() - tau_friction - tau_load_internal) / j;
        const Real theta_new = motor_.theta_m() + dt * omega_speed;

        motor_.set_omega(omega_new);
        motor_.set_theta(theta_new);

        // -- 7. Shift trapezoidal-companion history forward. --
        i_a_prev_ = i_a;
        i_b_prev_ = i_b;
        i_c_prev_ = i_c;
        i_a_      = i_a;
        i_b_      = i_b;
        i_c_      = i_c;
        v_La_prev_ = v_La_new;
        v_Lb_prev_ = v_Lb_new;
        v_Lc_prev_ = v_Lc_new;
        history_initialized_ = true;
    }

    /// Variant for circuits that solve the DC OP at t=0 and want the
    /// transient history zeroed (no companion carry-over from the OP).
    void reset_history_for_transient_start(Real i_a, Real i_b, Real i_c) noexcept {
        const auto [i_sa_op, i_sb_op] = motors::clarke(i_a, i_b, i_c);
        motor_.set_stator_current(i_sa_op, i_sb_op);

        i_a_prev_ = i_a;
        i_b_prev_ = i_b;
        i_c_prev_ = i_c;
        i_a_      = i_a;
        i_b_      = i_b;
        i_c_      = i_c;
        v_La_prev_ = 0.0;
        v_Lb_prev_ = 0.0;
        v_Lc_prev_ = 0.0;
        tau_em_   = motor_.electromagnetic_torque();
        history_initialized_ = true;
    }

    /// Provided to keep the CRTP DynamicDeviceBase contract happy. The
    /// runtime path uses `advance_state` directly — this is a no-op marker.
    void update_history_impl() noexcept {
        history_initialized_ = true;
    }

private:
    motors::InductionMotor motor_;

    // Phase currents (after each accepted solve) + companion history.
    Real i_a_      = 0.0;
    Real i_b_      = 0.0;
    Real i_c_      = 0.0;
    Real i_a_prev_ = 0.0;
    Real i_b_prev_ = 0.0;
    Real i_c_prev_ = 0.0;
    Real v_La_prev_ = 0.0;
    Real v_Lb_prev_ = 0.0;
    Real v_Lc_prev_ = 0.0;

    Real tau_em_ = 0.0;

    NodeIndex branch_a_ = -1;
    NodeIndex branch_b_ = -1;
    NodeIndex branch_c_ = -1;

    bool history_initialized_ = false;
};

}  // namespace pulsim::v1
