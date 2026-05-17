#pragma once

// TODO(consolidate-motors): integration into Circuit::DeviceVariant and
// Circuit::add_bldc_motor is handled by the parent agent after this file
// lands. This header only ships the device class itself; runtime wiring
// (variant entry, stamp_rlc / advance_state / restore_state branches,
// pybind11 binding, Python re-export) is added separately so this file
// can be reviewed in isolation.

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/motors/bldc_motor.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1 {

// =============================================================================
// BLDC Motor — runtime device (consolidate-motors-and-three-phase, Phase C.1)
// =============================================================================
//
// Brushless-DC machine modeled as a 4-terminal device (A, B, C, N) with 3
// reserved MNA branch rows — one per phase line current — and 2 internal
// mechanical states (ω_m, θ_m). Mirrors PmsmDevice in shape; the only thing
// that changes is the back-EMF waveform (trapezoidal flat-top instead of
// sinusoidal) and the torque expression (energy-balance form valid for any
// back-EMF shape, see `motors::BldcMotor`).
//
// Per-phase electrical equation (each phase k = a, b, c):
//
//   v_k − v_N = R_s · i_k + L_s · di_k/dt + e_k(θ_e, ω_m)
//
// with
//
//   e_k(θ_e, ω_m) = K_e_peak · f_k(θ_e) · ω_e
//   f_a(θ_e) = trapezoidal profile (+1 / ramp / -1 / ramp) on θ_e ∈ [0, 2π)
//   f_b(θ_e) = f_a(θ_e - 2π/3),   f_c(θ_e) = f_a(θ_e - 4π/3)
//   θ_e = p · θ_m,                 ω_e = p · ω_m
//
// Trapezoidal companion (mirrors Inductor / DcMotorDevice / PmsmDevice):
//
//   f_brk = (v_k − v_N) − (R_s + 2L_s/dt) · i_k
//         + (2L_s/dt) · i_k_prev + V_Lk_prev − e_k(θ_e_prev)
//
// The back-EMF uses θ_e at the *previous* accepted step (semi-implicit), so
// Newton iterations on i_a / i_b / i_c do not have to chase ω_m at the same
// time. ω_m and θ_m advance forward-Euler in `advance_state()`.
//
// Electromagnetic torque (energy-balance form, finite at ω_m = 0):
//
//   τ_em = K_e_peak · p · (f_a(θ_e) · i_a + f_b(θ_e) · i_b + f_c(θ_e) · i_c)
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
// which is stamped as a voltage source (V = e_k) in series with R_s.
//
// External load torque can be set via `set_load_torque(τ)`; defaults to 0.

class BldcMotorDevice : public DynamicDeviceBase<BldcMotorDevice> {
public:
    using Base   = DynamicDeviceBase<BldcMotorDevice>;
    using Params = motors::BldcMotorParams;

    // 4 pins: A, B, C, N
    static constexpr std::size_t num_pins   = 4;
    // Inductive-class device for stamping classification (matches PmsmDevice).
    static constexpr int         device_type = static_cast<int>(DeviceType::Inductor);

    BldcMotorDevice() = default;

    explicit BldcMotorDevice(Params params, std::string name = "")
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

    // ---------------------------------------------------------------------
    // MNA wiring — branch indices reserved by Circuit::add_bldc_motor
    // ---------------------------------------------------------------------

    void set_branch_indices(NodeIndex a, NodeIndex b, NodeIndex c) noexcept {
        branch_a_ = a;
        branch_b_ = b;
        branch_c_ = c;
    }
    [[nodiscard]] NodeIndex branch_a() const noexcept { return branch_a_; }
    [[nodiscard]] NodeIndex branch_b() const noexcept { return branch_b_; }
    [[nodiscard]] NodeIndex branch_c() const noexcept { return branch_c_; }

    /// Convenience alias matching the brief's single-index API (Circuit
    /// uses `branch_a_` as the "primary" row; b and c follow contiguously).
    void set_branch_index(NodeIndex br_a) noexcept {
        branch_a_ = br_a;
        branch_b_ = (br_a >= 0) ? br_a + 1 : -1;
        branch_c_ = (br_a >= 0) ? br_a + 2 : -1;
    }
    [[nodiscard]] NodeIndex branch_index() const noexcept { return branch_a_; }

    // ---------------------------------------------------------------------
    // Per-phase observables (used by the stamping ladders and tests)
    // ---------------------------------------------------------------------

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
    [[nodiscard]] Real tau_em()   const noexcept { return tau_em_; }

    [[nodiscard]] Real omega_electrical() const noexcept {
        return motor_.omega_electrical();
    }
    [[nodiscard]] Real theta_electrical() const noexcept {
        return motor_.theta_electrical(motor_.theta_m());
    }

    /// Trapezoidal companion conductance for each phase R+L branch:
    ///   g_eq = 1 / (R_s + 2·L_s/dt). Matches the internal stamping math —
    /// useful for diagnostics and test fixtures.
    [[nodiscard]] Real companion_conductance(Real dt) const noexcept {
        return Real{1} / (motor_.params().R_s +
                          Real{2} * motor_.params().L_s / dt);
    }

    /// Per-phase back-EMF evaluated at the CURRENT θ_m / ω_m. The Jacobian
    /// ladder reads this when computing the residual.
    [[nodiscard]] Real back_emf_a() const noexcept {
        return motor_.back_emf_phase(0, motor_.theta_m());
    }
    [[nodiscard]] Real back_emf_b() const noexcept {
        return motor_.back_emf_phase(1, motor_.theta_m());
    }
    [[nodiscard]] Real back_emf_c() const noexcept {
        return motor_.back_emf_phase(2, motor_.theta_m());
    }

    /// Per-phase profile in per-unit [-1, +1] at the device's current angle.
    /// Used by `electromagnetic_torque_at_current_angle` and by tests that
    /// want the dimensionless shape without re-deriving K_e · ω_e.
    [[nodiscard]] Real profile_a() const noexcept {
        return motor_.back_emf_profile(0, motor_.theta_m());
    }
    [[nodiscard]] Real profile_b() const noexcept {
        return motor_.back_emf_profile(1, motor_.theta_m());
    }
    [[nodiscard]] Real profile_c() const noexcept {
        return motor_.back_emf_profile(2, motor_.theta_m());
    }

    /// Per-phase profile at an arbitrary θ_m (used by tests sweeping the
    /// open-circuit back-EMF waveform).
    [[nodiscard]] Real back_emf_profile_at(int phase_index, Real theta_m) const {
        return motor_.back_emf_profile(phase_index, theta_m);
    }

    /// Per-phase back-EMF [V] at an arbitrary θ_m (using the motor's
    /// current ω_m). Used by tests that sweep θ_m at fixed ω_m.
    [[nodiscard]] Real back_emf_at(int phase_index, Real theta_m) const {
        return motor_.back_emf_phase(phase_index, theta_m);
    }

    /// Electromagnetic torque at the device's current θ_m and the given
    /// phase currents. The math object handles the angle / pole-pair work.
    [[nodiscard]] Real electromagnetic_torque(Real i_a, Real i_b,
                                              Real i_c) const {
        return motor_.electromagnetic_torque(i_a, i_b, i_c, motor_.theta_m());
    }

    /// Same as above but at an arbitrary θ_m. Convenience for the locked-
    /// rotor / open-circuit unit tests.
    [[nodiscard]] Real electromagnetic_torque_at(Real i_a, Real i_b,
                                                 Real i_c,
                                                 Real theta_m) const {
        return motor_.electromagnetic_torque(i_a, i_b, i_c, theta_m);
    }

    // ---------------------------------------------------------------------
    // Setters
    // ---------------------------------------------------------------------

    void set_initial_omega(Real w) noexcept { motor_.set_omega(w); }
    void set_initial_theta(Real t) noexcept { motor_.set_theta(t); }
    void set_load_torque(Real tau) noexcept { motor_.set_load_torque(tau); }
    [[nodiscard]] Real load_torque() const noexcept { return motor_.load_torque(); }

    [[nodiscard]] const Params& params() const noexcept { return motor_.params(); }
    [[nodiscard]] const motors::BldcMotor& motor() const noexcept { return motor_; }
    [[nodiscard]] bool history_initialized() const noexcept {
        return history_initialized_;
    }

    // ---------------------------------------------------------------------
    // History advance (post-solve hook called by the Circuit ladder)
    // ---------------------------------------------------------------------

    /// Advance the mechanical + electrical history after each accepted
    /// timestep. The MNA layer has already produced (i_a, i_b, i_c) line
    /// currents and (v_a, v_b, v_c) terminal voltages measured relative to
    /// the neutral node. Computes the new torque from the trapezoidal
    /// back-EMF projection, advances ω and θ by forward-Euler, and shifts
    /// the companion-model history forward.
    ///
    /// Order of operations matches PmsmDevice::advance_state — the
    /// back-EMF used here is the one at θ_m as it stood at the START of
    /// this step (semi-implicit). θ_m is then advanced at the end so the
    /// NEXT step's stamp picks up the updated angle.
    void advance_state(Real v_a_term, Real v_b_term, Real v_c_term,
                       Real i_a, Real i_b, Real i_c, Real dt) noexcept {
        const auto& p = motor_.params();

        // Back-EMF at the angle that produced this solve (semi-implicit).
        const Real e_a = back_emf_a();
        const Real e_b = back_emf_b();
        const Real e_c = back_emf_c();

        // Inductor voltage carried into the next step's companion model:
        // V_Lk = v_k_term − R_s · i_k − e_k.
        const Real v_La_new = v_a_term - p.R_s * i_a - e_a;
        const Real v_Lb_new = v_b_term - p.R_s * i_b - e_b;
        const Real v_Lc_new = v_c_term - p.R_s * i_c - e_c;

        // Torque from energy balance (finite at ω = 0).
        tau_em_ = motor_.electromagnetic_torque(i_a, i_b, i_c,
                                                motor_.theta_m());

        // Mechanical forward-Euler step — delegate to the math object so
        // both paths (header-only and device-wrapper) stay in sync.
        motor_.advance(i_a, i_b, i_c, dt);

        i_a_prev_  = i_a;
        i_b_prev_  = i_b;
        i_c_prev_  = i_c;
        i_a_       = i_a;
        i_b_       = i_b;
        i_c_       = i_c;
        v_La_prev_ = v_La_new;
        v_Lb_prev_ = v_Lb_new;
        v_Lc_prev_ = v_Lc_new;
        history_initialized_ = true;
    }

    /// Variant for circuits that solve the DC OP at t=0 and want the
    /// transient history zeroed (no companion-model carry-over from the OP).
    void reset_history_for_transient_start(Real i_a, Real i_b, Real i_c) noexcept {
        i_a_prev_  = i_a;
        i_b_prev_  = i_b;
        i_c_prev_  = i_c;
        i_a_       = i_a;
        i_b_       = i_b;
        i_c_       = i_c;
        // No "previous" L_s · di/dt at t=0 — start with v_L_prev = 0.
        v_La_prev_ = 0.0;
        v_Lb_prev_ = 0.0;
        v_Lc_prev_ = 0.0;
        tau_em_    = 0.0;
        history_initialized_ = true;
    }

    /// CRTP DynamicDeviceBase contract hook. The runtime path doesn't call
    /// this — it calls `advance_state` directly.
    void update_history_impl() noexcept {
        history_initialized_ = true;
    }

private:
    motors::BldcMotor motor_{};

    // Cached observables from the last accepted step (also used by the
    // trapezoidal companion model on the next step).
    Real i_a_       = 0.0;
    Real i_b_       = 0.0;
    Real i_c_       = 0.0;
    Real i_a_prev_  = 0.0;
    Real i_b_prev_  = 0.0;
    Real i_c_prev_  = 0.0;
    Real v_La_prev_ = 0.0;
    Real v_Lb_prev_ = 0.0;
    Real v_Lc_prev_ = 0.0;
    Real tau_em_    = 0.0;

    // MNA branch row indices (set by Circuit::add_bldc_motor).
    NodeIndex branch_a_ = -1;
    NodeIndex branch_b_ = -1;
    NodeIndex branch_c_ = -1;

    bool history_initialized_ = false;
};

}  // namespace pulsim::v1
