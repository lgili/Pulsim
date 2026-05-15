#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/motors/dc_motor.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1 {

// =============================================================================
// DC Motor — runtime device (Phase: integrate-3φ-motors-magnetics, Track 2)
// =============================================================================
//
// Wraps the analytical `motors::DcMotor` model so the Circuit can stamp the
// armature into MNA. Models the motor as a single 2-terminal device with one
// reserved branch row for the armature current and two internal state vars
// (mechanical angular velocity ω and rotor angle θ).
//
// Stamping (transient, trapezoidal companion model — mirrors the existing
// Inductor approach so the solver stays stable):
//
//   v_a+ − v_a− = R_a · i_a + L_a · di_a/dt + K_e · ω
//
// Trapezoidal: L_a · di_a/dt ≈ (2L_a/dt) · (i_a − i_a_prev) − V_L_prev
// with V_L_prev = the inductor voltage at the previous timestep.
//
// Substituting, the branch residual is:
//
//   f_br = (v_a+ − v_a−) − (R_a + 2L_a/dt) · i_a
//        + (2L_a/dt) · i_a_prev + V_L_prev − K_e · ω_prev
//
// The mechanical state is advanced by forward-Euler after each accepted
// timestep (the math is gentle enough — typical mechanical time constants are
// orders of magnitude slower than the electrical loop, so trap on i_a is the
// dominant accuracy gain):
//
//   τ_em = K_t · i_a_new
//   ω_new = ω + dt · (τ_em − τ_load − b · ω) / J
//   θ_new = θ + dt · ω
//
// For DC OP the inductor is a short (di/dt = 0), so the armature reduces to
// the resistive equation
//
//   v_a+ − v_a− = R_a · i_a + K_e · ω_init
//
// — same Jacobian structure but without the (2L_a/dt) term.
//
// External load torque can be set via `set_tau_load(τ)`; defaults to 0 N·m so
// no-load step response works out of the box.

class DcMotorDevice : public DynamicDeviceBase<DcMotorDevice> {
public:
    using Base = DynamicDeviceBase<DcMotorDevice>;
    using Params = motors::DcMotorParams;

    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    DcMotorDevice() = default;

    explicit DcMotorDevice(Params params, std::string name = "")
        : Base(std::move(name))
        , params_(std::move(params))
        , i_a_(params_.i_a_init)
        , i_a_prev_(params_.i_a_init)
        , v_a_prev_(0.0)
        , v_L_prev_(0.0)
        , omega_(params_.omega_init)
        , theta_(params_.theta_init)
        , tau_load_(0.0)
        , branch_index_(-1)
        , history_initialized_(false) {}

    /// Reserved MNA branch row index for the armature current. Set by
    /// `Circuit::add_dc_motor` at construction time.
    void set_branch_index(NodeIndex idx) noexcept { branch_index_ = idx; }
    [[nodiscard]] NodeIndex branch_index() const noexcept { return branch_index_; }

    /// External shaft load torque [N·m]. Defaults to 0 — set to model loading.
    void set_tau_load(Real tau) noexcept { tau_load_ = tau; }
    [[nodiscard]] Real tau_load() const noexcept { return tau_load_; }

    [[nodiscard]] Real i_a()        const noexcept { return i_a_; }
    [[nodiscard]] Real i_a_prev()   const noexcept { return i_a_prev_; }
    [[nodiscard]] Real v_a_prev()   const noexcept { return v_a_prev_; }
    [[nodiscard]] Real v_L_prev()   const noexcept { return v_L_prev_; }
    [[nodiscard]] Real omega()      const noexcept { return omega_; }
    [[nodiscard]] Real theta()      const noexcept { return theta_; }
    [[nodiscard]] const Params& params() const noexcept { return params_; }
    [[nodiscard]] bool history_initialized() const noexcept {
        return history_initialized_;
    }

    /// Trapezoidal companion conductance for the armature R+L branch:
    /// g_eq = 1 / (R_a + 2·L_a/dt). Useful for diagnostics & matching the
    /// internal stamping math.
    [[nodiscard]] Real companion_conductance(Real dt) const noexcept {
        return 1.0 / (params_.R_a + 2.0 * params_.L_a / dt);
    }

    /// Mechanical time constant τ_m ≈ J · R_a / (K_t · K_e). Identical to the
    /// math object's accessor — handy for tests.
    [[nodiscard]] Real mechanical_time_constant() const noexcept {
        const Real denom = params_.K_t * params_.K_e;
        if (std::abs(denom) < 1e-30) return 0.0;
        return params_.J * params_.R_a / denom;
    }

    /// Called by the `Circuit::update_history` ladder after each accepted
    /// timestep with the new MNA solution. Advances mechanical state and
    /// shifts the armature companion-model history forward.
    ///
    /// `v_terminal` is the new (v_a+ − v_a−) voltage across the motor
    /// terminals; `i_branch` is the newly-solved armature current.
    void advance_state(Real v_terminal, Real i_branch, Real dt) noexcept {
        // Back-EMF using the *previous* speed (semi-implicit). Avoids the
        // circular dependency between i_a (this step) and ω (next step).
        const Real v_back_emf = params_.K_e * omega_;

        // V_L = v_terminal − R_a·i − K_e·ω
        const Real v_L_new = v_terminal - params_.R_a * i_branch - v_back_emf;

        // Forward-Euler mechanical step using the *new* armature current.
        const Real tau_em = params_.K_t * i_branch;
        const Real omega_new =
            omega_ + dt * (tau_em - tau_load_ - params_.b * omega_) / params_.J;
        const Real theta_new = theta_ + dt * omega_;

        i_a_prev_ = i_branch;
        i_a_ = i_branch;
        v_a_prev_ = v_terminal;
        v_L_prev_ = v_L_new;
        omega_ = omega_new;
        theta_ = theta_new;
        history_initialized_ = true;
    }

    /// Variant for circuits that solve the DC OP at t=0 and want the
    /// transient history zeroed (no companion-model carry-over from the OP).
    void reset_history_for_transient_start(Real v_terminal,
                                           Real i_branch) noexcept {
        i_a_prev_ = i_branch;
        i_a_ = i_branch;
        v_a_prev_ = v_terminal;
        // No "previous" L_a · di/dt at t=0 — start with v_L_prev = 0.
        v_L_prev_ = 0.0;
        history_initialized_ = true;
    }

    /// Provided to keep the CRTP DynamicDeviceBase contract happy. The
    /// runtime path doesn't call this — it calls `advance_state` directly.
    void update_history_impl() noexcept {
        history_initialized_ = true;
    }

private:
    Params params_{};
    Real i_a_         = 0.0;   ///< Current armature current (read by tests).
    Real i_a_prev_    = 0.0;   ///< Trapezoidal companion: i at previous step.
    Real v_a_prev_    = 0.0;   ///< Terminal voltage at previous step.
    Real v_L_prev_    = 0.0;   ///< Inductor voltage at previous step.
    Real omega_       = 0.0;   ///< Mechanical angular velocity (rad/s).
    Real theta_       = 0.0;   ///< Rotor angle (rad).
    Real tau_load_    = 0.0;   ///< External load torque (N·m).
    NodeIndex branch_index_ = -1;
    bool history_initialized_ = false;
};

}  // namespace pulsim::v1
