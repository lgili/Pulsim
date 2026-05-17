#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/motors/mechanical.hpp"

#include <array>
#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1 {

// =============================================================================
// Mechanical — runtime device (consolidate-motors-and-three-phase, Phase B.2a)
// =============================================================================
//
// Signal-domain mechanical primitive registered in `RuntimeCircuit::DeviceVariant`
// to enable multi-shaft topologies (gearbox between two motors, belt-coupled
// inertias, dyno load with its own quadratic / constant torque profile). The
// device exposes:
//
//   - state    : `omega_m_` and `theta_m_` evolved per step.
//   - input    : `tau_input_` set externally by the user (typically a motor's
//                electromagnetic torque, summed by a mixed-domain block).
//   - load     : optional `tau_load_const` (constant) + `tau_load_quad_coeff`
//                (k·ω·|ω| — fan / propeller / windage). The load opposes
//                motion, so positive ω yields positive (subtractive) torque.
//
// The device has **no electrical pins**. `stamp_impl` is a no-op (no MNA
// contribution). State advance happens via the variant-dispatch branch the
// `RuntimeCircuit::advance_state` walker invokes on accepted timestep — the
// runtime calls `device.update_history()` after the linear solve.
//
// Coupling to motors: today the user wires `device.set_tau_input(motor.
// electromagnetic_torque())` and `motor.set_tau_load(gear_ratio * device.
// reaction_torque())` in a mixed-domain block (or from Python at each step).
// A more declarative shaft-binding DSL is a follow-up — this device gives the
// foundation for it.

class MechanicalDevice : public DynamicDeviceBase<MechanicalDevice> {
public:
    using Base = DynamicDeviceBase<MechanicalDevice>;
    using ScalarType = Scalar;

    struct Params {
        motors::Shaft shaft;                ///< J, b_friction, friction_coulomb, omega (initial)
        Real theta_init = 0.0;              ///< rad — initial mechanical angle
        Real tau_load_const = 0.0;          ///< N·m — constant load torque (opposes motion sign)
        Real tau_load_quad_coeff = 0.0;     ///< N·m / (rad/s)² — quadratic load (k·ω·|ω|)
        Real gear_ratio = 1.0;              ///< speed ratio if this shaft is downstream of a
                                            ///< gearbox; pure scalar (no torque reflection here)
    };

    // 0 electrical pins — the mechanical device floats outside the MNA graph.
    static constexpr std::size_t num_pins = 0;
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    MechanicalDevice() = default;

    explicit MechanicalDevice(Params params, std::string name = "")
        : Base(std::move(name))
        , params_(std::move(params))
        , omega_m_(params_.shaft.omega)
        , theta_m_(params_.theta_init)
        , tau_input_(0.0)
    {}

    // ---- DynamicDeviceBase interface (CRTP impls) --------------------------

    template<SparseMatrixLike Matrix, VectorLike Vec>
    void stamp_impl(Matrix& /*G*/, Vec& /*b*/,
                    std::span<const NodeIndex> /*nodes*/) {
        // No electrical contribution — mechanical-only device.
    }

    static constexpr auto jacobian_pattern_impl() {
        // Empty pattern: no MNA contribution.
        return std::array<std::array<int, 2>, 0>{};
    }

    void update_history_impl() {
        advance(Base::timestep());
    }

    // ---- Mechanical step ---------------------------------------------------

    /// Forward-Euler advance under net torque `τ_net = τ_input − τ_load`,
    /// with load = const + k·ω·|ω| and friction inside Shaft. Updates the
    /// shaft's stored ω and this device's θ. Callers may also drive this
    /// directly between accepted timesteps if integrating in a custom
    /// scheduler — pass the same dt the simulator used.
    void advance(Real dt) {
        if (dt <= Real{0}) return;
        const Real omega = omega_m_;
        const Real tau_quad = params_.tau_load_quad_coeff * omega * std::abs(omega);
        const Real tau_load_total = params_.tau_load_const + tau_quad;
        const Real tau_net = tau_input_ - tau_load_total;
        // Reuse Shaft::advance for the friction + inertia step. Shaft owns
        // its own ω, so synchronize before and after.
        params_.shaft.omega = omega_m_;
        params_.shaft.advance(tau_net, dt);
        omega_m_ = params_.shaft.omega;
        theta_m_ += dt * omega_m_;
    }

    // ---- External coupling -------------------------------------------------

    void set_tau_input(Real tau) noexcept { tau_input_ = tau; }
    [[nodiscard]] Real tau_input() const noexcept { return tau_input_; }

    void set_initial_omega(Real w) noexcept {
        omega_m_ = w;
        params_.shaft.omega = w;
    }
    void set_initial_theta(Real t) noexcept { theta_m_ = t; }

    [[nodiscard]] Real omega_m() const noexcept { return omega_m_; }
    [[nodiscard]] Real theta_m() const noexcept { return theta_m_; }
    [[nodiscard]] const Params& params() const noexcept { return params_; }

    /// Reaction torque exposed to upstream couplings (e.g., a motor coupled
    /// via gearbox sees `motor.set_tau_load(device.reaction_torque())`).
    /// In this minimal model the reaction equals the load + friction.
    [[nodiscard]] Real reaction_torque() const noexcept {
        const Real tau_quad = params_.tau_load_quad_coeff * omega_m_ * std::abs(omega_m_);
        const Real tau_load_total = params_.tau_load_const + tau_quad;
        const Real tau_friction =
            params_.shaft.b_friction * omega_m_ +
            params_.shaft.friction_coulomb *
                (omega_m_ > Real{0} ? Real{1} :
                 omega_m_ < Real{0} ? Real{-1} : Real{0});
        return tau_load_total + tau_friction;
    }

private:
    Params params_{};
    Real omega_m_ = 0.0;
    Real theta_m_ = 0.0;
    Real tau_input_ = 0.0;
};

}  // namespace pulsim::v1
