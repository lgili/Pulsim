#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/motors/single_phase_induction_motor.hpp"

#include <array>
#include <cmath>
#include <string>
#include <utility>

namespace pulsim::v1 {

// =============================================================================
// Single-phase induction motor — runtime device (CC compressor model)
// =============================================================================
//
// 2-pin device wrapping the PSC (Permanent Split Capacitor) single-phase
// induction motor math from `motors::SinglePhaseInductionMotor`. Used as
// the electrical model for the "compressor convencional" found in
// pre-inverter domestic refrigerators / freezers.
//
// Pin layout: { line, neutral }. The run capacitor + auxiliary winding
// are internal to the device (no extra Circuit pins).
//
// Reserved MNA branch rows (2): i_main (α-axis) and i_aux (β-axis).
// Per-phase trapezoidal companion stamping mirrors the 3φ induction
// motor — `R_eff = R_s + 2 · L_s / dt` plus a Norton back-EMF source
// derived from the rotor flux. The auxiliary branch eq also subtracts
// the internal capacitor voltage `V_cap` so the run capacitor's
// dynamics influence the auxiliary current correctly.
//
// State advance (post-step):
//   1. Read i_main, i_aux from the MNA solution.
//   2. Advance rotor flux ψ_rα/β via forward Euler under the new
//      stator currents.
//   3. Compute electromagnetic torque T_em.
//   4. Advance ω_m, θ_m via the mechanical equation.
//   5. Advance V_cap: V_cap_new = V_cap + dt · i_aux / C_run.
//   6. Save current i_main / i_aux as `i_prev` for the next step's
//      trapezoidal companion.

class SinglePhaseInductionMotorDevice
    : public DynamicDeviceBase<SinglePhaseInductionMotorDevice> {
public:
    using Base = DynamicDeviceBase<SinglePhaseInductionMotorDevice>;
    using ScalarType = Scalar;
    using Params = motors::SinglePhaseInductionMotorParams;

    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    SinglePhaseInductionMotorDevice() = default;

    explicit SinglePhaseInductionMotorDevice(Params params, std::string name = "")
        : Base(std::move(name))
        , params_(std::move(params))
        , math_(params_, std::string{Base::name()})
    {}

    // ---- Branch reservation -----------------------------------------------

    /// Reserve the (i_main, i_aux) branch rows. Convention: branch_index
    /// points to i_main; branch_index + 1 → i_aux.
    void set_branch_indices(Index br_main, Index br_aux) noexcept {
        branch_index_main_ = br_main;
        branch_index_aux_  = br_aux;
    }
    [[nodiscard]] Index branch_index_main() const noexcept {
        return branch_index_main_;
    }
    [[nodiscard]] Index branch_index_aux() const noexcept {
        return branch_index_aux_;
    }

    // ---- DynamicDeviceBase interface (CRTP impls) -------------------------

    template<SparseMatrixLike Matrix, VectorLike Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;
        if (branch_index_main_ < 0 || branch_index_aux_ < 0) return;
        const Real dt = Base::timestep();
        if (!(dt > Real{0})) return;
        const NodeIndex n_line    = nodes[0];
        const NodeIndex n_neutral = nodes[1];

        // Per-branch trapezoidal companion: R_eq = R_s + 2·L_s/dt,
        // v_hist = 2·L_s/dt · i_prev + V_L_prev.
        const auto stamp_branch = [&](NodeIndex br,
                                       Real R_s, Real L_s,
                                       Real i_prev, Real v_L_prev,
                                       Real back_emf, Real v_cap_subtract) {
            const Real two_L_over_dt = Real{2} * L_s / dt;
            const Real R_eq = R_s + two_L_over_dt;
            const Real v_hist = two_L_over_dt * i_prev + v_L_prev;

            // KCL contributions: current flows from line → neutral.
            if (n_line    >= 0) G.coeffRef(n_line,    br) += Real{1};
            if (n_neutral >= 0) G.coeffRef(n_neutral, br) -= Real{1};

            // Branch eq: v_line - v_neutral - V_cap - R_eq · i + v_hist
            //                              - back_emf = 0
            if (n_line    >= 0) G.coeffRef(br, n_line)    += Real{1};
            if (n_neutral >= 0) G.coeffRef(br, n_neutral) -= Real{1};
            G.coeffRef(br, br) -= R_eq;
            if (br < static_cast<Index>(b.size())) {
                b[br] -= (v_hist - back_emf - v_cap_subtract);
            }
        };

        // Main winding (α-axis, no capacitor in series).
        stamp_branch(branch_index_main_,
                     params_.R_s_main, params_.L_s_main,
                     i_main_prev_, v_L_main_prev_,
                     back_emf_alpha_(), /*v_cap_subtract=*/Real{0});

        // Auxiliary winding (β-axis, V_cap in series so subtracts from
        // the driving voltage).
        stamp_branch(branch_index_aux_,
                     params_.R_s_aux, params_.L_s_aux,
                     i_aux_prev_, v_L_aux_prev_,
                     back_emf_beta_(), /*v_cap_subtract=*/V_cap_);
    }

    static constexpr auto jacobian_pattern_impl() {
        // 2 windings × (2 KCL + 2 branch-eq + 1 diagonal) = 10 entries.
        // Local index convention: 0=line, 1=neutral, 2=br_main, 3=br_aux.
        return std::array<std::array<int, 2>, 10>{{
            {0, 2}, {1, 2}, {2, 0}, {2, 1}, {2, 2},
            {0, 3}, {1, 3}, {3, 0}, {3, 1}, {3, 3}
        }};
    }

    /// Post-step advance called by the variant walker. Reads the MNA
    /// solution to update i_main, i_aux, then evolves rotor flux,
    /// mechanical state, and the run-capacitor voltage.
    void advance_state(Real i_main, Real i_aux, Real dt) {
        if (!(dt > Real{0})) return;
        const Real omega_e =
            static_cast<Real>(params_.pole_pairs) * math_.omega_m();
        const Real Lr = params_.L_r;
        const Real Lm = params_.L_m;
        const Real Rr = params_.R_r;
        const Real psi_ra_now = math_.psi_r_alpha();
        const Real psi_rb_now = math_.psi_r_beta();

        // Rotor flux derivatives (forward Euler).
        const Real dpsi_ra_dt = -Rr / Lr * psi_ra_now +
                                Rr * Lm / Lr * i_main -
                                omega_e * psi_rb_now;
        const Real dpsi_rb_dt = -Rr / Lr * psi_rb_now +
                                Rr * Lm / Lr * i_aux +
                                omega_e * psi_ra_now;

        // Push new state to the math object.
        // (math_ owns its state; we mutate via its public setters.)
        math_.set_omega(math_.omega_m());  // no-op, just keeps API symmetry
        // Direct state mutation: we cannot go through math_.advance(...)
        // because that re-derives currents from voltages. Instead, hand
        // the already-solved currents to the math object via internal
        // setters. To minimize new public API, expose a single
        // `advance_state_external` on the math object that takes the
        // already-known stator currents.
        math_.advance_state_external_(i_main, i_aux,
                                       dpsi_ra_dt, dpsi_rb_dt, dt);

        // Update inductor companion history per branch.
        const Real two_L_main_over_dt =
            Real{2} * params_.L_s_main / dt;
        const Real two_L_aux_over_dt =
            Real{2} * params_.L_s_aux / dt;
        const Real v_L_main_new =
            two_L_main_over_dt * (i_main - i_main_prev_) - v_L_main_prev_;
        const Real v_L_aux_new =
            two_L_aux_over_dt * (i_aux - i_aux_prev_) - v_L_aux_prev_;
        i_main_prev_   = i_main;
        i_aux_prev_    = i_aux;
        v_L_main_prev_ = v_L_main_new;
        v_L_aux_prev_  = v_L_aux_new;

        // Advance the run-capacitor voltage (forward Euler).
        V_cap_ += dt * i_aux / params_.C_run;
    }

    void reset_history_for_transient_start(Real i_main, Real i_aux) noexcept {
        i_main_prev_ = i_main;
        i_aux_prev_  = i_aux;
        v_L_main_prev_ = Real{0};
        v_L_aux_prev_  = Real{0};
    }

    void update_history_impl() {
        // History maintained inside advance_state(); nothing else here.
    }

    // ---- Accessors --------------------------------------------------------

    [[nodiscard]] const Params& params() const noexcept { return params_; }
    [[nodiscard]] const motors::SinglePhaseInductionMotor& math() const noexcept {
        return math_;
    }
    [[nodiscard]] Real omega_m() const noexcept { return math_.omega_m(); }
    [[nodiscard]] Real theta_m() const noexcept { return math_.theta_m(); }
    [[nodiscard]] Real i_main()   const noexcept { return i_main_prev_; }
    [[nodiscard]] Real i_aux()    const noexcept { return i_aux_prev_; }
    [[nodiscard]] Real i_line()   const noexcept { return i_main_prev_ + i_aux_prev_; }
    [[nodiscard]] Real V_cap()    const noexcept { return V_cap_; }
    [[nodiscard]] Real v_L_main_prev() const noexcept { return v_L_main_prev_; }
    [[nodiscard]] Real v_L_aux_prev()  const noexcept { return v_L_aux_prev_; }
    [[nodiscard]] Real back_emf_alpha() const noexcept { return back_emf_alpha_(); }
    [[nodiscard]] Real back_emf_beta()  const noexcept { return back_emf_beta_(); }
    [[nodiscard]] Real electromagnetic_torque() const noexcept {
        return math_.electromagnetic_torque();
    }
    [[nodiscard]] Real slip(Real omega_sync_mech) const noexcept {
        return math_.slip(omega_sync_mech);
    }

    void set_initial_omega(Real w) noexcept { math_.set_omega(w); }
    void set_load_torque(Real tau) noexcept { math_.set_load_torque(tau); }
    [[nodiscard]] Real load_torque() const noexcept {
        return math_.load_torque();
    }

private:
    Params params_{};
    motors::SinglePhaseInductionMotor math_{};
    Index branch_index_main_ = -1;
    Index branch_index_aux_  = -1;

    // Trapezoidal-companion history per branch.
    Real i_main_prev_   = 0.0;
    Real i_aux_prev_    = 0.0;
    Real v_L_main_prev_ = 0.0;
    Real v_L_aux_prev_  = 0.0;

    // Run capacitor voltage (V), evolved forward-Euler.
    Real V_cap_ = 0.0;

    // Back-EMF projections from rotor flux into the stator frame.
    // For each axis: e = (L_m / L_r) · dψ_r/dt, but we approximate
    // semi-implicitly using the most recent psi_r and omega_e — the
    // value used by the trapezoidal stamp at the upcoming step is the
    // value just before advance_state.
    [[nodiscard]] Real back_emf_alpha_() const noexcept {
        const Real omega_e =
            static_cast<Real>(params_.pole_pairs) * math_.omega_m();
        return params_.L_m / params_.L_r *
               (-params_.R_r / params_.L_r * math_.psi_r_alpha() +
                params_.R_r * params_.L_m / params_.L_r * i_main_prev_ -
                omega_e * math_.psi_r_beta());
    }
    [[nodiscard]] Real back_emf_beta_() const noexcept {
        const Real omega_e =
            static_cast<Real>(params_.pole_pairs) * math_.omega_m();
        return params_.L_m / params_.L_r *
               (-params_.R_r / params_.L_r * math_.psi_r_beta() +
                params_.R_r * params_.L_m / params_.L_r * i_aux_prev_ +
                omega_e * math_.psi_r_alpha());
    }
};

}  // namespace pulsim::v1
