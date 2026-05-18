#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/magnetic/hysteresis_inductor.hpp"
#include "pulsim/v1/magnetic/bh_curve.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <string>
#include <utility>

namespace pulsim::v1 {

// =============================================================================
// HysteresisInductor — runtime device (deferred-items follow-up, Item 1)
// =============================================================================
//
// 2-pin nonlinear inductor that tracks Jiles-Atherton hysteresis state
// for telemetry. First-pass implementation:
//
//   - MNA stamp uses a CONSTANT effective inductance `L_eff` derived
//     from the geometry at construction (or user-supplied). This keeps
//     the stamping linear; the saturation curve does NOT feed back
//     into the MNA Jacobian.
//
//   - After each accepted timestep, `update_history_impl` reads the
//     branch current, derives the flux `λ ≈ L_eff · i`, and advances
//     the J-A state via `apply_flux_step(λ)`. The resulting
//     magnetization M, internal field H, and flux λ are exposed as
//     accessors for telemetry.
//
// Full nonlinear coupling (Newton-iterated effective inductance from
// the i(λ) saturation curve) is a follow-up; this wrapper covers the
// "device exists in DeviceVariant + hysteresis loop is integrated for
// telemetry" milestone called out in `consolidate-motors-and-three-phase`
// Phase B.3.

class HysteresisInductorDevice : public DynamicDeviceBase<HysteresisInductorDevice> {
public:
    using Base = DynamicDeviceBase<HysteresisInductorDevice>;
    using ScalarType = Scalar;

    struct Params {
        magnetic::HysteresisInductor::Geometry geom{};
        magnetic::JilesAthertonParams ja{};
        /// Effective inductance used for the MNA stamp (H). When 0,
        /// the device falls back to a permeability-based default:
        ///   L_eff = N² · μ_0 · μ_r · A_e / l_e
        /// with `mu_r_init = 1000` (typical soft-iron initial
        /// permeability). Users with a measurement set the field
        /// directly to override.
        Real L_eff = 0.0;
        /// Initial inductor current (A) — used to seed the trapezoidal
        /// companion history at simulation start.
        Real i_init = 0.0;
    };

    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    HysteresisInductorDevice() = default;

    explicit HysteresisInductorDevice(Params params, std::string name = "")
        : Base(std::move(name))
        , params_(std::move(params))
        , math_(params_.geom, params_.ja, std::string{Base::name()})
        , i_prev_(params_.i_init)
        , v_L_prev_(0.0)
    {
        // Derive a default L_eff from geometry when unset.
        if (!(params_.L_eff > Real{0})) {
            constexpr Real mu_0 = Real{4e-7} * std::numbers::pi_v<Real>;
            constexpr Real mu_r_init = 1000.0;   // typical soft-iron initial μ_r
            const Real N = params_.geom.turns;
            const Real A = params_.geom.area;
            const Real l = params_.geom.path_length;
            if (l > Real{0}) {
                params_.L_eff = N * N * mu_0 * mu_r_init * A / l;
            } else {
                params_.L_eff = Real{1e-3};   // safety floor
            }
        }
    }

    // ---- Branch reservation -------------------------------------------------

    void set_branch_index(Index br) noexcept { branch_index_ = br; }
    [[nodiscard]] Index branch_index() const noexcept { return branch_index_; }

    // ---- DynamicDeviceBase interface (CRTP impls) --------------------------

    template<SparseMatrixLike Matrix, VectorLike Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        // Trapezoidal companion stamp for a linear inductor:
        //   v_pos − v_neg = R_eq · i + V_hist
        // with R_eq = 2·L_eff/dt and V_hist = −(2·L_eff/dt·i_prev + V_L_prev).
        if (nodes.size() < 2 || branch_index_ < 0) return;
        const Index n_pos = nodes[0];
        const Index n_neg = nodes[1];
        const Index br    = branch_index_;
        const Real dt = Base::timestep();
        if (!(dt > Real{0})) return;
        const Real two_L_over_dt = Real{2} * params_.L_eff / dt;
        const Real v_hist = two_L_over_dt * i_prev_ + v_L_prev_;

        // KCL contributions at the two terminal nodes.
        if (n_pos >= 0) G.coeffRef(n_pos, br) += Real{1};
        if (n_neg >= 0) G.coeffRef(n_neg, br) -= Real{1};

        // Branch equation: v_pos − v_neg − R_eq · i + V_hist = 0
        if (n_pos >= 0) G.coeffRef(br, n_pos) += Real{1};
        if (n_neg >= 0) G.coeffRef(br, n_neg) -= Real{1};
        G.coeffRef(br, br) -= two_L_over_dt;
        if (br < static_cast<Index>(b.size())) b[br] -= v_hist;
    }

    static constexpr auto jacobian_pattern_impl() {
        // 2 KCL entries + 2 branch-eq entries + 1 diagonal = 5 sparsity
        // entries. We expose them as (row, col) tuples relative to the
        // local pin/branch indexing (terminal 0/1, branch 2).
        return std::array<std::array<int, 2>, 5>{{
            {0, 2}, {1, 2}, {2, 0}, {2, 1}, {2, 2}
        }};
    }

    /// Read the most-recent branch current from the MNA solution and
    /// advance the J-A state. Called by the variant walker once per
    /// accepted timestep.
    void advance_state(Real i_branch) {
        const Real lambda_new = params_.L_eff * i_branch;
        const Real v_L_new = (Real{2} * params_.L_eff / Base::timestep())
                                 * (i_branch - i_prev_) - v_L_prev_;
        i_prev_   = i_branch;
        v_L_prev_ = v_L_new;
        math_.apply_flux_step(lambda_new);
    }

    void update_history_impl() {
        // Called by DynamicDeviceBase::update_history(). The variant
        // walker pushes the latest i_branch via advance_state(i) just
        // before this entry — nothing further to do here.
    }

    /// Reset history (used at transient start to seed from the DC OP).
    void reset_history_for_transient_start(Real i_init) noexcept {
        i_prev_ = i_init;
        v_L_prev_ = Real{0};
    }

    // ---- Accessors ---------------------------------------------------------

    [[nodiscard]] const Params& params() const noexcept { return params_; }
    [[nodiscard]] const magnetic::HysteresisInductor& math() const noexcept {
        return math_;
    }
    [[nodiscard]] Real L_eff() const noexcept { return params_.L_eff; }
    [[nodiscard]] Real i_prev() const noexcept { return i_prev_; }
    [[nodiscard]] Real v_L_prev() const noexcept { return v_L_prev_; }
    /// Latest flux linkage (V·s) integrated through the J-A state.
    [[nodiscard]] Real flux() const noexcept { return math_.flux(); }
    /// Hysteretic material magnetization (A/m).
    [[nodiscard]] Real magnetization() const noexcept {
        return math_.magnetization();
    }
    /// Irreversible (loop-pinning) component of the magnetization.
    [[nodiscard]] Real magnetization_irreversible() const noexcept {
        return math_.magnetization_irreversible();
    }

private:
    Params params_{};
    magnetic::HysteresisInductor math_{};
    Index branch_index_ = -1;
    Real i_prev_   = 0.0;
    Real v_L_prev_ = 0.0;
};

}  // namespace pulsim::v1
