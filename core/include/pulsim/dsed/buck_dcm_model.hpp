#pragma once

// =============================================================================
// Pulsim — PED engine: synchronous-buck DCM model (Gate 3 reference).
// =============================================================================
//
// Three-mode buck converter for Discontinuous Conduction Mode:
//
//   * Mode A (HS_ON)         — V_in drives the inductor; i_L ramps UP
//   * Mode B (LS_CONDUCTING) — synchronous rectifier; i_L ramps DOWN
//   * Mode C (ZERO_CURRENT)  — body diode blocks reverse current;
//                              i_L clamped to 0; only v_C decays
//
// Mode-transition events:
//
//   A → B  : HS gate falling edge (scheduled).
//   B → C  : i_L = 0 crossing (Illinois root-find).
//   C → A  : HS gate rising edge (scheduled).
//
// Analytical DCM buck regulator equation (Erickson & Maksimovic
// 3rd ed. §5.2.3 eq. 5.44):
//
//   K       = 2·L / (R_load · T_sw)
//   K_crit  = 1 - D
//   DCM     when K < K_crit
//   M(D, K) = V_out / V_in = 2 / (1 + sqrt(1 + 4·K/D²))
//
// For (V_in=24, D=0.5, L=100µH, C=100µF, R_load=240Ω, f_sw=100 kHz):
//   K = 0.0833, K_crit = 0.5, M ≈ 0.7913, V_out_steady ≈ 19.0 V.
//
// Direct port of `prototype/dsed/buck_dcm_model.py` validated against
// the analytical regulator: V_out error = 0.0043 % (well within the
// 1 % Gate 3A target).

#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>

#include "pulsim/dsed/event_predictor.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::dsed::buck {

/// Three operating modes for a sync-buck in DCM.
enum class BuckMode : std::uint8_t {
    HS_ON = 0,           // High-side switch conducting
    LS_CONDUCTING = 1,   // Low-side conducting (i_L > 0)
    ZERO_CURRENT = 2,    // Body diode blocking, i_L = 0
};

/// Sync-buck operating point + passives, sized for DCM by default.
struct BuckDCMParams {
    Real V_in     = Real{24};
    Real L        = Real{100e-6};
    Real C        = Real{100e-6};
    Real R_load   = Real{240};      // 100× CCM → deep DCM at D=0.5
    Real f_sw     = Real{100e3};
    Real D        = Real{0.5};

    [[nodiscard]] Real T_sw() const noexcept {
        return Real{1} / f_sw;
    }
    /// Erickson's DCM design constant: K = 2L/(R·T).
    [[nodiscard]] Real K() const noexcept {
        return Real{2} * L / (R_load * T_sw());
    }
    /// DCM/CCM boundary value for a buck: K_crit = 1 - D.
    [[nodiscard]] Real K_crit() const noexcept {
        return Real{1} - D;
    }
    /// Analytical DCM conversion ratio M = 2 / (1 + sqrt(1 + 4K/D²)).
    [[nodiscard]] Real M_dcm() const noexcept {
        const Real k = K();
        return Real{2}
             / (Real{1} + std::sqrt(Real{1} + Real{4} * k / (D * D)));
    }
    /// CCM reference ratio M_ccm = D.
    [[nodiscard]] Real M_ccm() const noexcept { return D; }
    /// Analytical DCM steady-state output voltage.
    [[nodiscard]] Real V_out_steady() const noexcept {
        return M_dcm() * V_in;
    }
    /// CCM-equivalent steady-state V_out (for cross-check).
    [[nodiscard]] Real V_out_ccm_steady() const noexcept {
        return M_ccm() * V_in;
    }
    /// Operating-point check (Erickson eq. 5.46).
    [[nodiscard]] bool is_dcm_at_steady_state() const noexcept {
        return K() < K_crit();
    }
};

/// 3-mode sync-buck DCM ODE source.
///
/// State: x = (i_L, v_C) ∈ R²
/// Mode is mutated externally by the PED scheduler's event handler.
class BuckDCMModel {
public:
    explicit BuckDCMModel(BuckDCMParams params) : p_{params} {
        const Real L = p_.L;
        const Real C = p_.C;
        const Real R = p_.R_load;
        // A matrix shared between HS_ON and LS_CONDUCTING; only b differs.
        A_(0, 0) = Real{0};
        A_(0, 1) = -Real{1} / L;
        A_(1, 0) = Real{1} / C;
        A_(1, 1) = -Real{1} / (R * C);
        // b vectors per mode.
        b_HSon_  = Vector(2);
        b_HSon_(0) = p_.V_in / L;
        b_HSon_(1) = Real{0};
        b_LSc_   = Vector(2);
        b_LSc_(0) = Real{0};
        b_LSc_(1) = Real{0};
        // Mode C: rank-reduced — i_L = 0 algebraically; only v_C decays.
        A_zero_(0, 0) = Real{0};
        A_zero_(0, 1) = Real{0};
        A_zero_(1, 0) = Real{0};
        A_zero_(1, 1) = -Real{1} / (R * C);
    }

    [[nodiscard]] BuckMode current_mask() const noexcept { return mode_; }
    void set_mask(BuckMode m) noexcept { mode_ = m; }

    [[nodiscard]] const BuckDCMParams& params() const noexcept { return p_; }

    /// RHS dispatched on current mode (LTI per mode → ``t`` unused).
    [[nodiscard]] Vector rhs(Real /*t*/, const Vector& x) const {
        if (mode_ == BuckMode::HS_ON) {
            return A_ * x + b_HSon_;
        }
        if (mode_ == BuckMode::LS_CONDUCTING) {
            return A_ * x + b_LSc_;
        }
        // ZERO_CURRENT: i_L held at 0; only v_C decays through R_load.
        return A_zero_ * x;
    }

    /// Current mode's A matrix (satisfies `HasLTIPerMode` for the
    /// auto-dispatch scheduler). HS_ON and LS_CONDUCTING share the
    /// same A (only the b vector differs); ZERO_CURRENT uses the
    /// rank-reduced A_zero (i_L row clamped to 0).
    [[nodiscard]] DenseMatrix A_matrix() const {
        return (mode_ == BuckMode::ZERO_CURRENT) ? A_zero_ : A_;
    }

    /// Current mode's b vector (constant within each mode; ``t`` unused).
    [[nodiscard]] Vector b_vector(Real /*t*/) const {
        if (mode_ == BuckMode::HS_ON) return b_HSon_;
        if (mode_ == BuckMode::LS_CONDUCTING) return b_LSc_;
        // ZERO_CURRENT: zero forcing (cap decays through R_load only)
        return Vector::Zero(2);
    }

private:
    BuckDCMParams p_;
    DenseMatrix A_{2, 2};
    DenseMatrix A_zero_{2, 2};
    Vector b_HSon_;
    Vector b_LSc_;
    BuckMode mode_ = BuckMode::HS_ON;
};

/// Switch-function + mode-schedule callable for the PED scheduler.
///
/// Knows about gate edges analytically (via ``next_edge_after``) and
/// receives notifications from the scheduler when a ZCD predicate
/// fires (via ``register_zcd_transition``) so subsequent
/// ``operator()(t)`` calls return ZERO_CURRENT for the remainder of
/// the cycle.
class BuckDCMSwitchFn {
public:
    BuckDCMSwitchFn(Real T_sw, Real D) noexcept : T_sw_{T_sw}, D_{D} {}

    /// Returns the active mode at time t given gate schedule + ZCD history.
    [[nodiscard]] BuckMode operator()(Real t) const {
        const Real phase = std::fmod(t / T_sw_, Real{1});
        const auto cycle = cycle_idx_(t);
        if (phase < D_) return BuckMode::HS_ON;
        // Past HS-off boundary in this cycle.
        auto it = zcd_history_.find(cycle);
        if (it != zcd_history_.end() && t > it->second) {
            return BuckMode::ZERO_CURRENT;
        }
        return BuckMode::LS_CONDUCTING;
    }

    /// Next *gate* edge time (does not include ZCD events).
    [[nodiscard]] Real next_edge_after(Real t) const {
        const auto k = cycle_idx_(t);
        const Real T  = T_sw_;
        const std::array<Real, 3> candidates = {
            static_cast<Real>(k) * T + D_ * T,           // HS-off
            static_cast<Real>(k + 1) * T,                // HS-on next cycle
            static_cast<Real>(k + 1) * T + D_ * T,
        };
        constexpr Real kEps = Real{1e-15};
        for (Real c : candidates) {
            if (c > t + kEps) return c;
        }
        return static_cast<Real>(k + 2) * T;
    }

    /// Called by the PED event handler when a ZCD fires in this cycle.
    void register_zcd_transition(Real t) {
        zcd_history_[cycle_idx_(t)] = t;
    }

    /// For tests: clear all ZCD history (e.g. when re-running scenarios).
    void reset_zcd_history() noexcept { zcd_history_.clear(); }

    /// For tests: how many ZCD events have been registered.
    [[nodiscard]] std::size_t zcd_count() const noexcept {
        return zcd_history_.size();
    }

private:
    [[nodiscard]] std::int64_t cycle_idx_(Real t) const {
        return static_cast<std::int64_t>(std::floor(t / T_sw_));
    }

    Real T_sw_;
    Real D_;
    std::unordered_map<std::int64_t, Real> zcd_history_;
};

/// Build the inductor-current zero-crossing predicate for the PED scheduler.
///
/// Returns a ``PredicateFn`` (signature ``Real(Real, const Vector&)``) that
/// reports x[0] = i_L. Sign change from positive to negative is the ZCD.
/// The predicate is purely state-driven; ``t`` is unused.
[[nodiscard]] inline PredicateFn make_zcd_predicate() {
    return [](Real /*t*/, const Vector& x) -> Real {
        return x(0);
    };
}

/// State-projection callback to clamp i_L = 0 on ZCD (algebraic
/// constraint of ZERO_CURRENT mode). Gate 3's analytical seed for the
/// Gate 4 partial_refactor-based projection.
[[nodiscard]] inline StateProjectionFn make_zcd_projection() {
    return [](Real /*t*/, const Vector& x, PredicateType ptype) -> Vector {
        if (ptype == PredicateType::CurrentZC) {
            Vector y = x;
            y(0) = Real{0};
            return y;
        }
        return x;
    };
}

}  // namespace pulsim::dsed::buck
