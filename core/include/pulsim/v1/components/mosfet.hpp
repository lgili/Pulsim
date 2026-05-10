#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/ad/ad_scalar.hpp"

#include <cmath>

namespace pulsim::v1 {

// =============================================================================
// MOSFET Device (CRTP - Nonlinear, 3-terminal)
// =============================================================================
//
// Behavioral mode: Level-1 Shichman-Hodges (cutoff / triode / saturation).
// Ideal      mode: piecewise-linear two-state model gated by Vgs vs Vth.
//                  Drain-source path becomes a linear `Rds_on = 1/g_on` when
//                  on and `Roff = 1/g_off` when off. Body diode is not yet
//                  embedded in the Ideal stamp (planned in a follow-up
//                  change; the Behavioral tier remains the source of truth
//                  for body-diode reverse recovery analyses for now).
//
// In Ideal mode pwl_state_ is the canonical on/off bit; the kernel mutates it
// via commit_pwl_state() at events located by should_commute() (Vgs threshold).
// In Behavioral mode the same bit shadows the Shichman-Hodges region for
// telemetry parity.
//
/// MOSFET Level 1 model (Shichman-Hodges) plus PWL Ideal alternative.
/// Terminals: Gate (0), Drain (1), Source (2)
class MOSFET : public NonlinearDeviceBase<MOSFET> {
public:
    using Base = NonlinearDeviceBase<MOSFET>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::MOSFET);

    struct Params {
        Scalar vth = 2.0;           // Threshold voltage (V)
        Scalar kp = 0.1;            // Transconductance parameter (A/V^2)
        Scalar lambda = 0.01;       // Channel-length modulation (1/V)
        Scalar g_off = 1e-12;       // Off-state conductance
        bool is_nmos = true;      // NMOS if true, PMOS if false
        Scalar g_on = 1e3;          // On-state conductance for Ideal mode (1/Rds_on)
    };

    explicit MOSFET(std::string name = "")
        : Base(std::move(name)), params_() {}

    explicit MOSFET(Params params, std::string name)
        : Base(std::move(name)), params_(params) {}

    explicit MOSFET(Scalar vth, Scalar kp, bool is_nmos = true, std::string name = "")
        : Base(std::move(name))
        , params_{vth, kp, 0.01, 1e-12, is_nmos, 1e3} {}

    // --- SwitchingMode contract -----------------------------------------------
    [[nodiscard]] SwitchingMode switching_mode() const noexcept { return mode_; }
    void set_switching_mode(SwitchingMode mode) noexcept { mode_ = mode; }

    [[nodiscard]] Scalar event_hysteresis() const noexcept { return event_hysteresis_; }
    void set_event_hysteresis(Scalar h) noexcept { event_hysteresis_ = h; }

    // --- PWL two-state contract -----------------------------------------------
    [[nodiscard]] bool pwl_state() const noexcept { return pwl_state_; }
    void commit_pwl_state(bool on) noexcept { pwl_state_ = on; }

    /// Commute when Vgs crosses the threshold. NMOS turns on when Vgs > Vth;
    /// PMOS turns on when Vgs < -Vth (sign already folded into ctx via the
    /// is_nmos-aware caller, or detected here from params_).
    [[nodiscard]] bool should_commute(const PwlEventContext& ctx) const noexcept {
        const Scalar h = std::max<Scalar>(ctx.event_hysteresis, event_hysteresis_);
        const Scalar vgs_signed = params_.is_nmos ? ctx.control_voltage
                                                   : -ctx.control_voltage;
        return pwl_state_
            ? (vgs_signed < params_.vth - h)
            : (vgs_signed > params_.vth + h);
    }

    // --- Stamping --------------------------------------------------------------

    /// Stamp Jacobian for Newton iteration
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) {
            return;
        }
        const SwitchingMode active_mode = resolve_switching_mode(mode_);
        if (active_mode == SwitchingMode::Ideal) {
            stamp_jacobian_ideal(J, f, x, nodes);
        } else {
#ifdef PULSIM_USE_AD_STAMP
            stamp_jacobian_via_ad(J, f, x, nodes);
#else
            stamp_jacobian_behavioral(J, f, x, nodes);
#endif
        }
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        // For initial guess, stamp small conductance.
        if (nodes.size() < 3) return;
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        const Scalar g = (resolve_switching_mode(mode_) == SwitchingMode::Ideal && pwl_state_)
            ? params_.g_on
            : params_.g_off;

        if (n_drain >= 0) {
            G.coeffRef(n_drain, n_drain) += g;
            if (n_source >= 0) G.coeffRef(n_drain, n_source) -= g;
        }
        if (n_source >= 0) {
            G.coeffRef(n_source, n_source) += g;
            if (n_drain >= 0) G.coeffRef(n_source, n_drain) -= g;
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        // 3x3 = 9 entries max, but we mainly use D-S path
        return StaticSparsityPattern<9>{{
            JacobianEntry{0, 0}, JacobianEntry{0, 1}, JacobianEntry{0, 2},
            JacobianEntry{1, 0}, JacobianEntry{1, 1}, JacobianEntry{1, 2},
            JacobianEntry{2, 0}, JacobianEntry{2, 1}, JacobianEntry{2, 2}
        }};
    }

    [[nodiscard]] const Params& params() const { return params_; }
    [[nodiscard]] bool is_conducting() const noexcept { return pwl_state_; }

    // ---- Phase 2 of `add-automatic-differentiation` --------------------------
    //
    // Templated drain-current expression for the Behavioral (Shichman-Hodges)
    // model. Identical math to `stamp_jacobian_behavioral` but evaluated on a
    // generic scalar so AD can derive ∂id/∂(v_g, v_d, v_s) automatically.
    //
    // All physical coefficients (`vth`, `kp`, `lambda`, `g_off`, `sign`) stay
    // as `Real` per the Phase 1 plumbing notes; only the terminal voltages
    // are `S`. This protects the derivative chain when `S = ADReal`.
    template <typename S>
    [[nodiscard]] S drain_current_behavioral(S v_g, S v_d, S v_s) const {
        const Real sign = params_.is_nmos ? Real{1.0} : Real{-1.0};
        const S vgs = sign * (v_g - v_s);
        const S vds = sign * (v_d - v_s);
        const Real vth = params_.vth;
        const Real kp = params_.kp;
        const Real lambda = params_.lambda;

        S id;
        if (vgs <= vth) {
            // Cutoff region: small linear leakage.
            id = params_.g_off * vds;
        } else if (vds < vgs - vth) {
            // Triode (linear) region.
            const S vov = vgs - vth;
            id = kp * (vov * vds - Real{0.5} * vds * vds)
                * (Real{1.0} + lambda * vds);
        } else {
            // Saturation region.
            const S vov = vgs - vth;
            id = Real{0.5} * kp * vov * vov * (Real{1.0} + lambda * vds);
        }
        return sign * id;
    }

    /// AD-derived stamp of the Behavioral residual + Jacobian. Replicates
    /// the manual `stamp_jacobian_behavioral` Norton companion form
    /// (`i_eq = id − Σ ∂id/∂x_i · x_i`) so cross-validation against the
    /// manual stamp passes within floating-point precision at every
    /// operating point.
    template <typename Matrix, typename Vec>
    void stamp_jacobian_via_ad(Matrix& J, Vec& f, const Vec& x,
                               std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        const Scalar v_g = (n_gate >= 0) ? x[n_gate] : Scalar{0.0};
        const Scalar v_d = (n_drain >= 0) ? x[n_drain] : Scalar{0.0};
        const Scalar v_s = (n_source >= 0) ? x[n_source] : Scalar{0.0};

        auto seeded = ad::seed_from_values({v_g, v_d, v_s});
        const ad::ADReal id_ad =
            drain_current_behavioral<ad::ADReal>(seeded[0], seeded[1], seeded[2]);

        // Mirror manual stamp side-effect: pwl_state_ tracks (region != cutoff).
        const Real sign = params_.is_nmos ? Real{1.0} : Real{-1.0};
        const Scalar vgs_signed = sign * (v_g - v_s);
        pwl_state_ = (vgs_signed > params_.vth);

        const Scalar id = id_ad.value();
        const Scalar di_dvg = (id_ad.derivatives().size() > 0)
            ? Scalar{id_ad.derivatives()[0]} : Scalar{0.0};
        const Scalar di_dvd = (id_ad.derivatives().size() > 1)
            ? Scalar{id_ad.derivatives()[1]} : Scalar{0.0};
        const Scalar di_dvs = (id_ad.derivatives().size() > 2)
            ? Scalar{id_ad.derivatives()[2]} : Scalar{0.0};

        // Norton companion offset (Taylor-residual form):
        //   i_eq = id − ∇id · x
        //        = id − gm·vgs − gds·vds  (manual form, after change of basis
        //                                  via vgs = v_g − v_s, vds = v_d − v_s)
        const Scalar i_eq = id - di_dvg * v_g - di_dvd * v_d - di_dvs * v_s;

        // Drain row: + ∂id/∂x_i.
        if (n_drain >= 0) {
            J.coeffRef(n_drain, n_drain) += di_dvd;
            if (n_gate >= 0)   J.coeffRef(n_drain, n_gate)   += di_dvg;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) += di_dvs;
        }
        // Source row: − ∂id/∂x_i (current-leaving convention reversed).
        if (n_source >= 0) {
            if (n_drain >= 0) J.coeffRef(n_source, n_drain) -= di_dvd;
            if (n_gate >= 0)  J.coeffRef(n_source, n_gate)  -= di_dvg;
            J.coeffRef(n_source, n_source) -= di_dvs;
        }

        // Norton companion residual contribution.
        if (n_drain >= 0)  f[n_drain]  -= i_eq;
        if (n_source >= 0) f[n_source] += i_eq;
    }

private:
    // --- Behavioral (Shichman-Hodges) Jacobian stamp --------------------------
    template<typename Matrix, typename Vec>
    void stamp_jacobian_behavioral(Matrix& J, Vec& f, const Vec& x,
                                   std::span<const NodeIndex> nodes) {
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        // Get terminal voltages.
        const Scalar vg = (n_gate >= 0) ? x[n_gate] : Scalar{0.0};
        const Scalar vd = (n_drain >= 0) ? x[n_drain] : Scalar{0.0};
        const Scalar vs = (n_source >= 0) ? x[n_source] : Scalar{0.0};

        // For PMOS, negate voltages.
        const Scalar sign = params_.is_nmos ? Scalar{1.0} : Scalar{-1.0};
        const Scalar vgs = sign * (vg - vs);
        const Scalar vds = sign * (vd - vs);

        Scalar id = Scalar{0.0};   // Drain current
        Scalar gm = Scalar{0.0};   // Transconductance dId/dVgs
        Scalar gds = Scalar{0.0};  // Output conductance dId/dVds

        const Scalar vth = params_.vth;
        const Scalar kp = params_.kp;
        const Scalar lambda = params_.lambda;

        if (vgs <= vth) {
            // Cutoff region.
            id = params_.g_off * vds;
            gds = params_.g_off;
            pwl_state_ = false;
        } else if (vds < vgs - vth) {
            // Linear (triode) region.
            const Scalar vov = vgs - vth;
            id = kp * (vov * vds - Scalar{0.5} * vds * vds) * (Scalar{1.0} + lambda * vds);
            gm = kp * vds * (Scalar{1.0} + lambda * vds);
            gds = kp * (vov - vds) * (Scalar{1.0} + lambda * vds)
                + kp * (vov * vds - Scalar{0.5} * vds * vds) * lambda;
            pwl_state_ = true;
        } else {
            // Saturation region.
            const Scalar vov = vgs - vth;
            id = Scalar{0.5} * kp * vov * vov * (Scalar{1.0} + lambda * vds);
            gm = kp * vov * (Scalar{1.0} + lambda * vds);
            gds = Scalar{0.5} * kp * vov * vov * lambda;
            pwl_state_ = true;
        }

        // Apply sign for PMOS.
        id *= sign;

        // Stamp Jacobian (Norton equivalent).
        // I_eq = id - gm * vgs - gds * vds
        const Scalar i_eq = id - gm * vgs - gds * vds;

        // Conductance stamps: drain-source path.
        if (n_drain >= 0) {
            J.coeffRef(n_drain, n_drain) += gds;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) -= gds;
            if (n_gate >= 0) J.coeffRef(n_drain, n_gate) += gm;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) -= gm;  // gm contribution
        }
        if (n_source >= 0) {
            J.coeffRef(n_source, n_source) += gds;
            if (n_drain >= 0) J.coeffRef(n_source, n_drain) -= gds;
            if (n_gate >= 0) J.coeffRef(n_source, n_gate) -= gm;
            J.coeffRef(n_source, n_source) += gm;  // gm contribution
        }

        // Current source stamps.
        if (n_drain >= 0) f[n_drain] -= i_eq;
        if (n_source >= 0) f[n_source] += i_eq;
    }

    // --- Ideal (PWL two-state) Jacobian stamp ---------------------------------
    template<typename Matrix, typename Vec>
    void stamp_jacobian_ideal(Matrix& J, Vec& f, const Vec& x,
                              std::span<const NodeIndex> nodes) const {
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        const Scalar vd = (n_drain >= 0) ? x[n_drain] : Scalar{0.0};
        const Scalar vs = (n_source >= 0) ? x[n_source] : Scalar{0.0};
        const Scalar vds = vd - vs;

        const Scalar g = pwl_state_ ? params_.g_on : params_.g_off;
        const Scalar id = g * vds;

        // Pure drain-source resistive stamp; no gm contribution.
        if (n_drain >= 0) {
            J.coeffRef(n_drain, n_drain) += g;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) -= g;
        }
        if (n_source >= 0) {
            J.coeffRef(n_source, n_source) += g;
            if (n_drain >= 0) J.coeffRef(n_source, n_drain) -= g;
        }

        if (n_drain >= 0) f[n_drain] -= id;
        if (n_source >= 0) f[n_source] += id;
    }

    Params params_;
    Scalar event_hysteresis_ = Scalar{1e-9};
    SwitchingMode mode_ = SwitchingMode::Auto;
    bool pwl_state_ = false;
};

template<>
struct device_traits<MOSFET> {
    static constexpr DeviceType type = DeviceType::MOSFET;
    static constexpr std::size_t num_pins = 3;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = true;
    static constexpr bool supports_pwl = true;
    static constexpr std::size_t jacobian_size = 9;
};

}  // namespace pulsim::v1
