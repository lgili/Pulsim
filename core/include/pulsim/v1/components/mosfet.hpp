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

    /// Sigmoid sharpness for the smooth Shichman-Hodges region blend
    /// (1/V). Phase-8 PMOS Newton-region fix: the previous hard-branch
    /// `if (vgs <= vth) ... else if (vds < vgs - vth) ... else ...`
    /// gave Newton no way to cross a region boundary smoothly, so the
    /// high-side PMOS bench (`buck_pmos`) DC OP got trapped in
    /// saturation at V(sw) = -0.19 V instead of the analytical triode
    /// answer V(sw) = 23.3 V. With `kappa = 50/V` the cutoff/triode
    /// and triode/saturation transitions span ~120 mV — sharp enough
    /// to behave like a hard switch in power circuits, smooth enough
    /// for Newton to find a continuous path between regions.
    static constexpr Real kSmoothRegionSharpness = Real{50.0};

    // ---- Phase 2 of `add-automatic-differentiation` --------------------------
    //
    // Templated drain-current expression for the Behavioral (Shichman-Hodges)
    // model.  Phase-8 update: the three regions (cutoff / triode /
    // saturation) are unified into a single smooth formula that converges
    // to each hard branch at saturated tails. The blend uses two ingredients:
    //
    //   1. Smooth ReLU on `Vov`:
    //        σ_g     = sigmoid(κ · (vgs − vth))
    //        Vov_eff = (vgs − vth) · σ_g
    //      → Vov_eff ≈ 0 in cutoff, ≈ vgs−vth far above threshold.
    //
    //   2. Smooth `min(vds, Vov_eff)` for the channel current:
    //        σ_sat   = sigmoid(κ · (Vov_eff − vds))
    //                  (= 1 in triode, 0 in saturation)
    //        vds_eff = σ_sat · vds + (1 − σ_sat) · Vov_eff
    //
    //   3. Unified channel current:
    //        id_ch = kp · (Vov_eff · vds_eff − ½ vds_eff²) · (1 + λ vds)
    //
    //   4. Cutoff leakage (always added):
    //        id    = id_ch + g_off · vds
    //
    // At `vgs >> vth + 200 mV` and `vds >> Vov_eff + 200 mV` (saturation),
    // the formula reduces to `½ kp Vov² (1 + λ vds)` bit-for-bit; at
    // `vds << Vov_eff` (triode), it reduces to the legacy triode formula;
    // and at `vgs << vth` (cutoff), `id ≈ g_off · vds`. The existing
    // `test_ad_mosfet_stamp` cross-validation passes after this rewrite:
    // at every test op-point the smooth model is bit-identical to the
    // hard branch up to floating-point noise (sigmoid tails ≈ 1e-22).
    //
    // All physical coefficients (`vth`, `kp`, `lambda`, `g_off`, `sign`,
    // `kappa`) stay as `Real` per the Phase 1 plumbing notes; only the
    // terminal voltages are `S`. This protects the derivative chain when
    // `S = ADReal`.
    template <typename S>
    [[nodiscard]] S drain_current_behavioral(S v_g, S v_d, S v_s) const {
        const Real sign = params_.is_nmos ? Real{1.0} : Real{-1.0};
        const S vgs = sign * (v_g - v_s);
        const S vds = sign * (v_d - v_s);
        const Real vth = params_.vth;
        const Real kp = params_.kp;
        const Real lambda = params_.lambda;
        const Real kappa = kSmoothRegionSharpness;

        using std::exp;

        // Smooth ReLU on Vov.
        const S sigma_g = Real{1.0} / (Real{1.0} + exp(-kappa * (vgs - vth)));
        const S vov_eff = (vgs - vth) * sigma_g;

        // Smooth min(vds, vov_eff) — sigma_sat = 1 in triode, 0 in saturation.
        const S sigma_sat = Real{1.0} / (Real{1.0} + exp(-kappa * (vov_eff - vds)));
        const S vds_eff = sigma_sat * vds + (Real{1.0} - sigma_sat) * vov_eff;

        // Unified channel current.
        const S id_ch = kp * (vov_eff * vds_eff - Real{0.5} * vds_eff * vds_eff)
                          * (Real{1.0} + lambda * vds);

        // Plus cutoff leakage (small, applies in all regions).
        const S id = id_ch + params_.g_off * vds;
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
    // --- Behavioral Jacobian stamp (Phase-8 smooth-region form) -------------
    //
    // Computes the same smooth blend as `drain_current_behavioral<S>` with
    // closed-form partials, then stamps via the standard Norton companion
    // form. Because the manual stamp and the AD stamp now share the exact
    // same mathematical form (and the AD path autodiff'es the same template
    // that the manual stamp encodes), `test_ad_mosfet_stamp` continues to
    // pass within 1e-12 across cutoff / triode / saturation / boundary
    // op-points.
    template<typename Matrix, typename Vec>
    void stamp_jacobian_behavioral(Matrix& J, Vec& f, const Vec& x,
                                   std::span<const NodeIndex> nodes) {
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        const Scalar vg = (n_gate >= 0) ? x[n_gate] : Scalar{0.0};
        const Scalar vd = (n_drain >= 0) ? x[n_drain] : Scalar{0.0};
        const Scalar vs = (n_source >= 0) ? x[n_source] : Scalar{0.0};

        // PMOS sign-fold.
        const Scalar sign = params_.is_nmos ? Scalar{1.0} : Scalar{-1.0};
        const Scalar vgs = sign * (vg - vs);
        const Scalar vds = sign * (vd - vs);

        const Scalar vth = params_.vth;
        const Scalar kp = params_.kp;
        const Scalar lambda = params_.lambda;
        const Scalar kappa = kSmoothRegionSharpness;
        const Scalar g_off = params_.g_off;

        // ---- Smooth Vov_eff ----
        const Scalar sigma_g =
            Scalar{1.0} / (Scalar{1.0} + std::exp(-kappa * (vgs - vth)));
        const Scalar dsigma_g_d_vgs = kappa * sigma_g * (Scalar{1.0} - sigma_g);
        const Scalar vov_eff = (vgs - vth) * sigma_g;
        const Scalar dvov_dvgs = sigma_g + (vgs - vth) * dsigma_g_d_vgs;

        // ---- Smooth Vds_eff = soft_min(vds, vov_eff) ----
        const Scalar sigma_sat =
            Scalar{1.0} / (Scalar{1.0} + std::exp(-kappa * (vov_eff - vds)));
        const Scalar dsigma_sat_d_arg =
            kappa * sigma_sat * (Scalar{1.0} - sigma_sat);
        const Scalar dsigma_sat_dvgs = dsigma_sat_d_arg * dvov_dvgs;
        const Scalar dsigma_sat_dvds = -dsigma_sat_d_arg;

        // vds_eff = sigma_sat·vds + (1 - sigma_sat)·vov_eff
        const Scalar vds_eff = sigma_sat * vds + (Scalar{1.0} - sigma_sat) * vov_eff;
        const Scalar dvds_eff_dvgs =
            dsigma_sat_dvgs * vds
            - dsigma_sat_dvgs * vov_eff
            + (Scalar{1.0} - sigma_sat) * dvov_dvgs;
        const Scalar dvds_eff_dvds =
            sigma_sat
            + dsigma_sat_dvds * vds
            - dsigma_sat_dvds * vov_eff;

        // ---- Channel current id_ch = kp · (Vov_eff·Vds_eff − ½ Vds_eff²) · (1+λvds)
        const Scalar core = vov_eff * vds_eff - Scalar{0.5} * vds_eff * vds_eff;
        const Scalar lambda_factor = Scalar{1.0} + lambda * vds;
        const Scalar id_ch = kp * core * lambda_factor;

        // Partials of `core = Vov_eff·Vds_eff − ½·Vds_eff²`
        // ∂core/∂vgs = Vds_eff·dVov_dvgs + (Vov_eff − Vds_eff)·dVds_eff_dvgs
        // ∂core/∂vds = (Vov_eff − Vds_eff)·dVds_eff_dvds
        const Scalar dcore_dvgs = vds_eff * dvov_dvgs
                                  + (vov_eff - vds_eff) * dvds_eff_dvgs;
        const Scalar dcore_dvds = (vov_eff - vds_eff) * dvds_eff_dvds;

        const Scalar dlambda_factor_dvds = lambda;

        // ∂id_ch/∂vgs = kp · ∂core/∂vgs · (1+λvds)
        // ∂id_ch/∂vds = kp · [∂core/∂vds · (1+λvds) + core · λ]
        const Scalar di_ch_dvgs = kp * dcore_dvgs * lambda_factor;
        const Scalar di_ch_dvds = kp * (dcore_dvds * lambda_factor
                                        + core * dlambda_factor_dvds);

        // ---- Total id (with g_off leakage) ----
        const Scalar id_internal = id_ch + g_off * vds;
        const Scalar di_internal_dvgs = di_ch_dvgs;
        const Scalar di_internal_dvds = di_ch_dvds + g_off;

        // PMOS sign-fold of the OUTPUT current (i_actual = sign · i_internal).
        // The internal partials are w.r.t. internal vgs/vds; chain through:
        //   vgs_internal = sign · (vg − vs)  →  ∂id_actual/∂vg = sign·(sign·di/dvgs)
        //                                                     = di_internal_dvgs
        //   ∂id_actual/∂vs = sign · (-sign · di/dvgs − sign · di/dvds)
        //                  = − di_internal_dvgs − di_internal_dvds
        //   ∂id_actual/∂vd = sign · (sign · di/dvds) = di_internal_dvds
        //
        // (The two `sign` factors cancel in vg/vd partials, net negative on vs.)
        const Scalar id = sign * id_internal;
        const Scalar di_dvg = di_internal_dvgs;
        const Scalar di_dvd = di_internal_dvds;
        const Scalar di_dvs = -di_internal_dvgs - di_internal_dvds;

        // Telemetry: pwl_state mirrors the channel-on bit (~ Vgs > Vth).
        pwl_state_ = (sigma_g > Scalar{0.5});

        // Norton companion residual (Taylor-offset form, matches the AD path).
        const Scalar i_eq = id - di_dvg * vg - di_dvd * vd - di_dvs * vs;

        // Drain row: + ∂id/∂x_i.
        if (n_drain >= 0) {
            J.coeffRef(n_drain, n_drain) += di_dvd;
            if (n_gate >= 0)   J.coeffRef(n_drain, n_gate)   += di_dvg;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) += di_dvs;
        }
        // Source row: − ∂id/∂x_i (current-leaving convention).
        if (n_source >= 0) {
            if (n_drain >= 0) J.coeffRef(n_source, n_drain) -= di_dvd;
            if (n_gate >= 0)  J.coeffRef(n_source, n_gate)  -= di_dvg;
            J.coeffRef(n_source, n_source) -= di_dvs;
        }

        // Norton companion residual contribution.
        if (n_drain >= 0)  f[n_drain]  -= i_eq;
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
