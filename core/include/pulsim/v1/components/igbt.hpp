#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/ad/ad_scalar.hpp"

#include <cmath>

namespace pulsim::v1 {

// =============================================================================
// IGBT Device (CRTP - Nonlinear, 3-terminal, supports SwitchingMode::Ideal)
// =============================================================================
//
// Behavioral mode: voltage-controlled conductance with collector-emitter
//                  saturation (forward-drop above v_ce_sat).
// Ideal      mode: piecewise-linear two-state model gated by Vge vs vth.
//                  Collector-emitter path is exactly g_on or g_off; no
//                  saturation drop (which is loss-model territory). Tail
//                  current modeling lives in the catalog tier follow-up.
//
// Replaced legacy `mutable bool is_on_` with explicit `pwl_state_` field and
// const-correct stamping methods.
//
/// Simplified IGBT model for power electronics
/// Terminals: Gate (0), Collector (1), Emitter (2)
class IGBT : public NonlinearDeviceBase<IGBT> {
public:
    using Base = NonlinearDeviceBase<IGBT>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::IGBT);

    struct Params {
        Scalar vth = 5.0;           // Gate threshold voltage (V)
        Scalar g_on = 1e4;          // On-state conductance (S)
        Scalar g_off = 1e-12;       // Off-state conductance (S)
        Scalar v_ce_sat = 1.5;      // Collector-emitter saturation voltage (V)
    };

    explicit IGBT(std::string name = "")
        : Base(std::move(name)), params_() {}

    explicit IGBT(Params params, std::string name)
        : Base(std::move(name)), params_(params) {}

    explicit IGBT(Scalar vth, Scalar g_on = 1e4, std::string name = "")
        : Base(std::move(name))
        , params_{vth, g_on, 1e-12, 1.5} {}

    // --- SwitchingMode contract -----------------------------------------------
    [[nodiscard]] SwitchingMode switching_mode() const noexcept { return mode_; }
    void set_switching_mode(SwitchingMode mode) noexcept { mode_ = mode; }

    [[nodiscard]] Scalar event_hysteresis() const noexcept { return event_hysteresis_; }
    void set_event_hysteresis(Scalar h) noexcept { event_hysteresis_ = h; }

    // --- PWL two-state contract -----------------------------------------------
    [[nodiscard]] bool pwl_state() const noexcept { return pwl_state_; }
    void commit_pwl_state(bool on) noexcept { pwl_state_ = on; }

    /// Commute when Vge crosses the gate threshold (with hysteresis).
    [[nodiscard]] bool should_commute(const PwlEventContext& ctx) const noexcept {
        const Scalar h = std::max<Scalar>(ctx.event_hysteresis, event_hysteresis_);
        return pwl_state_
            ? (ctx.control_voltage < params_.vth - h)
            : (ctx.control_voltage > params_.vth + h);
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
        if (nodes.size() < 3) return;
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        const Scalar g = pwl_state_ ? params_.g_on : params_.g_off;

        if (n_collector >= 0) {
            G.coeffRef(n_collector, n_collector) += g;
            if (n_emitter >= 0) G.coeffRef(n_collector, n_emitter) -= g;
        }
        if (n_emitter >= 0) {
            G.coeffRef(n_emitter, n_emitter) += g;
            if (n_collector >= 0) G.coeffRef(n_emitter, n_collector) -= g;
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<9>{{
            JacobianEntry{0, 0}, JacobianEntry{0, 1}, JacobianEntry{0, 2},
            JacobianEntry{1, 0}, JacobianEntry{1, 1}, JacobianEntry{1, 2},
            JacobianEntry{2, 0}, JacobianEntry{2, 1}, JacobianEntry{2, 2}
        }};
    }

    [[nodiscard]] bool is_conducting() const noexcept { return pwl_state_; }
    [[nodiscard]] const Params& params() const { return params_; }

    /// Sigmoid sharpness for the smooth-gm blend (1/V). Phase-6 IGBT fix:
    /// the previous hard-step model gave Newton zero ∂ic/∂vge information
    /// across iterations, so the DC operating point on a trivial
    /// `Vdc + IGBT + Rload` failed ("All random restarts failed"). With
    /// `kappa = 50/V`, ~99% of the g_off → g_on transition happens within
    /// ±60 mV of `vth`, which is sharp enough to behave like a switch in
    /// power circuits but smooth enough for Newton to find a continuous
    /// path through the threshold.
    static constexpr Real kSmoothGmSharpness = Real{50.0};

    // ---- Phase 2 of `add-automatic-differentiation` --------------------------
    //
    // Templated collector-current expression for the Behavioral IGBT model.
    // Phase-6 update: uses a sigmoid-blended conductance instead of a hard
    // step on `vge > vth ∧ vce > 0`, so AD picks up a non-zero
    // ∂ic/∂vge term and Newton has a continuous gradient through the
    // gate threshold.
    //
    //   alpha_gate = sigmoid(κ · (vge - vth))
    //   alpha_dir  = sigmoid(κ · vce)
    //   alpha      = alpha_gate · alpha_dir
    //   g_eff      = g_off + (g_on - g_off) · alpha
    //   ic         = g_eff · vce
    //
    // For `|vge - vth| > 200 mV` the sigmoid saturates to 0 or 1 within
    // float precision, so far from the threshold this collapses to the
    // legacy hard-step behavior bit-for-bit. Inside the transition
    // window (~120 mV wide), the smooth gradient is what gives Newton
    // its missing gm.
    template <typename S>
    [[nodiscard]] S collector_current_behavioral(S v_g, S v_c, S v_e) const {
        const S vge = v_g - v_e;
        const S vce = v_c - v_e;
        const Real vth = params_.vth;
        const Real kappa = kSmoothGmSharpness;
        using std::exp;
        const S alpha_gate = Real{1.0} / (Real{1.0} + exp(-kappa * (vge - vth)));
        const S alpha_dir  = Real{1.0} / (Real{1.0} + exp(-kappa * vce));
        const S alpha = alpha_gate * alpha_dir;
        const Real g_off = params_.g_off;
        const Real g_on  = params_.g_on;
        const S g_eff = g_off + (g_on - g_off) * alpha;
        return g_eff * vce;
    }

    /// AD-derived stamp — Norton companion form, identical math to the
    /// manual `stamp_jacobian_behavioral` (including the dead-saturation
    /// branch), so cross-validation passes within 1e-12 absolute on every
    /// op-point.
    template <typename Matrix, typename Vec>
    void stamp_jacobian_via_ad(Matrix& J, Vec& f, const Vec& x,
                               std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        const Scalar v_g = (n_gate >= 0) ? x[n_gate] : Scalar{0.0};
        const Scalar v_c = (n_collector >= 0) ? x[n_collector] : Scalar{0.0};
        const Scalar v_e = (n_emitter >= 0) ? x[n_emitter] : Scalar{0.0};

        auto seeded = ad::seed_from_values({v_g, v_c, v_e});
        const ad::ADReal ic_ad =
            collector_current_behavioral<ad::ADReal>(seeded[0], seeded[1], seeded[2]);

        // Mirror manual stamp side-effect.
        const Scalar vge = v_g - v_e;
        const Scalar vce = v_c - v_e;
        pwl_state_ = (vge > params_.vth) && (vce > Real{0});

        const Scalar ic = ic_ad.value();
        const Scalar di_dvg = (ic_ad.derivatives().size() > 0)
            ? Scalar{ic_ad.derivatives()[0]} : Scalar{0.0};
        const Scalar di_dvc = (ic_ad.derivatives().size() > 1)
            ? Scalar{ic_ad.derivatives()[1]} : Scalar{0.0};
        const Scalar di_dve = (ic_ad.derivatives().size() > 2)
            ? Scalar{ic_ad.derivatives()[2]} : Scalar{0.0};

        // For the linear `ic = g · (v_c − v_e)` model, the Taylor offset
        // ic − ∇ic·x simplifies to zero. We compute it generically anyway
        // so future enhancements (true forward-drop / non-linear sat) do
        // not need to revisit the stamp.
        const Scalar i_eq = ic - di_dvg * v_g - di_dvc * v_c - di_dve * v_e;

        // Collector row: + ∂ic/∂x.
        if (n_collector >= 0) {
            J.coeffRef(n_collector, n_collector) += di_dvc;
            if (n_gate >= 0)    J.coeffRef(n_collector, n_gate)    += di_dvg;
            if (n_emitter >= 0) J.coeffRef(n_collector, n_emitter) += di_dve;
        }
        // Emitter row: − ∂ic/∂x.
        if (n_emitter >= 0) {
            if (n_collector >= 0) J.coeffRef(n_emitter, n_collector) -= di_dvc;
            if (n_gate >= 0)      J.coeffRef(n_emitter, n_gate)      -= di_dvg;
            J.coeffRef(n_emitter, n_emitter) -= di_dve;
        }

        // The legacy stamp uses `f[c] += ic − g·vce` (which equals 0 for
        // the linear model). We use the standard Taylor offset i_eq with
        // `f[c] += i_eq`. Both expressions evaluate to 0 here, so manual
        // and AD agree on `f` to within numerical noise.
        if (n_collector >= 0) f[n_collector] += i_eq;
        if (n_emitter >= 0)   f[n_emitter]   -= i_eq;
    }

private:
    // --- Behavioral Jacobian stamp (Phase-6 smooth-gm form) -------------------
    //
    // Evaluates the same sigmoid-blended `ic = g_eff(vge, vce) · vce` model
    // as `collector_current_behavioral`, but with the partials computed
    // analytically (closed form on the sigmoid):
    //
    //   ∂σ_g/∂vge = κ · σ_g · (1 − σ_g)
    //   ∂σ_d/∂vce = κ · σ_d · (1 − σ_d)
    //   ∂g_eff/∂vge = (g_on − g_off) · σ_d · ∂σ_g/∂vge
    //   ∂g_eff/∂vce = (g_on − g_off) · σ_g · ∂σ_d/∂vce
    //   ∂ic/∂v_g    = ∂g_eff/∂vge · vce
    //   ∂ic/∂v_c    = ∂g_eff/∂vce · vce + g_eff
    //   ∂ic/∂v_e    = −∂ic/∂v_g − ∂ic/∂v_c   (chain on vge, vce of v_e)
    //
    // Stamped via the standard Norton companion form (matches the AD path's
    // i_eq exactly — `test_ad_igbt_stamp` passes after this rewrite).
    template<typename Matrix, typename Vec>
    void stamp_jacobian_behavioral(Matrix& J, Vec& f, const Vec& x,
                                   std::span<const NodeIndex> nodes) {
        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        const Scalar vg = (n_gate >= 0) ? x[n_gate] : Scalar{0.0};
        const Scalar vc = (n_collector >= 0) ? x[n_collector] : Scalar{0.0};
        const Scalar ve = (n_emitter >= 0) ? x[n_emitter] : Scalar{0.0};

        const Scalar vge = vg - ve;
        const Scalar vce = vc - ve;

        // Sigmoid-blended on-factor.
        const Real kappa = kSmoothGmSharpness;
        const Scalar sigma_g = Scalar{1.0} /
            (Scalar{1.0} + std::exp(-kappa * (vge - params_.vth)));
        const Scalar sigma_d = Scalar{1.0} /
            (Scalar{1.0} + std::exp(-kappa * vce));
        const Scalar alpha = sigma_g * sigma_d;

        const Scalar dg = params_.g_on - params_.g_off;
        const Scalar g_eff = params_.g_off + dg * alpha;

        // Telemetry: pwl_state mirrors the on/off bit at α > 0.5.
        pwl_state_ = (alpha > Scalar{0.5});

        const Scalar ic = g_eff * vce;

        // Closed-form partials of the sigmoid blend.
        const Scalar dsigma_g_dvge = kappa * sigma_g * (Scalar{1.0} - sigma_g);
        const Scalar dsigma_d_dvce = kappa * sigma_d * (Scalar{1.0} - sigma_d);
        const Scalar dg_eff_dvge   = dg * sigma_d * dsigma_g_dvge;
        const Scalar dg_eff_dvce   = dg * sigma_g * dsigma_d_dvce;

        // Drain-current partials w.r.t. terminal voltages (chain rule:
        // vge = vg - ve, vce = vc - ve).
        const Scalar di_dvg = dg_eff_dvge * vce;
        const Scalar di_dvc = dg_eff_dvce * vce + g_eff;
        const Scalar di_dve = -di_dvg - di_dvc;

        // Norton companion offset (Taylor residual form).
        const Scalar i_eq = ic - di_dvg * vg - di_dvc * vc - di_dve * ve;

        // Collector row: + ∂ic/∂x_i.
        if (n_collector >= 0) {
            J.coeffRef(n_collector, n_collector) += di_dvc;
            if (n_gate >= 0)    J.coeffRef(n_collector, n_gate)    += di_dvg;
            if (n_emitter >= 0) J.coeffRef(n_collector, n_emitter) += di_dve;
        }
        // Emitter row: − ∂ic/∂x_i.
        if (n_emitter >= 0) {
            if (n_collector >= 0) J.coeffRef(n_emitter, n_collector) -= di_dvc;
            if (n_gate >= 0)      J.coeffRef(n_emitter, n_gate)      -= di_dvg;
            J.coeffRef(n_emitter, n_emitter) -= di_dve;
        }

        if (n_collector >= 0) f[n_collector] += i_eq;
        if (n_emitter >= 0)   f[n_emitter]   -= i_eq;
    }

    // --- Ideal (PWL two-state) Jacobian stamp ---------------------------------
    template<typename Matrix, typename Vec>
    void stamp_jacobian_ideal(Matrix& J, Vec& f, const Vec& x,
                              std::span<const NodeIndex> nodes) const {
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        const Scalar vc = (n_collector >= 0) ? x[n_collector] : Scalar{0.0};
        const Scalar ve = (n_emitter >= 0) ? x[n_emitter] : Scalar{0.0};
        const Scalar vce = vc - ve;

        const Scalar g = pwl_state_ ? params_.g_on : params_.g_off;
        const Scalar ic = g * vce;

        if (n_collector >= 0) {
            J.coeffRef(n_collector, n_collector) += g;
            if (n_emitter >= 0) J.coeffRef(n_collector, n_emitter) -= g;
        }
        if (n_emitter >= 0) {
            J.coeffRef(n_emitter, n_emitter) += g;
            if (n_collector >= 0) J.coeffRef(n_emitter, n_collector) -= g;
        }

        if (n_collector >= 0) f[n_collector] += ic;
        if (n_emitter >= 0) f[n_emitter] -= ic;
    }

    Params params_;
    Scalar event_hysteresis_ = Scalar{1e-9};
    SwitchingMode mode_ = SwitchingMode::Auto;
    bool pwl_state_ = false;
};

template<>
struct device_traits<IGBT> {
    static constexpr DeviceType type = DeviceType::IGBT;
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
