#pragma once

#include "pulsim/v1/components/base.hpp"
#include "pulsim/v1/ad/ad_scalar.hpp"

#include <cmath>

namespace pulsim::v1 {

// =============================================================================
// Voltage Controlled Switch (3-terminal: control, t1, t2)
// =============================================================================
//
// V-controlled switch suitable for PWM-driven converter cells. ON when the
// control-node voltage exceeds v_threshold.
//
//  - Behavioral mode: tanh-smoothed transition over `hysteresis` width.
//                     Adds a derivative-of-conductance term to the Jacobian
//                     so Newton converges across the threshold.
//
//  - Ideal mode    : sharp two-state stamp. Conductance is exactly g_on or
//                     g_off based on pwl_state_; no derivative-of-conductance
//                     contribution. The kernel commutes via the event
//                     scheduler (should_commute() / commit_pwl_state()).

class VoltageControlledSwitch : public NonlinearDeviceBase<VoltageControlledSwitch> {
public:
    using Base = NonlinearDeviceBase<VoltageControlledSwitch>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::Switch);

    struct Params {
        Scalar v_threshold = 2.5;
        Scalar g_on = 1e3;
        Scalar g_off = 1e-9;
        Scalar hysteresis = 0.1;  ///< Behavioral-mode tanh smoothing width (V).
    };

    explicit VoltageControlledSwitch(Scalar v_th = 2.5, Scalar g_on = 1e3, Scalar g_off = 1e-9,
                                     std::string name = "")
        : Base(std::move(name)), v_threshold_(v_th), g_on_(g_on), g_off_(g_off) {}

    explicit VoltageControlledSwitch(Params params, std::string name = "")
        : Base(std::move(name)), v_threshold_(params.v_threshold),
          g_on_(params.g_on), g_off_(params.g_off), hysteresis_(params.hysteresis) {}

    // --- SwitchingMode contract -----------------------------------------------
    [[nodiscard]] SwitchingMode switching_mode() const noexcept { return mode_; }
    void set_switching_mode(SwitchingMode mode) noexcept { mode_ = mode; }

    [[nodiscard]] Scalar event_hysteresis() const noexcept { return event_hysteresis_; }
    void set_event_hysteresis(Scalar h) noexcept { event_hysteresis_ = h; }

    // --- PWL two-state contract -----------------------------------------------
    [[nodiscard]] bool pwl_state() const noexcept { return pwl_state_; }
    void commit_pwl_state(bool closed) noexcept { pwl_state_ = closed; }

    /// Commute when the control-node voltage crosses v_threshold (with
    /// hysteresis). Direction is determined by the current state.
    [[nodiscard]] bool should_commute(const PwlEventContext& ctx) const noexcept {
        const Scalar h = std::max<Scalar>(ctx.event_hysteresis, event_hysteresis_);
        return pwl_state_
            ? (ctx.control_voltage < v_threshold_ - h)
            : (ctx.control_voltage > v_threshold_ + h);
    }

    // --- Stamping --------------------------------------------------------------

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
        const NodeIndex n_t1 = nodes[1];
        const NodeIndex n_t2 = nodes[2];

        // For the linear-only stamp (no Newton context): trust the committed
        // pwl_state_ if explicitly Ideal, else fall back to off-state for
        // safe DC initialization (legacy behavior).
        const SwitchingMode active_mode = resolve_switching_mode(mode_);
        const Scalar g = (active_mode == SwitchingMode::Ideal && pwl_state_)
            ? g_on_
            : g_off_;

        if (n_t1 >= 0) {
            G.coeffRef(n_t1, n_t1) += g;
            if (n_t2 >= 0) G.coeffRef(n_t1, n_t2) -= g;
        }
        if (n_t2 >= 0) {
            G.coeffRef(n_t2, n_t2) += g;
            if (n_t1 >= 0) G.coeffRef(n_t2, n_t1) -= g;
        }
    }

    [[nodiscard]] Scalar v_threshold() const { return v_threshold_; }
    [[nodiscard]] Scalar g_on() const { return g_on_; }
    [[nodiscard]] Scalar g_off() const { return g_off_; }
    [[nodiscard]] Scalar hysteresis() const { return hysteresis_; }

    // ---- Phase 2 of `add-automatic-differentiation` --------------------------
    //
    // Templated current expression for the Behavioral VCSwitch model:
    //   sigmoid = ½ · (1 + tanh((v_ctrl − v_th) / hysteresis))
    //   g       = g_off + (g_on − g_off) · sigmoid
    //   i_sw    = g · (v_t1 − v_t2)
    // Same single-source-of-truth shared between the manual `stamp_jacobian_behavioral`
    // and the AD-driven `stamp_jacobian_via_ad`. All physical coefficients
    // stay as `Real` so AD's derivative chain is preserved.
    template <typename S>
    [[nodiscard]] S switch_current_behavioral(S v_ctrl, S v_t1, S v_t2) const {
        using std::tanh;
        const S v_norm = (v_ctrl - v_threshold_) / hysteresis_;
        const S tanh_val = tanh(v_norm);
        const S sigmoid = Real{0.5} * (Real{1.0} + tanh_val);
        const S g = g_off_ + (g_on_ - g_off_) * sigmoid;
        const S v_sw = v_t1 - v_t2;
        return g * v_sw;
    }

    /// AD-derived stamp — standard form (matching VCSwitch's manual stamp,
    /// which carries `f[t1] += i_sw` and `J[t1, *] = ∂i/∂x`, NOT the Norton
    /// companion `i_eq` form used by MOSFET/IGBT). AD-derived partials cover
    /// the implicit derivative of the conductance through `tanh`.
    template <typename Matrix, typename Vec>
    void stamp_jacobian_via_ad(Matrix& J, Vec& f, const Vec& x,
                               std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;
        const NodeIndex n_ctrl = nodes[0];
        const NodeIndex n_t1 = nodes[1];
        const NodeIndex n_t2 = nodes[2];

        const Scalar v_ctrl = (n_ctrl >= 0) ? x[n_ctrl] : Scalar{0.0};
        const Scalar v_t1 = (n_t1 >= 0) ? x[n_t1] : Scalar{0.0};
        const Scalar v_t2 = (n_t2 >= 0) ? x[n_t2] : Scalar{0.0};

        auto seeded = ad::seed_from_values({v_ctrl, v_t1, v_t2});
        const ad::ADReal i_ad = switch_current_behavioral<ad::ADReal>(
            seeded[0], seeded[1], seeded[2]);

        // Mirror manual stamp side-effect.
        pwl_state_ = (v_ctrl > v_threshold_);

        const Scalar i_sw = i_ad.value();
        const Scalar di_dvctrl = (i_ad.derivatives().size() > 0)
            ? Scalar{i_ad.derivatives()[0]} : Scalar{0.0};
        const Scalar di_dvt1 = (i_ad.derivatives().size() > 1)
            ? Scalar{i_ad.derivatives()[1]} : Scalar{0.0};
        const Scalar di_dvt2 = (i_ad.derivatives().size() > 2)
            ? Scalar{i_ad.derivatives()[2]} : Scalar{0.0};

        // Standard-form J entries: t1 row carries +∂i/∂x; t2 row mirrors with
        // the sign flipped (current entering t2 = current leaving t1).
        if (n_t1 >= 0) {
            J.coeffRef(n_t1, n_t1) += di_dvt1;
            if (n_t2 >= 0)   J.coeffRef(n_t1, n_t2)   += di_dvt2;
            if (n_ctrl >= 0) J.coeffRef(n_t1, n_ctrl) += di_dvctrl;
        }
        if (n_t2 >= 0) {
            J.coeffRef(n_t2, n_t2) -= di_dvt2;
            if (n_t1 >= 0)   J.coeffRef(n_t2, n_t1)   -= di_dvt1;
            if (n_ctrl >= 0) J.coeffRef(n_t2, n_ctrl) -= di_dvctrl;
        }

        // Standard-form residual contribution.
        if (n_t1 >= 0) f[n_t1] += i_sw;
        if (n_t2 >= 0) f[n_t2] -= i_sw;
    }

private:
    // --- Behavioral (tanh-smoothed) stamp -------------------------------------
    template<typename Matrix, typename Vec>
    void stamp_jacobian_behavioral(Matrix& J, Vec& f, const Vec& x,
                                   std::span<const NodeIndex> nodes) {
        const NodeIndex n_ctrl = nodes[0];
        const NodeIndex n_t1 = nodes[1];
        const NodeIndex n_t2 = nodes[2];

        const Scalar v_ctrl = (n_ctrl >= 0) ? x[n_ctrl] : Scalar{0.0};
        const Scalar v_t1 = (n_t1 >= 0) ? x[n_t1] : Scalar{0.0};
        const Scalar v_t2 = (n_t2 >= 0) ? x[n_t2] : Scalar{0.0};

        // g = g_off + (g_on - g_off) * sigmoid((v_ctrl - v_th) / hysteresis)
        const Scalar v_norm = (v_ctrl - v_threshold_) / hysteresis_;
        const Scalar tanh_val = std::tanh(v_norm);
        const Scalar sigmoid = Scalar{0.5} * (Scalar{1.0} + tanh_val);
        const Scalar g = g_off_ + (g_on_ - g_off_) * sigmoid;

        const Scalar dsigmoid = Scalar{0.5} / hysteresis_ *
                                (Scalar{1.0} - tanh_val * tanh_val);
        const Scalar dg_dvctrl = (g_on_ - g_off_) * dsigmoid;

        const Scalar v_sw = v_t1 - v_t2;
        const Scalar i_sw = g * v_sw;

        // Update the soft state estimate (used for telemetry; PWL kernel
        // does not consume this in Behavioral mode).
        pwl_state_ = (sigmoid > Scalar{0.5});

        if (n_t1 >= 0) {
            J.coeffRef(n_t1, n_t1) += g;
            if (n_t2 >= 0) J.coeffRef(n_t1, n_t2) -= g;
            if (n_ctrl >= 0) J.coeffRef(n_t1, n_ctrl) += dg_dvctrl * v_sw;
        }
        if (n_t2 >= 0) {
            J.coeffRef(n_t2, n_t2) += g;
            if (n_t1 >= 0) J.coeffRef(n_t2, n_t1) -= g;
            if (n_ctrl >= 0) J.coeffRef(n_t2, n_ctrl) -= dg_dvctrl * v_sw;
        }

        if (n_t1 >= 0) f[n_t1] += i_sw;
        if (n_t2 >= 0) f[n_t2] -= i_sw;
    }

    // --- Ideal (sharp PWL) stamp ----------------------------------------------
    template<typename Matrix, typename Vec>
    void stamp_jacobian_ideal(Matrix& J, Vec& f, const Vec& x,
                              std::span<const NodeIndex> nodes) const {
        const NodeIndex n_t1 = nodes[1];
        const NodeIndex n_t2 = nodes[2];

        const Scalar v_t1 = (n_t1 >= 0) ? x[n_t1] : Scalar{0.0};
        const Scalar v_t2 = (n_t2 >= 0) ? x[n_t2] : Scalar{0.0};
        const Scalar v_sw = v_t1 - v_t2;

        const Scalar g = pwl_state_ ? g_on_ : g_off_;
        const Scalar i_sw = g * v_sw;

        if (n_t1 >= 0) {
            J.coeffRef(n_t1, n_t1) += g;
            if (n_t2 >= 0) J.coeffRef(n_t1, n_t2) -= g;
        }
        if (n_t2 >= 0) {
            J.coeffRef(n_t2, n_t2) += g;
            if (n_t1 >= 0) J.coeffRef(n_t2, n_t1) -= g;
        }

        if (n_t1 >= 0) f[n_t1] += i_sw;
        if (n_t2 >= 0) f[n_t2] -= i_sw;
    }

    Scalar v_threshold_ = 2.5;
    Scalar g_on_ = 1e3;
    Scalar g_off_ = 1e-9;
    Scalar hysteresis_ = 0.5;             ///< Behavioral-mode tanh width.
    Scalar event_hysteresis_ = Scalar{1e-9}; ///< PWL-mode commute hysteresis (V).
    SwitchingMode mode_ = SwitchingMode::Auto;
    bool pwl_state_ = false;
};

template<>
struct device_traits<VoltageControlledSwitch> {
    static constexpr DeviceType type = DeviceType::Switch;
    static constexpr std::size_t num_pins = 3;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear (depends on control voltage)
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = false;
    static constexpr bool supports_pwl = true;
    static constexpr std::size_t jacobian_size = 9;  // 3x3
};

}  // namespace pulsim::v1
