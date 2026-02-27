#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Example: Ideal Diode Device (CRTP - Nonlinear)
// =============================================================================

class IdealDiode : public NonlinearDeviceBase<IdealDiode> {
public:
    using Base = NonlinearDeviceBase<IdealDiode>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Diode);

    struct Params {
        Scalar g_on = 1e3;     // On-state conductance (1/R_on)
        Scalar g_off = 1e-9;   // Off-state conductance (leakage)
        Scalar v_threshold = 0.0;  // Threshold voltage (0 for ideal)
        Scalar v_smooth = 0.1;     // Smoothing voltage for transition (0 = sharp)
    };

    explicit IdealDiode(Scalar g_on = 1e3, Scalar g_off = 1e-9, std::string name = "")
        : Base(std::move(name)), g_on_(g_on), g_off_(g_off), v_smooth_(0.1), is_on_(false) {}

    /// Set smoothing voltage (higher = smoother transition, 0 = sharp)
    void set_smoothing(Scalar v_smooth) { v_smooth_ = v_smooth; }
    [[nodiscard]] Scalar smoothing() const { return v_smooth_; }
    [[nodiscard]] Scalar g_on() const { return g_on_; }
    [[nodiscard]] Scalar g_off() const { return g_off_; }

    /// Stamp Jacobian for Newton iteration with smooth transition
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];   // Anode
        const NodeIndex n_minus = nodes[1];  // Cathode

        // Get voltage across diode
        Scalar v_anode = (n_plus >= 0) ? x[n_plus] : 0.0;
        Scalar v_cathode = (n_minus >= 0) ? x[n_minus] : 0.0;
        Scalar v_diode = v_anode - v_cathode;

        Scalar g, i_diode, dg_dv;

        if (v_smooth_ > 0) {
            // Smooth transition using tanh
            // g varies smoothly from g_off to g_on as v_diode goes from negative to positive
            Scalar alpha = std::tanh(v_diode / v_smooth_);  // ranges from -1 to 1
            Scalar g_avg = (g_on_ + g_off_) / 2.0;
            Scalar g_diff = (g_on_ - g_off_) / 2.0;
            g = g_avg + g_diff * alpha;

            // Derivative of conductance w.r.t. voltage (for improved Jacobian)
            Scalar dalpha_dv = (1.0 - alpha * alpha) / v_smooth_;  // derivative of tanh
            dg_dv = g_diff * dalpha_dv;

            // Current: i = g * v
            i_diode = g * v_diode;

            // Update state indicator
            is_on_ = (alpha > 0);
        } else {
            // Sharp transition (original behavior)
            is_on_ = (v_diode > 0.0);
            g = is_on_ ? g_on_ : g_off_;
            dg_dv = 0.0;
            i_diode = g * v_diode;
        }

        // Stamp conductance (with additional term from dg/dv for smooth model)
        // The Jacobian of i = g(v)*v is: di/dv = g + v*dg/dv
        Scalar J_diag = g + v_diode * dg_dv;

        if (n_plus >= 0) {
            J.coeffRef(n_plus, n_plus) += J_diag;
            if (n_minus >= 0) {
                J.coeffRef(n_plus, n_minus) -= J_diag;
            }
        }
        if (n_minus >= 0) {
            J.coeffRef(n_minus, n_minus) += J_diag;
            if (n_plus >= 0) {
                J.coeffRef(n_minus, n_plus) -= J_diag;
            }
        }

        // Update residual (KCL contribution)
        if (n_plus >= 0) {
            f[n_plus] += i_diode;
        }
        if (n_minus >= 0) {
            f[n_minus] -= i_diode;
        }
    }

    /// Regular stamp for linear-like behavior
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        Scalar g = is_on_ ? g_on_ : g_off_;

        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g;
            }
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g;
            }
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] bool is_conducting() const { return is_on_; }

private:
    Scalar g_on_;
    Scalar g_off_;
    Scalar v_smooth_;  // Smoothing voltage (0 = sharp transition)
    mutable bool is_on_;
};

template<>
struct device_traits<IdealDiode> {
    static constexpr DeviceType type = DeviceType::Diode;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear!
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;
};

}  // namespace pulsim::v1
