#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Voltage Controlled Switch (3-terminal: control, t1, t2)
// =============================================================================

/// Voltage-controlled switch - ON when V(control) > threshold
/// Ideal for PWM-driven switching applications (Buck, etc.)
/// Terminals: Control (0), Terminal1 (1), Terminal2 (2)
class VoltageControlledSwitch : public NonlinearDeviceBase<VoltageControlledSwitch> {
public:
    using Base = NonlinearDeviceBase<VoltageControlledSwitch>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::Switch);

    struct Params {
        Scalar v_threshold = 2.5;  // Threshold voltage
        Scalar g_on = 1e3;         // On-state conductance (1mΩ)
        Scalar g_off = 1e-9;       // Off-state conductance
        Scalar hysteresis = 0.1;   // Hysteresis for smooth transition
    };

    explicit VoltageControlledSwitch(Scalar v_th = 2.5, Scalar g_on = 1e3, Scalar g_off = 1e-9,
                                     std::string name = "")
        : Base(std::move(name)), v_threshold_(v_th), g_on_(g_on), g_off_(g_off) {}

    explicit VoltageControlledSwitch(Params params, std::string name = "")
        : Base(std::move(name)), v_threshold_(params.v_threshold),
          g_on_(params.g_on), g_off_(params.g_off), hysteresis_(params.hysteresis) {}

    /// Stamp Jacobian for Newton iteration
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;

        const NodeIndex n_ctrl = nodes[0];
        const NodeIndex n_t1 = nodes[1];
        const NodeIndex n_t2 = nodes[2];

        // Get voltages
        Scalar v_ctrl = (n_ctrl >= 0) ? x[n_ctrl] : 0.0;
        Scalar v_t1 = (n_t1 >= 0) ? x[n_t1] : 0.0;
        Scalar v_t2 = (n_t2 >= 0) ? x[n_t2] : 0.0;

        // Smooth switching function using tanh
        // g = g_off + (g_on - g_off) * sigmoid((v_ctrl - v_th) / hysteresis)
        Scalar v_norm = (v_ctrl - v_threshold_) / hysteresis_;
        const Scalar tanh_val = std::tanh(v_norm);
        Scalar sigmoid = 0.5 * (1.0 + tanh_val);
        Scalar g = g_off_ + (g_on_ - g_off_) * sigmoid;

        // Derivative of g with respect to v_ctrl
        Scalar dsigmoid = 0.5 / hysteresis_ * (1.0 - tanh_val * tanh_val);
        Scalar dg_dvctrl = (g_on_ - g_off_) * dsigmoid;

        // Current through switch: i = g * (v_t1 - v_t2)
        Scalar v_sw = v_t1 - v_t2;
        Scalar i_sw = g * v_sw;

        // Stamp conductance between t1 and t2
        if (n_t1 >= 0) {
            J.coeffRef(n_t1, n_t1) += g;
            if (n_t2 >= 0) J.coeffRef(n_t1, n_t2) -= g;
            // Derivative w.r.t. control voltage
            if (n_ctrl >= 0) J.coeffRef(n_t1, n_ctrl) += dg_dvctrl * v_sw;
        }
        if (n_t2 >= 0) {
            J.coeffRef(n_t2, n_t2) += g;
            if (n_t1 >= 0) J.coeffRef(n_t2, n_t1) -= g;
            // Derivative w.r.t. control voltage
            if (n_ctrl >= 0) J.coeffRef(n_t2, n_ctrl) -= dg_dvctrl * v_sw;
        }

        // Current residuals
        if (n_t1 >= 0) f[n_t1] += i_sw;
        if (n_t2 >= 0) f[n_t2] -= i_sw;
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        // For DC/initial: assume switch is OFF (safe starting point)
        if (nodes.size() < 3) return;
        const NodeIndex n_t1 = nodes[1];
        const NodeIndex n_t2 = nodes[2];

        Scalar g = g_off_;
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

private:
    Scalar v_threshold_ = 2.5;
    Scalar g_on_ = 1e3;
    Scalar g_off_ = 1e-9;
    Scalar hysteresis_ = 0.5;  // Wider hysteresis for smoother convergence
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
    static constexpr std::size_t jacobian_size = 9;  // 3x3
};

}  // namespace pulsim::v1
