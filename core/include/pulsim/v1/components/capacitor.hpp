#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Example: Capacitor Device (CRTP with dynamics)
// =============================================================================

class Capacitor : public DynamicDeviceBase<Capacitor> {
public:
    using Base = DynamicDeviceBase<Capacitor>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Capacitor);

    /// Parameter structure for Capacitor
    struct Params {
        Scalar capacitance = 1e-6;
        Scalar initial_voltage = 0.0;
    };

    explicit Capacitor(Scalar capacitance, Scalar initial_voltage = 0.0, std::string name = "")
        : Base(std::move(name))
        , capacitance_(capacitance)
        , v_prev_(initial_voltage)
        , i_prev_(0.0) {}

    /// Stamp implementation using Trapezoidal companion model
    /// Correct formula: I_eq = (2C/dt) * V_n - (2C/dt) * V_{n-1} - I_{n-1}
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        // Trapezoidal: G_eq = 2C/dt
        const Scalar g_eq = 2.0 * capacitance_ / dt_;

        // Equivalent current source: I_eq = G_eq * V_{n-1} + I_{n-1}
        const Scalar i_eq = g_eq * v_prev_ + i_prev_;

        // Stamp conductance
        if (n_plus >= 0) {
            G.coeffRef(n_plus, n_plus) += g_eq;
            if (n_minus >= 0) {
                G.coeffRef(n_plus, n_minus) -= g_eq;
            }
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, n_minus) += g_eq;
            if (n_plus >= 0) {
                G.coeffRef(n_minus, n_plus) -= g_eq;
            }
        }

        // Stamp equivalent current source
        if (n_plus >= 0) {
            b[n_plus] += i_eq;
        }
        if (n_minus >= 0) {
            b[n_minus] -= i_eq;
        }
    }

    /// Update history after accepted timestep
    void update_history_impl() {
        // Store current values for next step
        // Note: v_current and i_current must be set by the solver after each step
        v_prev_ = v_current_;
        i_prev_ = i_current_;
    }

    /// Set current state (called by solver after each Newton iteration)
    void set_current_state(Scalar v, Scalar i) {
        v_current_ = v;
        i_current_ = i;
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] Scalar capacitance() const { return capacitance_; }
    [[nodiscard]] Scalar voltage_prev() const { return v_prev_; }
    [[nodiscard]] Scalar current_prev() const { return i_prev_; }

private:
    Scalar capacitance_;
    Scalar v_prev_;      // Voltage at previous timestep
    Scalar i_prev_;      // Current at previous timestep (CRITICAL for Trapezoidal!)
    Scalar v_current_ = 0.0;  // Current voltage (set by solver)
    Scalar i_current_ = 0.0;  // Current current (set by solver)
};

template<>
struct device_traits<Capacitor> {
    static constexpr DeviceType type = DeviceType::Capacitor;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = true;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;
};

}  // namespace pulsim::v1
