#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Example: Inductor Device (CRTP with dynamics)
// =============================================================================

class Inductor : public DynamicDeviceBase<Inductor> {
public:
    using Base = DynamicDeviceBase<Inductor>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Inductor);

    struct Params {
        Scalar inductance = 1e-3;
        Scalar initial_current = 0.0;
    };

    explicit Inductor(Scalar inductance, Scalar initial_current = 0.0, std::string name = "")
        : Base(std::move(name))
        , inductance_(inductance)
        , i_prev_(initial_current)
        , v_prev_(0.0) {}

    /// Stamp implementation using Trapezoidal companion model
    /// For inductor: V = L * dI/dt
    /// Trapezoidal: V_n = (2L/dt) * I_n - (2L/dt) * I_{n-1} - V_{n-1}
    /// Companion model is a resistor R_eq = 2L/dt in series with voltage V_eq
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        // Trapezoidal: R_eq = 2L/dt (conductance g_eq = dt/(2L))
        const Scalar g_eq = dt_ / (2.0 * inductance_);

        // Equivalent voltage: V_eq = (2L/dt) * I_{n-1} + V_{n-1}
        const Scalar v_eq = (2.0 * inductance_ / dt_) * i_prev_ + v_prev_;

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

        // Stamp equivalent current source (from V_eq through g_eq)
        const Scalar i_eq = g_eq * v_eq;
        if (n_plus >= 0) {
            b[n_plus] += i_eq;
        }
        if (n_minus >= 0) {
            b[n_minus] -= i_eq;
        }
    }

    void update_history_impl() {
        i_prev_ = i_current_;
        v_prev_ = v_current_;
    }

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

    [[nodiscard]] Scalar inductance() const { return inductance_; }
    [[nodiscard]] Scalar current_prev() const { return i_prev_; }
    [[nodiscard]] Scalar voltage_prev() const { return v_prev_; }

private:
    Scalar inductance_;
    Scalar i_prev_;
    Scalar v_prev_;
    Scalar i_current_ = 0.0;
    Scalar v_current_ = 0.0;
};

template<>
struct device_traits<Inductor> {
    static constexpr DeviceType type = DeviceType::Inductor;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = true;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;
};

}  // namespace pulsim::v1
