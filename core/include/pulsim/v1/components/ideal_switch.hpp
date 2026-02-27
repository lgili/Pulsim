#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Example: Ideal Switch Device (CRTP)
// =============================================================================

class IdealSwitch : public LinearDeviceBase<IdealSwitch> {
public:
    using Base = LinearDeviceBase<IdealSwitch>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Switch);

    struct Params {
        Scalar g_on = 1e6;     // On-state conductance
        Scalar g_off = 1e-12;  // Off-state conductance
        bool initial_state = false;
    };

    explicit IdealSwitch(Scalar g_on = 1e6, Scalar g_off = 1e-12, bool closed = false, std::string name = "")
        : Base(std::move(name)), g_on_(g_on), g_off_(g_off), is_closed_(closed) {}

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        Scalar g = is_closed_ ? g_on_ : g_off_;

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

    void close() { is_closed_ = true; }
    void open() { is_closed_ = false; }
    void set_state(bool closed) { is_closed_ = closed; }
    [[nodiscard]] bool is_closed() const { return is_closed_; }
    [[nodiscard]] Scalar g_on() const { return g_on_; }
    [[nodiscard]] Scalar g_off() const { return g_off_; }

private:
    Scalar g_on_;
    Scalar g_off_;
    bool is_closed_;
};

template<>
struct device_traits<IdealSwitch> {
    static constexpr DeviceType type = DeviceType::Switch;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;  // Piecewise linear
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;
};

}  // namespace pulsim::v1
