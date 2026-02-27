#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Example: Resistor Device (CRTP)
// =============================================================================

class Resistor : public LinearDeviceBase<Resistor> {
public:
    using Base = LinearDeviceBase<Resistor>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::Resistor);

    /// Parameter structure for Resistor
    struct Params {
        Scalar resistance = 1000.0;
    };

    explicit Resistor(Scalar resistance, std::string name = "")
        : Base(std::move(name)), resistance_(resistance) {}

    /// Stamp implementation (called via CRTP)
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const Scalar g = 1.0 / resistance_;

        // Stamp conductance matrix
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

    /// Jacobian pattern (compile-time)
    static constexpr auto jacobian_pattern_impl() {
        // Resistor contributes to 4 positions: (n+,n+), (n+,n-), (n-,n+), (n-,n-)
        return StaticSparsityPattern<4>{{
            JacobianEntry{0, 0},
            JacobianEntry{0, 1},
            JacobianEntry{1, 0},
            JacobianEntry{1, 1}
        }};
    }

    [[nodiscard]] Scalar resistance() const { return resistance_; }
    void set_resistance(Scalar r) { resistance_ = r; }

private:
    Scalar resistance_;
};

// Specialization of device_traits for Resistor
template<>
struct device_traits<Resistor> {
    static constexpr DeviceType type = DeviceType::Resistor;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;  // Conduction loss = I²R
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 4;  // 2x2 contribution
};

}  // namespace pulsim::v1
