#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Example: Voltage Source Device (CRTP)
// =============================================================================

class VoltageSource : public LinearDeviceBase<VoltageSource> {
public:
    using Base = LinearDeviceBase<VoltageSource>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::VoltageSource);

    struct Params {
        Scalar voltage = 0.0;
    };

    explicit VoltageSource(Scalar voltage, std::string name = "")
        : Base(std::move(name)), voltage_(voltage), branch_index_(-1) {}

    /// Set the branch index (MNA requires extra row/col for voltage sources)
    void set_branch_index(NodeIndex idx) { branch_index_ = idx; }
    [[nodiscard]] NodeIndex branch_index() const { return branch_index_; }

    /// Stamp implementation
    /// MNA formulation: adds equation V+ - V- = V_source
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2 || branch_index_ < 0) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];
        const NodeIndex br = branch_index_;

        // Stamp the MNA extension for voltage source
        // Row br: V+ - V- = V_source
        // Also affects KCL: current flows from n+ to n-
        if (n_plus >= 0) {
            G.coeffRef(n_plus, br) += 1.0;
            G.coeffRef(br, n_plus) += 1.0;
        }
        if (n_minus >= 0) {
            G.coeffRef(n_minus, br) -= 1.0;
            G.coeffRef(br, n_minus) -= 1.0;
        }

        // RHS: voltage value
        b[br] = voltage_;
    }

    static constexpr auto jacobian_pattern_impl() {
        // Voltage source adds 4 entries plus diagonal
        return StaticSparsityPattern<5>{{
            JacobianEntry{0, 2},  // n+ to branch
            JacobianEntry{1, 2},  // n- to branch
            JacobianEntry{2, 0},  // branch to n+
            JacobianEntry{2, 1},  // branch to n-
            JacobianEntry{2, 2}   // branch diagonal (zero but pattern exists)
        }};
    }

    [[nodiscard]] Scalar voltage() const { return voltage_; }
    void set_voltage(Scalar v) { voltage_ = v; }

private:
    Scalar voltage_;
    NodeIndex branch_index_;
};

template<>
struct device_traits<VoltageSource> {
    static constexpr DeviceType type = DeviceType::VoltageSource;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 1;  // Branch current
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 5;
};

}  // namespace pulsim::v1
