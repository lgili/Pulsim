#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Example: Current Source Device (CRTP)
// =============================================================================

class CurrentSource : public LinearDeviceBase<CurrentSource> {
public:
    using Base = LinearDeviceBase<CurrentSource>;
    static constexpr std::size_t num_pins = 2;
    static constexpr int device_type = static_cast<int>(DeviceType::CurrentSource);

    struct Params {
        Scalar current = 0.0;
    };

    explicit CurrentSource(Scalar current, std::string name = "")
        : Base(std::move(name)), current_(current) {}

    /// Stamp implementation - current source only affects RHS
    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& /*G*/, Vec& b, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 2) return;

        const NodeIndex n_plus = nodes[0];
        const NodeIndex n_minus = nodes[1];

        // Current flows from n+ to n- (conventional)
        // KCL: current entering n+ is positive
        if (n_plus >= 0) {
            b[n_plus] -= current_;  // Current leaves n+
        }
        if (n_minus >= 0) {
            b[n_minus] += current_;  // Current enters n-
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        // Current source doesn't affect G matrix, only RHS
        return StaticSparsityPattern<0>{{}};
    }

    [[nodiscard]] Scalar current() const { return current_; }
    void set_current(Scalar i) { current_ = i; }

private:
    Scalar current_;
};

template<>
struct device_traits<CurrentSource> {
    static constexpr DeviceType type = DeviceType::CurrentSource;
    static constexpr std::size_t num_pins = 2;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = true;
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = false;
    static constexpr bool has_thermal_model = false;
    static constexpr std::size_t jacobian_size = 0;
};

}  // namespace pulsim::v1
