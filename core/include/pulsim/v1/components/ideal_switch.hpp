#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// Ideal Switch (CRTP - already piecewise-linear)
// =============================================================================
//
// IdealSwitch is intrinsically piecewise-linear: state is set externally
// (`close()`/`open()`/`set_state()`); stamping picks g_on or g_off accordingly.
// SwitchingMode controls only the conceptual labeling; the stamp is identical
// in Behavioral and Ideal modes (there is no smoothing here to begin with).
// The mode field is kept for API uniformity with diode/MOSFET/IGBT and to let
// the kernel discover that the device is PWL-eligible via supports_pwl.

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

    // --- State control --------------------------------------------------------
    void close() { is_closed_ = true; }
    void open() { is_closed_ = false; }
    void set_state(bool closed) { is_closed_ = closed; }
    [[nodiscard]] bool is_closed() const { return is_closed_; }
    [[nodiscard]] Scalar g_on() const { return g_on_; }
    [[nodiscard]] Scalar g_off() const { return g_off_; }

    // --- SwitchingMode contract -----------------------------------------------
    [[nodiscard]] SwitchingMode switching_mode() const noexcept { return mode_; }
    void set_switching_mode(SwitchingMode mode) noexcept { mode_ = mode; }

    // --- PWL two-state contract -----------------------------------------------
    /// Mirrors `is_closed()`. Provided for uniform PWL discovery.
    [[nodiscard]] bool pwl_state() const noexcept { return is_closed_; }
    /// Mirrors `set_state(closed)`. Provided for uniform PWL discovery.
    void commit_pwl_state(bool closed) noexcept { is_closed_ = closed; }

    /// IdealSwitch is externally commanded; it never auto-commutes from
    /// electrical observables. The kernel triggers commute via set_state()
    /// when the controlling logic decides to flip the switch.
    [[nodiscard]] bool should_commute(const PwlEventContext& /*ctx*/) const noexcept {
        return false;
    }

private:
    Scalar g_on_;
    Scalar g_off_;
    bool is_closed_;
    SwitchingMode mode_ = SwitchingMode::Auto;
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
    static constexpr bool supports_pwl = true;
    static constexpr std::size_t jacobian_size = 4;
};

}  // namespace pulsim::v1
