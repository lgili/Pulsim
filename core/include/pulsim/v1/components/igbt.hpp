#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// IGBT Device (CRTP - Nonlinear, 3-terminal)
// =============================================================================

/// Simplified IGBT model for power electronics
/// Terminals: Gate (0), Collector (1), Emitter (2)
class IGBT : public NonlinearDeviceBase<IGBT> {
public:
    using Base = NonlinearDeviceBase<IGBT>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::IGBT);

    struct Params {
        Scalar vth = 5.0;           // Gate threshold voltage (V)
        Scalar g_on = 1e4;          // On-state conductance (S)
        Scalar g_off = 1e-12;       // Off-state conductance (S)
        Scalar v_ce_sat = 1.5;      // Collector-emitter saturation voltage (V)
    };

    explicit IGBT(std::string name = "")
        : Base(std::move(name)), params_(), is_on_(false) {}

    explicit IGBT(Params params, std::string name)
        : Base(std::move(name)), params_(params), is_on_(false) {}

    explicit IGBT(Scalar vth, Scalar g_on = 1e4, std::string name = "")
        : Base(std::move(name))
        , params_{vth, g_on, 1e-12, 1.5}
        , is_on_(false) {}

    /// Stamp Jacobian for Newton iteration
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;

        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        Scalar vg = (n_gate >= 0) ? x[n_gate] : 0.0;
        Scalar vc = (n_collector >= 0) ? x[n_collector] : 0.0;
        Scalar ve = (n_emitter >= 0) ? x[n_emitter] : 0.0;

        Scalar vge = vg - ve;
        Scalar vce = vc - ve;

        // Determine state
        is_on_ = (vge > params_.vth) && (vce > 0);
        Scalar g = is_on_ ? params_.g_on : params_.g_off;

        // Model as voltage-controlled conductance with saturation
        Scalar ic = g * vce;
        if (is_on_ && vce > params_.v_ce_sat) {
            // Add forward voltage drop
            ic = g * (vce - params_.v_ce_sat) + params_.g_on * params_.v_ce_sat;
        }

        // Stamp collector-emitter conductance
        if (n_collector >= 0) {
            J.coeffRef(n_collector, n_collector) += g;
            if (n_emitter >= 0) J.coeffRef(n_collector, n_emitter) -= g;
        }
        if (n_emitter >= 0) {
            J.coeffRef(n_emitter, n_emitter) += g;
            if (n_collector >= 0) J.coeffRef(n_emitter, n_collector) -= g;
        }

        // Residual
        if (n_collector >= 0) f[n_collector] += ic - g * vce;
        if (n_emitter >= 0) f[n_emitter] -= ic - g * vce;
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;
        const NodeIndex n_collector = nodes[1];
        const NodeIndex n_emitter = nodes[2];

        Scalar g = is_on_ ? params_.g_on : params_.g_off;

        if (n_collector >= 0) {
            G.coeffRef(n_collector, n_collector) += g;
            if (n_emitter >= 0) G.coeffRef(n_collector, n_emitter) -= g;
        }
        if (n_emitter >= 0) {
            G.coeffRef(n_emitter, n_emitter) += g;
            if (n_collector >= 0) G.coeffRef(n_emitter, n_collector) -= g;
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        return StaticSparsityPattern<9>{{
            JacobianEntry{0, 0}, JacobianEntry{0, 1}, JacobianEntry{0, 2},
            JacobianEntry{1, 0}, JacobianEntry{1, 1}, JacobianEntry{1, 2},
            JacobianEntry{2, 0}, JacobianEntry{2, 1}, JacobianEntry{2, 2}
        }};
    }

    [[nodiscard]] bool is_conducting() const { return is_on_; }
    [[nodiscard]] const Params& params() const { return params_; }

private:
    Params params_;
    mutable bool is_on_;
};

template<>
struct device_traits<IGBT> {
    static constexpr DeviceType type = DeviceType::IGBT;
    static constexpr std::size_t num_pins = 3;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = true;
    static constexpr std::size_t jacobian_size = 9;
};

}  // namespace pulsim::v1
