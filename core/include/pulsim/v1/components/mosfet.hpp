#pragma once

#include "pulsim/v1/components/base.hpp"

namespace pulsim::v1 {

// =============================================================================
// MOSFET Device (CRTP - Nonlinear, 3-terminal)
// =============================================================================

/// MOSFET Level 1 model (Shichman-Hodges)
/// Terminals: Gate (0), Drain (1), Source (2)
class MOSFET : public NonlinearDeviceBase<MOSFET> {
public:
    using Base = NonlinearDeviceBase<MOSFET>;
    static constexpr std::size_t num_pins = 3;
    static constexpr int device_type = static_cast<int>(DeviceType::MOSFET);

    struct Params {
        Scalar vth = 2.0;           // Threshold voltage (V)
        Scalar kp = 0.1;            // Transconductance parameter (A/V^2)
        Scalar lambda = 0.01;       // Channel-length modulation (1/V)
        Scalar g_off = 1e-12;       // Off-state conductance
        bool is_nmos = true;      // NMOS if true, PMOS if false
    };

    explicit MOSFET(std::string name = "")
        : Base(std::move(name)), params_() {}

    explicit MOSFET(Params params, std::string name)
        : Base(std::move(name)), params_(params) {}

    explicit MOSFET(Scalar vth, Scalar kp, bool is_nmos = true, std::string name = "")
        : Base(std::move(name))
        , params_{vth, kp, 0.01, 1e-12, is_nmos} {}

    /// Stamp Jacobian for Newton iteration
    /// Implements Level 1 MOSFET equations
    template<typename Matrix, typename Vec>
    void stamp_jacobian_impl(Matrix& J, Vec& f, const Vec& x, std::span<const NodeIndex> nodes) {
        if (nodes.size() < 3) return;

        const NodeIndex n_gate = nodes[0];
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        // Get terminal voltages
        Scalar vg = (n_gate >= 0) ? x[n_gate] : 0.0;
        Scalar vd = (n_drain >= 0) ? x[n_drain] : 0.0;
        Scalar vs = (n_source >= 0) ? x[n_source] : 0.0;

        // For PMOS, negate voltages
        Scalar sign = params_.is_nmos ? 1.0 : -1.0;
        Scalar vgs = sign * (vg - vs);
        Scalar vds = sign * (vd - vs);

        Scalar id = 0.0;      // Drain current
        Scalar gm = 0.0;      // Transconductance dId/dVgs
        Scalar gds = 0.0;     // Output conductance dId/dVds

        Scalar vth = params_.vth;
        Scalar kp = params_.kp;
        Scalar lambda = params_.lambda;

        if (vgs <= vth) {
            // Cutoff region
            id = params_.g_off * vds;
            gds = params_.g_off;
        } else if (vds < vgs - vth) {
            // Linear (triode) region
            Scalar vov = vgs - vth;
            id = kp * (vov * vds - 0.5 * vds * vds) * (1.0 + lambda * vds);
            gm = kp * vds * (1.0 + lambda * vds);
            gds = kp * (vov - vds) * (1.0 + lambda * vds) + kp * (vov * vds - 0.5 * vds * vds) * lambda;
        } else {
            // Saturation region
            Scalar vov = vgs - vth;
            id = 0.5 * kp * vov * vov * (1.0 + lambda * vds);
            gm = kp * vov * (1.0 + lambda * vds);
            gds = 0.5 * kp * vov * vov * lambda;
        }

        // Apply sign for PMOS
        id *= sign;

        // Stamp Jacobian (Norton equivalent)
        // I_eq = id - gm * vgs - gds * vds
        Scalar i_eq = id - gm * vgs - gds * vds;

        // Conductance stamps: drain-source path
        if (n_drain >= 0) {
            J.coeffRef(n_drain, n_drain) += gds;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) -= gds;
            if (n_gate >= 0) J.coeffRef(n_drain, n_gate) += gm;
            if (n_source >= 0) J.coeffRef(n_drain, n_source) -= gm;  // gm contribution
        }
        if (n_source >= 0) {
            J.coeffRef(n_source, n_source) += gds;
            if (n_drain >= 0) J.coeffRef(n_source, n_drain) -= gds;
            if (n_gate >= 0) J.coeffRef(n_source, n_gate) -= gm;
            J.coeffRef(n_source, n_source) += gm;  // gm contribution
        }

        // Current source stamps
        if (n_drain >= 0) f[n_drain] -= i_eq;
        if (n_source >= 0) f[n_source] += i_eq;
    }

    template<typename Matrix, typename Vec>
    void stamp_impl(Matrix& G, Vec& /*b*/, std::span<const NodeIndex> nodes) {
        // For initial guess, stamp small conductance
        if (nodes.size() < 3) return;
        const NodeIndex n_drain = nodes[1];
        const NodeIndex n_source = nodes[2];

        if (n_drain >= 0) {
            G.coeffRef(n_drain, n_drain) += params_.g_off;
            if (n_source >= 0) G.coeffRef(n_drain, n_source) -= params_.g_off;
        }
        if (n_source >= 0) {
            G.coeffRef(n_source, n_source) += params_.g_off;
            if (n_drain >= 0) G.coeffRef(n_source, n_drain) -= params_.g_off;
        }
    }

    static constexpr auto jacobian_pattern_impl() {
        // 3x3 = 9 entries max, but we mainly use D-S path
        return StaticSparsityPattern<9>{{
            JacobianEntry{0, 0}, JacobianEntry{0, 1}, JacobianEntry{0, 2},
            JacobianEntry{1, 0}, JacobianEntry{1, 1}, JacobianEntry{1, 2},
            JacobianEntry{2, 0}, JacobianEntry{2, 1}, JacobianEntry{2, 2}
        }};
    }

    [[nodiscard]] const Params& params() const { return params_; }

private:
    Params params_;
};

template<>
struct device_traits<MOSFET> {
    static constexpr DeviceType type = DeviceType::MOSFET;
    static constexpr std::size_t num_pins = 3;
    static constexpr std::size_t num_internal_nodes = 0;
    static constexpr bool is_linear = false;  // Nonlinear
    static constexpr bool is_dynamic = false;
    static constexpr bool has_loss_model = true;
    static constexpr bool has_thermal_model = true;
    static constexpr std::size_t jacobian_size = 9;
};

}  // namespace pulsim::v1
