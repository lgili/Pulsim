#pragma once

// =============================================================================
// AD Validation Layer (Phase 4 of `add-automatic-differentiation`)
// =============================================================================
//
// Walks every nonlinear device in a `Circuit`, computes the build-selected
// stamp's Jacobian row (manual under default, AD under `PULSIM_USE_AD_STAMP`)
// against centered finite differences on the device's residual helper, and
// reports any per-entry disagreement above tolerance. The `f` Norton offset
// is intentionally NOT validated here (different forms across devices); the
// J row is the universal contract.
//
// Use this in CI to catch silent stamp regressions when:
//   * a manual stamp is edited (e.g. a sign flipped by accident),
//   * a templated residual helper drifts from the manual code path,
//   * a new device is added without an AD path.
//
// Typical CI usage: at the start of a benchmark / regression test, build the
// circuit, choose 2–3 representative operating points (one in each region for
// region-bearing devices), call `validate_nonlinear_jacobians`, assert the
// returned vector is empty.

#include "pulsim/v1/runtime_circuit.hpp"

#include <Eigen/Sparse>
#include <array>
#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace pulsim::v1::ad {

struct JacobianMismatch {
    std::string device_name;
    std::string device_type;
    Index op_point_index = 0;
    /// Position of the offending entry in the device-local Jacobian (small,
    /// 2×2 for 2-terminal devices, 3×3 for 3-terminal). Row 0 is always the
    /// "current-leaving" terminal (anode / drain / collector / t1).
    Index local_row = 0;
    Index local_col = 0;
    Real stamp_value = 0.0;
    Real fd_value = 0.0;
    Real abs_delta = 0.0;
};

namespace detail {

template <std::size_t NumTerminals, typename Device, typename ResidualFn>
void validate_one_device(Device& device,
                         const std::string& device_name,
                         const std::string& device_type,
                         const std::array<Index, NumTerminals>& terminal_nodes,
                         const Vector& x,
                         Index op_idx,
                         std::size_t canonical_row,
                         ResidualFn residual_at,
                         Real abs_tol,
                         std::vector<JacobianMismatch>& out) {
    std::array<Real, NumTerminals> terminal_v{};
    for (std::size_t i = 0; i < NumTerminals; ++i) {
        const Index node = terminal_nodes[i];
        terminal_v[i] = (node >= 0 && node < x.size()) ? x[node] : Real{0.0};
    }

    // Stamp through the device's regular path (build-flag selects manual vs AD).
    Eigen::SparseMatrix<Real> J(NumTerminals, NumTerminals);
    Eigen::VectorXd f = Eigen::VectorXd::Zero(NumTerminals);
    Vector x_local(NumTerminals);
    for (std::size_t i = 0; i < NumTerminals; ++i) {
        x_local[static_cast<Index>(i)] = terminal_v[i];
    }
    std::array<Index, NumTerminals> local_nodes{};
    for (std::size_t i = 0; i < NumTerminals; ++i) {
        local_nodes[i] = static_cast<Index>(i);
    }
    device.stamp_jacobian(J, f, x_local, local_nodes);
    J.makeCompressed();

    // Centered FD on the residual helper. The "current leaving" terminal is
    // index 0 by our convention (anode / drain / collector / t1). Build the
    // FD partial vector ∂i/∂terminal_j by perturbing terminal_v[j].
    constexpr Real h = Real{1e-6};
    std::array<Real, NumTerminals> fd_partials{};
    for (std::size_t j = 0; j < NumTerminals; ++j) {
        auto plus = terminal_v;
        auto minus = terminal_v;
        plus[j] += h;
        minus[j] -= h;
        const Real i_plus = residual_at(plus);
        const Real i_minus = residual_at(minus);
        fd_partials[j] = (i_plus - i_minus) / (Real{2.0} * h);
    }

    // J(canonical_row, j) is the partial of the leaving-current at the
    // device's canonical "current-out" terminal. Compare each entry against
    // the FD partial of the residual helper (which returns that same
    // current).
    const Index row = static_cast<Index>(canonical_row);
    for (std::size_t j = 0; j < NumTerminals; ++j) {
        const Real stamp_val = J.coeff(row, static_cast<Index>(j));
        const Real fd_val = fd_partials[j];
        const Real delta = std::abs(stamp_val - fd_val);
        if (delta > abs_tol) {
            JacobianMismatch m;
            m.device_name = device_name;
            m.device_type = device_type;
            m.op_point_index = op_idx;
            m.local_row = row;
            m.local_col = static_cast<Index>(j);
            m.stamp_value = stamp_val;
            m.fd_value = fd_val;
            m.abs_delta = delta;
            out.push_back(std::move(m));
        }
    }
}

}  // namespace detail

/// Validate every nonlinear device in `circuit` at each supplied operating
/// point. Returns a list of per-entry Jacobian mismatches above `abs_tol`.
/// An empty result means the build-selected stamp agrees with FD on every
/// device at every point, within tolerance.
[[nodiscard]] inline std::vector<JacobianMismatch>
validate_nonlinear_jacobians(Circuit& circuit,
                              std::span<const Vector> operating_points,
                              Real abs_tol = Real{1e-6}) {
    std::vector<JacobianMismatch> mismatches;

    for (Index op_idx = 0;
         op_idx < static_cast<Index>(operating_points.size()); ++op_idx) {
        const Vector& x = operating_points[op_idx];

        for (std::size_t dev_idx = 0; dev_idx < circuit.devices().size(); ++dev_idx) {
            const auto& conn = circuit.connections()[dev_idx];
            auto& device_variant = circuit.devices_mutable()[dev_idx];

            std::visit([&](auto& device) {
                using T = std::decay_t<decltype(device)>;

                if constexpr (std::is_same_v<T, IdealDiode>) {
                    // Canonical row = anode (terminal 0).
                    if (conn.nodes.size() < 2) return;
                    std::array<Index, 2> term{conn.nodes[0], conn.nodes[1]};
                    detail::validate_one_device<2>(
                        device, conn.name, "IdealDiode", term, x, op_idx,
                        /*canonical_row=*/0,
                        [&](const std::array<Real, 2>& v) {
                            return device.template forward_current_behavioral<Real>(v[0], v[1]);
                        },
                        abs_tol, mismatches);
                } else if constexpr (std::is_same_v<T, MOSFET>) {
                    // Canonical row = drain (terminal 1).
                    if (conn.nodes.size() < 3) return;
                    std::array<Index, 3> term{conn.nodes[0], conn.nodes[1], conn.nodes[2]};
                    detail::validate_one_device<3>(
                        device, conn.name, "MOSFET", term, x, op_idx,
                        /*canonical_row=*/1,
                        [&](const std::array<Real, 3>& v) {
                            return device.template drain_current_behavioral<Real>(v[0], v[1], v[2]);
                        },
                        abs_tol, mismatches);
                } else if constexpr (std::is_same_v<T, IGBT>) {
                    // Canonical row = collector (terminal 1).
                    if (conn.nodes.size() < 3) return;
                    std::array<Index, 3> term{conn.nodes[0], conn.nodes[1], conn.nodes[2]};
                    detail::validate_one_device<3>(
                        device, conn.name, "IGBT", term, x, op_idx,
                        /*canonical_row=*/1,
                        [&](const std::array<Real, 3>& v) {
                            return device.template collector_current_behavioral<Real>(v[0], v[1], v[2]);
                        },
                        abs_tol, mismatches);
                } else if constexpr (std::is_same_v<T, VoltageControlledSwitch>) {
                    // Canonical row = t1 (terminal 1; ctrl is observe-only).
                    if (conn.nodes.size() < 3) return;
                    std::array<Index, 3> term{conn.nodes[0], conn.nodes[1], conn.nodes[2]};
                    detail::validate_one_device<3>(
                        device, conn.name, "VoltageControlledSwitch", term, x, op_idx,
                        /*canonical_row=*/1,
                        [&](const std::array<Real, 3>& v) {
                            return device.template switch_current_behavioral<Real>(v[0], v[1], v[2]);
                        },
                        abs_tol, mismatches);
                }
                // Linear devices opt out of FD validation (Phase 3).
            }, device_variant);
        }
    }

    return mismatches;
}

}  // namespace pulsim::v1::ad
