#pragma once

// =============================================================================
// Pulsim v2 — Layer 4 V13: SH1 MOSFET Newton refresh
// =============================================================================
//
// Per-Newton-iteration refresh for `models::MosfetLevel1`.
// Iterates `BranchKind::Nonlinear` branches registered with
// `StoredKind::MosfetLevel1`, reads V(drain), V(source),
// V(gate) from the current state vector, evaluates the SH1
// current + Jacobian via AD, and stamps:
//
//   f[drain]  += I_D
//   f[source] -= I_D
//   J[drain,  drain]  += ∂I/∂V(drain)
//   J[drain,  source] += ∂I/∂V(source)
//   J[drain,  gate]   += ∂I/∂V(gate)
//   J[source, drain]  -= ∂I/∂V(drain)
//   J[source, source] -= ∂I/∂V(source)
//   J[source, gate]   -= ∂I/∂V(gate)
//
// (Gate KCL is unaffected — Level 1 ideal-gate has zero gate
// current. Parasitic gate capacitance is OUT OF SCOPE for V0
// of this device.)

#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/models/mosfet_level1.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/nonlinear_solve.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <algorithm>
#include <cmath>

namespace pulsim::v2::pwl {

/// Refresh function that stamps SH1 MOSFET contributions on
/// every Nonlinear branch registered as one. Clears J_nl
/// and f_nl first (consistent with `refresh_smooth_diodes`'s
/// standalone convention). Returns max(|i_drain|) for
/// residual-norm reporting.
inline Real refresh_mosfets_level1(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const topology::Graph& graph,
    const DevicePool& pool) {
    if (J_nl.rows() > 0) J_nl.setZero();
    if (f_nl.size() > 0) f_nl.setZero();
    Real max_abs_i = Real{0};

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        if (branch.kind != topology::BranchKind::Nonlinear) {
            continue;
        }
        if (pool.kind_of(branch.id) !=
                DevicePool::StoredKind::MosfetLevel1) {
            continue;
        }
        const auto& p = pool.mosfet_level1_params(branch.id);
        const Index gate_node =
            pool.mosfet_level1_gate_node(branch.id);

        const Index drain  = branch.from;
        const Index source = branch.to;
        const Index gate   = gate_node;

        const Real v_d = stamping::read_node_voltage(x, drain);
        const Real v_s = stamping::read_node_voltage(x, source);
        const Real v_g = stamping::read_node_voltage(x, gate);

        // Terminal order: drain, source, gate.
        const models::ModelInputs<models::MosfetLevel1> v_term{
            v_d, v_s, v_g};

        const auto [i, partials] =
            models::evaluate_current_and_jacobian<
                models::MosfetLevel1>(v_term, p);

        const bool d_active = stamping::node_is_active(drain);
        const bool s_active = stamping::node_is_active(source);
        const bool g_active = stamping::node_is_active(gate);

        // KCL residuals.
        if (d_active) f_nl[drain]  += i;
        if (s_active) f_nl[source] -= i;

        // Jacobian: drain row.
        if (d_active) {
            if (d_active) {
                J_nl.coeffRef(drain, drain) += partials[0];
            }
            if (s_active) {
                J_nl.coeffRef(drain, source) += partials[1];
            }
            if (g_active) {
                J_nl.coeffRef(drain, gate) += partials[2];
            }
        }
        // Jacobian: source row.
        if (s_active) {
            if (d_active) {
                J_nl.coeffRef(source, drain) -= partials[0];
            }
            if (s_active) {
                J_nl.coeffRef(source, source) -= partials[1];
            }
            if (g_active) {
                J_nl.coeffRef(source, gate) -= partials[2];
            }
        }

        max_abs_i = std::max(max_abs_i, std::abs(i));
    }

    return max_abs_i;
}

/// Combined refresh that runs both the smooth-blend
/// IdealDiode and SH1 MOSFET stampers in a single pass.
/// Useful as a drop-in `NonlinearRefreshFn` when a circuit
/// contains both kinds of nonlinear devices.
[[nodiscard]] inline NonlinearRefreshFn
make_combined_diode_mosfet_refresh() {
    return [](const Vector& x,
                sparse::Matrix& J_nl,
                Vector& f_nl,
                const topology::Graph& graph,
                const DevicePool& pool) -> Real {
        if (J_nl.rows() > 0) J_nl.setZero();
        if (f_nl.size() > 0) f_nl.setZero();
        // Diode pass first (writes diode contributions).
        // We reuse refresh_smooth_diodes' logic but inlined
        // to avoid double zero-out.
        Real max_abs_i = Real{0};

        for (Index b_id = 0;
             b_id < graph.num_branches(); ++b_id) {
            const auto& branch = graph.branch(b_id);
            if (branch.kind !=
                    topology::BranchKind::Nonlinear) {
                continue;
            }
            const auto kind = pool.kind_of(branch.id);
            if (kind ==
                    DevicePool::StoredKind::NonlinearDiode) {
                const auto& p =
                    pool.nonlinear_diode_params(branch.id);
                const stamping::BranchCoord coord{
                    branch.from, branch.to, branch.id};
                const Real v_from =
                    stamping::read_node_voltage(x, coord.from);
                const Real v_to =
                    stamping::read_node_voltage(x, coord.to);
                const models::ModelInputs<models::IdealDiode>
                    v_term{v_from, v_to};
                const auto [i, partials] =
                    models::evaluate_current_and_jacobian<
                        models::IdealDiode>(v_term, p);
                const bool from_active =
                    stamping::node_is_active(coord.from);
                const bool to_active =
                    stamping::node_is_active(coord.to);
                if (from_active) f_nl[coord.from] += i;
                if (to_active)   f_nl[coord.to]   -= i;
                if (from_active) {
                    J_nl.coeffRef(coord.from, coord.from) +=
                        partials[0];
                    if (to_active) {
                        J_nl.coeffRef(coord.from, coord.to) +=
                            partials[1];
                    }
                }
                if (to_active) {
                    if (from_active) {
                        J_nl.coeffRef(coord.to, coord.from) -=
                            partials[0];
                    }
                    J_nl.coeffRef(coord.to, coord.to) -=
                        partials[1];
                }
                max_abs_i =
                    std::max(max_abs_i, std::abs(i));
            } else if (kind ==
                    DevicePool::StoredKind::MosfetLevel1) {
                // Inline the MOSFET pass so we don't
                // double-zero.
                const auto& p =
                    pool.mosfet_level1_params(branch.id);
                const Index drain  = branch.from;
                const Index source = branch.to;
                const Index gate   =
                    pool.mosfet_level1_gate_node(branch.id);
                const Real v_d =
                    stamping::read_node_voltage(x, drain);
                const Real v_s =
                    stamping::read_node_voltage(x, source);
                const Real v_g =
                    stamping::read_node_voltage(x, gate);
                const models::ModelInputs<
                        models::MosfetLevel1>
                    v_term{v_d, v_s, v_g};
                const auto [i, partials] =
                    models::evaluate_current_and_jacobian<
                        models::MosfetLevel1>(v_term, p);
                const bool d_active =
                    stamping::node_is_active(drain);
                const bool s_active =
                    stamping::node_is_active(source);
                const bool g_active =
                    stamping::node_is_active(gate);
                if (d_active) f_nl[drain]  += i;
                if (s_active) f_nl[source] -= i;
                if (d_active) {
                    J_nl.coeffRef(drain, drain) +=
                        partials[0];
                    if (s_active) {
                        J_nl.coeffRef(drain, source) +=
                            partials[1];
                    }
                    if (g_active) {
                        J_nl.coeffRef(drain, gate) +=
                            partials[2];
                    }
                }
                if (s_active) {
                    if (d_active) {
                        J_nl.coeffRef(source, drain) -=
                            partials[0];
                    }
                    J_nl.coeffRef(source, source) -=
                        partials[1];
                    if (g_active) {
                        J_nl.coeffRef(source, gate) -=
                            partials[2];
                    }
                }
                max_abs_i =
                    std::max(max_abs_i, std::abs(i));
            }
        }
        return max_abs_i;
    };
}

}  // namespace pulsim::v2::pwl
