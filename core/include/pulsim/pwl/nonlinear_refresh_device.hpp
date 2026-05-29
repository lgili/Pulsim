#pragma once

// =============================================================================
// Pulsim — Layer 4: generic Newton-refresh stamper for AD-driven devices
// =============================================================================
//
// Collapses the per-device copy-paste (smooth diode / SH1 MOSFET / Level-1
// IGBT) into ONE terminal-count-generic loop. Audit 2026-05, finding #14b:
// the three `nonlinear_refresh_*` files were ~80-95% identical
// (iterate Nonlinear branches → filter by StoredKind → read terminals →
// evaluate_current_and_jacobian → stamp a (rows x N) conductance block).
//
// Each AD device model satisfies `models::DeviceModel` and exposes
// `num_terminals` + `current<S>()`. Convention shared by every device the
// Newton path stamps:
//   * terminal[0] and terminal[1] carry the branch current
//     (KCL: +i at terminal[0], -i at terminal[1]);
//   * ALL N terminals are control terminals contributing Jacobian columns
//     ∂i/∂v[k].
// Diodes are 2-terminal {anode, cathode}; MOSFET/IGBT are 3-terminal
// {drain/collector, source/emitter, gate}.
//
// Callers supply two small functors:
//   get_params(pool, branch_id)  -> Device::Params
//       (returned BY VALUE so override variants — e.g. continuation's
//        kappa/V_F0 homotopy — can transform a copy of the pool params);
//   get_terminals(pool, branch)  -> std::array<Index, Device::num_terminals>
//       (terminal node indices in [from, to, (gate...)] order).
//
// This helper does NOT clear J_nl / f_nl. The caller zeroes once, then may
// run several device passes (stamps are additive) before returning — that is
// how the combined diode+MOSFET+IGBT refresh avoids double-zeroing.

#include "pulsim/models/device_model.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace pulsim::pwl {

/// Stamp every `BranchKind::Nonlinear` branch registered as `target_kind`
/// with `Device`'s AD-evaluated current + Jacobian. Returns max(|i|) across
/// the stamped branches (the residual-norm indicator the Newton driver uses).
/// Does NOT zero J_nl/f_nl — see the file header.
template <models::DeviceModel Device, typename ParamFn, typename TerminalFn>
inline Real stamp_nonlinear_branches(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const topology::Graph& graph,
    const DevicePool& pool,
    DevicePool::StoredKind target_kind,
    ParamFn&& get_params,
    TerminalFn&& get_terminals) {
    constexpr Size N = Device::num_terminals;
    static_assert(N >= 2,
        "stamp_nonlinear_branches: device must have >= 2 terminals; "
        "terminal[0]/[1] carry the branch current.");

    Real max_abs_i = Real{0};

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        if (branch.kind != topology::BranchKind::Nonlinear) {
            continue;
        }
        if (pool.kind_of(branch.id) != target_kind) {
            continue;
        }

        const std::array<Index, N> term = get_terminals(pool, branch);
        const typename Device::Params p = get_params(pool, branch.id);

        models::ModelInputs<Device> v_term{};
        for (Size k = 0; k < N; ++k) {
            v_term[k] = stamping::read_node_voltage(x, term[k]);
        }
        const auto [i, partials] =
            models::evaluate_current_and_jacobian<Device>(v_term, p);

        const Index from = term[0];
        const Index to   = term[1];
        const bool from_active = stamping::node_is_active(from);
        const bool to_active   = stamping::node_is_active(to);

        // KCL: current flows terminal[0] -> terminal[1].
        if (from_active) f_nl[from] += i;
        if (to_active)   f_nl[to]   -= i;

        // Jacobian: the from-row gets +∂i/∂v[k] and the to-row gets -∂i/∂v[k]
        // for every control terminal k. An entry is stamped only when BOTH its
        // row node and its column node are active (ground rows/cols dropped) —
        // exactly the active-node guarding the hand-written stampers used.
        for (Size k = 0; k < N; ++k) {
            if (!stamping::node_is_active(term[k])) {
                continue;
            }
            if (from_active) J_nl.coeffRef(from, term[k]) += partials[k];
            if (to_active)   J_nl.coeffRef(to,   term[k]) -= partials[k];
        }

        max_abs_i = std::max(max_abs_i, std::abs(i));
    }

    return max_abs_i;
}

}  // namespace pulsim::pwl
