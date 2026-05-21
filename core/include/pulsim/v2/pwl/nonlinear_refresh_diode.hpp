#pragma once

// =============================================================================
// Pulsim v2 — Layer 4 V3: Smooth-blend IdealDiode Newton refresh
// =============================================================================
//
// Built-in `NonlinearRefreshFn` for the smooth-blend
// `models::IdealDiode` (the sigmoid-shifted Norton model that
// already lives in models/ideal_diode.hpp and was waiting for a
// Newton consumer since Layer 2).
//
// For V0 of the nonlinear-Newton OpenSpec we need a way for the
// caller to register the smooth-blend diode's params on a
// `BranchKind::Nonlinear` branch. We piggyback on the existing
// `DevicePool` machinery: a new `add_nonlinear_diode(...)`
// method stores the params. The refresh function looks them up
// per branch.

#include "pulsim/v2/models/device_model.hpp"
#include "pulsim/v2/models/ideal_diode.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/pwl/nonlinear_solve.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <cmath>

namespace pulsim::v2::pwl {

/// Refresh function that stamps the smooth-blend `IdealDiode`
/// on every `BranchKind::Nonlinear` branch registered as one.
///
/// Returns max(|i_diode|) across all diode branches as the
/// "residual norm" indicator.
///
/// Implementation pattern: for each nonlinear branch, read the
/// terminal voltages from `x`, call
/// `evaluate_current_and_jacobian<IdealDiode>` to get
/// `(i, partials)`, stamp:
///   f(from) += i;          f(to) -= i
///   J(from, from) += g00;  J(from, to) += g01
///   J(to,   from) -= g00;  J(to,   to) -= g01
inline Real refresh_smooth_diodes(
    const Vector& x,
    sparse::Matrix& J_nl,
    Vector& f_nl,
    const topology::Graph& graph,
    const DevicePool& pool) {
    // Zero out previous contributions.
    if (J_nl.rows() > 0) {
        J_nl.setZero();
    }
    if (f_nl.size() > 0) {
        f_nl.setZero();
    }

    Real max_abs_i = Real{0};

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        if (branch.kind != topology::BranchKind::Nonlinear) {
            continue;
        }
        // Only handle branches registered as smooth-blend diodes.
        // We use a sentinel: kind_of == Diode means SwitchedDiode
        // (the binary-on/off model). We need a distinct
        // `nonlinear_diode_params` slot in DevicePool — see the
        // add method below.
        const auto kind = pool.kind_of(branch.id);
        if (kind != DevicePool::StoredKind::NonlinearDiode) {
            continue;
        }
        const auto& p = pool.nonlinear_diode_params(branch.id);

        const stamping::BranchCoord coord{branch.from, branch.to,
                                            branch.id};
        const Real v_from =
            stamping::read_node_voltage(x, coord.from);
        const Real v_to =
            stamping::read_node_voltage(x, coord.to);
        const models::ModelInputs<models::IdealDiode> v_term{
            v_from, v_to};

        const auto [i, partials] =
            models::evaluate_current_and_jacobian<models::IdealDiode>(
                v_term, p);

        const bool from_active =
            stamping::node_is_active(coord.from);
        const bool to_active =
            stamping::node_is_active(coord.to);

        // Residual contributions.
        if (from_active) f_nl[coord.from] += i;
        if (to_active)   f_nl[coord.to]   -= i;

        // Jacobian contributions (standard 2x2 conductance pattern).
        if (from_active) {
            J_nl.coeffRef(coord.from, coord.from) += partials[0];
            if (to_active) {
                J_nl.coeffRef(coord.from, coord.to) += partials[1];
            }
        }
        if (to_active) {
            if (from_active) {
                J_nl.coeffRef(coord.to, coord.from) -= partials[0];
            }
            J_nl.coeffRef(coord.to, coord.to) -= partials[1];
        }

        max_abs_i = std::max(max_abs_i, std::abs(i));
    }

    return max_abs_i;
}

}  // namespace pulsim::v2::pwl
