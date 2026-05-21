#pragma once

// =============================================================================
// Pulsim v2 — Layer 3: Voltage-source constraint stamper
// =============================================================================
//
// `pulsim-v2-generic-stamping-pipeline` Phase 3.
//
// Standard MNA voltage-source handling. A voltage source between
// nodes (from, to) imposes the constraint `v[from] - v[to] = V`.
// The augmented MNA system adds one unknown (the branch current
// `i_branch`) and one constraint equation. The branch current
// participates in the KCL of the two terminal nodes.
//
// State-vector layout for an N-node circuit with M voltage sources:
//   [ v_0 v_1 ... v_{N-1}  i_b0 i_b1 ... i_b{M-1} ]
//                          ^-- the branch-current unknowns
//
// Caller passes `branch_var_id` = the absolute index of THIS source's
// branch-current row in the state vector (i.e. `N + source_index`).

#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"

namespace pulsim::v2::stamping {

// -----------------------------------------------------------------------------
// stamp_voltage_source — adds the source's contribution to J and f.
//
// (1) KCL contribution at the terminal nodes:
//       f[from] += i_branch       (current leaves from-terminal)
//       f[to]   -= i_branch
//     Jacobian:
//       J(from, branch_var_id) = +1
//       J(to,   branch_var_id) = -1
//
// (2) Constraint row at branch_var_id:
//       f[branch_var_id] = (v_from - v_to) - V
//     Jacobian:
//       J(branch_var_id, from) = +1
//       J(branch_var_id, to)   = -1
//
// Ground: stamping rows / cols for `kGround` is skipped, but the
// constraint row still reads ground's voltage as 0.
// -----------------------------------------------------------------------------
inline void stamp_voltage_source(sparse::Matrix& J, Vector& f,
                                  const Vector& x,
                                  const BranchCoord& coord,
                                  Index branch_var_id,
                                  Real V) noexcept {
    const bool from_active = node_is_active(coord.from);
    const bool to_active   = node_is_active(coord.to);
    const Real i_branch = x[branch_var_id];

    // (1) KCL contributions at the terminal nodes.
    if (from_active) {
        f[coord.from] += i_branch;
        J.coeffRef(coord.from, branch_var_id) += Real{1};
    }
    if (to_active) {
        f[coord.to] -= i_branch;
        J.coeffRef(coord.to, branch_var_id) -= Real{1};
    }

    // (2) Constraint row at branch_var_id.
    const Real v_from = read_node_voltage(x, coord.from);
    const Real v_to   = read_node_voltage(x, coord.to);
    f[branch_var_id] += (v_from - v_to) - V;
    if (from_active) {
        J.coeffRef(branch_var_id, coord.from) += Real{1};
    }
    if (to_active) {
        J.coeffRef(branch_var_id, coord.to) -= Real{1};
    }
}

}  // namespace pulsim::v2::stamping
