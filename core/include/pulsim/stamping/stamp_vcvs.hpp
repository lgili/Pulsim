#pragma once

// =============================================================================
// Pulsim — Layer 3: VCVS constraint stamper (V15)
// =============================================================================
//
// MNA stamping for the voltage-controlled voltage source:
//
//   V_out_pos − V_out_neg = Gain · (V_in_pos − V_in_neg)
//
// One branch-current unknown `I_vcvs` is added (like a regular
// VoltageSource). The constraint row reads:
//
//   (V_out_pos − V_out_neg) − Gain·(V_in_pos − V_in_neg) = 0
//
// KCL contributions are the same as a VoltageSource (current
// in at out_pos, out at out_neg).

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"

namespace pulsim::stamping {

/// Stamp the VCVS contributions into J and f. `coord.from`
/// and `coord.to` are the OUTPUT branch endpoints
/// (out_pos, out_neg). `in_pos_node` / `in_neg_node` are
/// the SENSE input node indices. `gain` is the dimensionless
/// gain.
inline void stamp_vcvs(sparse::Matrix& J, Vector& f,
                        const Vector& x,
                        const BranchCoord& coord,
                        Index in_pos_node,
                        Index in_neg_node,
                        Index branch_var_id,
                        Real gain) noexcept {
    const bool out_pos_active = node_is_active(coord.from);
    const bool out_neg_active = node_is_active(coord.to);
    const bool in_pos_active  = node_is_active(in_pos_node);
    const bool in_neg_active  = node_is_active(in_neg_node);
    const Real i_branch = x[branch_var_id];

    // (1) KCL contributions at the OUTPUT terminal nodes
    //     (same pattern as a regular VoltageSource).
    if (out_pos_active) {
        f[coord.from] += i_branch;
        J.coeffRef(coord.from, branch_var_id) += Real{1};
    }
    if (out_neg_active) {
        f[coord.to] -= i_branch;
        J.coeffRef(coord.to, branch_var_id) -= Real{1};
    }

    // (2) Constraint row at branch_var_id:
    //   (V_out_pos − V_out_neg) − gain·(V_in_pos − V_in_neg) = 0
    const Real v_out_pos = read_node_voltage(x, coord.from);
    const Real v_out_neg = read_node_voltage(x, coord.to);
    const Real v_in_pos  = read_node_voltage(x, in_pos_node);
    const Real v_in_neg  = read_node_voltage(x, in_neg_node);
    f[branch_var_id] +=
        (v_out_pos - v_out_neg) -
        gain * (v_in_pos - v_in_neg);
    if (out_pos_active) {
        J.coeffRef(branch_var_id, coord.from) += Real{1};
    }
    if (out_neg_active) {
        J.coeffRef(branch_var_id, coord.to) -= Real{1};
    }
    if (in_pos_active) {
        J.coeffRef(branch_var_id, in_pos_node) -= gain;
    }
    if (in_neg_active) {
        J.coeffRef(branch_var_id, in_neg_node) += gain;
    }
}

}  // namespace pulsim::stamping
