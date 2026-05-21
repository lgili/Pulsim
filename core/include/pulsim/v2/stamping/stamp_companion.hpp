#pragma once

// =============================================================================
// Pulsim v2 — Layer 3: Trapezoidal-companion stampers (Capacitor + Inductor)
// =============================================================================
//
// `pulsim-v2-trapezoidal-companion` Phase 3.
//
// Two helpers, one per dynamic device. Both stamp the LINEAR
// part of the trap companion (the g_eq conductance / constraint
// row) into J. The HISTORY part is added by Layer 5's
// HistoryState at solve time via b_extra — assembly emits no
// constants for dynamic devices.
//
//   stamp_capacitor_companion  → 4-entry conductance block at
//                                (from, to), like a resistor.
//   stamp_inductor_companion   → KCL contributions of i_L on the
//                                node rows + constraint row with
//                                +1/−1 on voltage columns and
//                                −(2L/dt) on branch_var_id.
//
// History contributions (added at solve time):
//   Capacitor:  b_extra(from)         += −I_hist
//               b_extra(to)           += +I_hist
//   Inductor:   b_extra(branch_var)   += −(2L/dt) · I_hist,L

#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"

namespace pulsim::v2::stamping {

// -----------------------------------------------------------------------------
// stamp_capacitor_companion — capacitor's linear (g_eq) stamp.
//
// Stamps the 4-entry conductance block (identical to a resistor
// with G = g_eq). Ground endpoints are skipped on rows/cols where
// `node_is_active` returns false.
//
// History contribution is NOT stamped here — Layer 5 adds it via
// b_extra at solve time. (`b` is touched only insofar as the
// caller may have pre-allocated it; this function does not write
// to b.)
// -----------------------------------------------------------------------------
inline void stamp_capacitor_companion(sparse::Matrix& J,
                                       Vector& /*b*/,
                                       const BranchCoord& coord,
                                       Real g_eq) noexcept {
    const bool from_active = node_is_active(coord.from);
    const bool to_active   = node_is_active(coord.to);

    // Conductance block (identical pattern to a resistor):
    //   J(from, from) += +g_eq
    //   J(from, to)   += −g_eq
    //   J(to,   from) += −g_eq
    //   J(to,   to)   += +g_eq
    if (from_active) {
        J.coeffRef(coord.from, coord.from) += g_eq;
        if (to_active) {
            J.coeffRef(coord.from, coord.to) -= g_eq;
        }
    }
    if (to_active) {
        if (from_active) {
            J.coeffRef(coord.to, coord.from) -= g_eq;
        }
        J.coeffRef(coord.to, coord.to) += g_eq;
    }
}

// -----------------------------------------------------------------------------
// stamp_inductor_companion — inductor's linear (g_eq_inv) stamp.
//
// Layout of `state vector x` with M voltage sources + K inductors:
//   x = [ v_0 … v_{N-1}  i_src_0 … i_src_{M-1}  i_L_0 … i_L_{K-1} ]
//
// `branch_var_id` is the absolute index of this inductor's branch
// current unknown. The caller (DevicePool) computes it.
//
// Stamping pattern (identical to a voltage source plus the
// constraint-row diagonal entry):
//   (1) KCL contributions of i_L on the terminal rows:
//         J(from, branch_var_id) += +1
//         J(to,   branch_var_id) += −1
//   (2) Constraint row at branch_var_id:
//         J(branch_var_id, from)            += +1
//         J(branch_var_id, to)              += −1
//         J(branch_var_id, branch_var_id)   += −(2L/dt) = −1 / g_eq_inv
//       The history contribution `+(2L/dt) · I_hist,L` lives in
//       b_extra at solve time (NOT stamped here).
// -----------------------------------------------------------------------------
inline void stamp_inductor_companion(sparse::Matrix& J,
                                      Vector& /*b*/,
                                      const BranchCoord& coord,
                                      Index branch_var_id,
                                      Real g_eq_inv) noexcept {
    const bool from_active = node_is_active(coord.from);
    const bool to_active   = node_is_active(coord.to);

    // (1) KCL of i_L on terminal rows.
    if (from_active) {
        J.coeffRef(coord.from, branch_var_id) += Real{1};
    }
    if (to_active) {
        J.coeffRef(coord.to, branch_var_id) -= Real{1};
    }

    // (2) Constraint row at branch_var_id.
    if (from_active) {
        J.coeffRef(branch_var_id, coord.from) += Real{1};
    }
    if (to_active) {
        J.coeffRef(branch_var_id, coord.to) -= Real{1};
    }
    // Diagonal: −(2L/dt) = −(1 / g_eq_inv). Guard against
    // g_eq_inv == 0 (which means dt = 0 — caller should not reach
    // this path; DispatchsOnly happens when dt != 0).
    const Real two_L_over_dt = Real{1} / g_eq_inv;
    J.coeffRef(branch_var_id, branch_var_id) -= two_L_over_dt;
}

}  // namespace pulsim::v2::stamping
