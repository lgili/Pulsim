#pragma once

// =============================================================================
// Pulsim — Layer 3: ideal-transformer stamper (Phase 4 C.4)
// =============================================================================
//
//   (v_s_from − v_s_to) − n·(v_p_from − v_p_to) = 0     constraint row
//   KCL secondary: i_s in at s_from, out at s_to          (as a VCVS)
//   KCL primary:   i_p = −n·i_s in at p_from, out at p_to (the CCCS)
//
// Sign check by power: currents INTO the dotted terminals through the
// device, v_p i_p + v_s i_s = 0 with v_s = n v_p gives i_p = −n i_s.
// A resistive load on the secondary pulls i_s < 0 (current leaves
// s_from into the load), so i_p = −n i_s > 0 flows into the primary
// from the source, as it should.
//
// The first two blocks are `stamp_vcvs` unchanged; the third is what
// makes it a transformer rather than an amplifier.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"

namespace pulsim::stamping {

/// `coord` is the SECONDARY branch (s_from, s_to); `p_from_node` /
/// `p_to_node` are the primary terminals; `branch_var_id` is the
/// secondary current unknown; `n` = N_s / N_p.
inline void stamp_ideal_transformer(sparse::Matrix& J, Vector& f,
                                    const Vector& x,
                                    const BranchCoord& coord,
                                    Index p_from_node,
                                    Index p_to_node,
                                    Index branch_var_id,
                                    Real n) noexcept {
    const bool s_from_active = node_is_active(coord.from);
    const bool s_to_active   = node_is_active(coord.to);
    const bool p_from_active = node_is_active(p_from_node);
    const bool p_to_active   = node_is_active(p_to_node);
    const Real i_s = x[branch_var_id];

    // (1) KCL at the secondary terminals.
    if (s_from_active) {
        f[coord.from] += i_s;
        J.coeffRef(coord.from, branch_var_id) += Real{1};
    }
    if (s_to_active) {
        f[coord.to] -= i_s;
        J.coeffRef(coord.to, branch_var_id) -= Real{1};
    }

    // (2) Constraint row: v_s − n·v_p = 0.
    const Real v_s = read_node_voltage(x, coord.from)
                     - read_node_voltage(x, coord.to);
    const Real v_p = read_node_voltage(x, p_from_node)
                     - read_node_voltage(x, p_to_node);
    f[branch_var_id] += v_s - n * v_p;
    if (s_from_active) J.coeffRef(branch_var_id, coord.from) += Real{1};
    if (s_to_active)   J.coeffRef(branch_var_id, coord.to)   -= Real{1};
    if (p_from_active) J.coeffRef(branch_var_id, p_from_node) -= n;
    if (p_to_active)   J.coeffRef(branch_var_id, p_to_node)   += n;

    // (3) KCL at the primary terminals: the reflected current
    //     i_p = −n·i_s, in at p_from, out at p_to.
    const Real i_p = -n * i_s;
    if (p_from_active) {
        f[p_from_node] += i_p;
        J.coeffRef(p_from_node, branch_var_id) += -n;
    }
    if (p_to_active) {
        f[p_to_node] -= i_p;
        J.coeffRef(p_to_node, branch_var_id) += n;
    }
}

}  // namespace pulsim::stamping
