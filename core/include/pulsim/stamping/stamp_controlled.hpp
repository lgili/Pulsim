#pragma once

// =============================================================================
// Pulsim — Layer 3: linear controlled-source stampers
// =============================================================================
//
// Siblings of `stamp_vcvs.hpp` / `stamp_ideal_transformer.hpp`. The
// kernel solves J·x = −f with node rows as KCL ("current leaving the
// node into the device") and one row per branch-current unknown.
//
//   VCCS     i = g_m·(v_cp − v_cn) leaves at `from`:
//              J[from,cp] −= g_m  J[from,cn] += g_m
//              J[to,cp]   += g_m  J[to,cn]   −= g_m
//   CCCS     i = β·x[col_c] leaves at `from`:
//              J[from,col_c] −= β  J[to,col_c] += β
//   CCVS     v_op − v_on = r_m·x[col_c], own unknown i_o (KCL as a
//            voltage source):
//              J[op,row] += 1  J[on,row] −= 1
//              J[row,op] += 1  J[row,on] −= 1  J[row,col_c] −= r_m
//   Gyrator  i_1 = g·v_2 INTO p1_pos, i_2 = −g·v_1 INTO p2_pos: two
//            VCCS of opposite sign, 8 entries. Lossless:
//            v_1·i_1 + v_2·i_2 = g·v_1·v_2 − g·v_2·v_1 = 0.
//
// All four are constant and mask-independent. Like their siblings they
// also add the residual f(x), which is zero at the x = 0 the assemblers
// pass.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"

namespace pulsim::stamping {

namespace detail {

/// f[row] += k·(v_a − v_b), J[row,a] += k, J[row,b] −= k (rows and
/// columns at ground skipped). The building block of every stamp here.
inline void stamp_transconductance(sparse::Matrix& J, Vector& f,
                                   const Vector& x, Index row, Index a,
                                   Index b, Real k) noexcept {
    if (!node_is_active(row)) return;
    f[row] += k * (read_node_voltage(x, a) - read_node_voltage(x, b));
    if (node_is_active(a)) J.coeffRef(row, a) += k;
    if (node_is_active(b)) J.coeffRef(row, b) -= k;
}

}  // namespace detail

/// `coord` = output (from, to).
inline void stamp_vccs(sparse::Matrix& J, Vector& f, const Vector& x,
                       const BranchCoord& coord, Index cp, Index cn,
                       Real gm) noexcept {
    detail::stamp_transconductance(J, f, x, coord.from, cp, cn, -gm);
    detail::stamp_transconductance(J, f, x, coord.to, cp, cn, gm);
}

/// `coord` = output (from, to); `col_c` = controlling current unknown.
inline void stamp_cccs(sparse::Matrix& J, Vector& f, const Vector& x,
                       const BranchCoord& coord, Index col_c,
                       Real beta) noexcept {
    const Real i_c = x[col_c];
    if (node_is_active(coord.from)) {
        f[coord.from] -= beta * i_c;
        J.coeffRef(coord.from, col_c) -= beta;
    }
    if (node_is_active(coord.to)) {
        f[coord.to] += beta * i_c;
        J.coeffRef(coord.to, col_c) += beta;
    }
}

/// `coord` = output (out_pos, out_neg); `branch_var_id` = own unknown.
inline void stamp_ccvs(sparse::Matrix& J, Vector& f, const Vector& x,
                       const BranchCoord& coord, Index branch_var_id,
                       Index col_c, Real rm) noexcept {
    const Real i_o = x[branch_var_id];
    if (node_is_active(coord.from)) {
        f[coord.from] += i_o;
        J.coeffRef(coord.from, branch_var_id) += Real{1};
    }
    if (node_is_active(coord.to)) {
        f[coord.to] -= i_o;
        J.coeffRef(coord.to, branch_var_id) -= Real{1};
    }
    f[branch_var_id] += (read_node_voltage(x, coord.from) -
                         read_node_voltage(x, coord.to)) - rm * x[col_c];
    if (node_is_active(coord.from)) {
        J.coeffRef(branch_var_id, coord.from) += Real{1};
    }
    if (node_is_active(coord.to)) {
        J.coeffRef(branch_var_id, coord.to) -= Real{1};
    }
    J.coeffRef(branch_var_id, col_c) -= rm;
}

/// `coord` = port 2 (p2_pos, p2_neg); `p1_pos`, `p1_neg` = port 1.
inline void stamp_gyrator(sparse::Matrix& J, Vector& f, const Vector& x,
                          const BranchCoord& coord, Index p1_pos,
                          Index p1_neg, Real g) noexcept {
    // i_1 = g·v_2 drawn from p1_pos, returned at p1_neg.
    detail::stamp_transconductance(J, f, x, p1_pos, coord.from, coord.to, g);
    detail::stamp_transconductance(J, f, x, p1_neg, coord.from, coord.to, -g);
    // i_2 = −g·v_1 drawn from p2_pos, returned at p2_neg.
    detail::stamp_transconductance(J, f, x, coord.from, p1_pos, p1_neg, -g);
    detail::stamp_transconductance(J, f, x, coord.to, p1_pos, p1_neg, g);
}

}  // namespace pulsim::stamping
