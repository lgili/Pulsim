#pragma once

// =============================================================================
// Pulsim v2 — Layer 2: Inductor device model (trap companion, branch-current)
// =============================================================================
//
// `pulsim-v2-trapezoidal-companion` Phase 1.
//
// Linear two-terminal dynamic device:  v_L = L · di_L/dt.
//
// At each timestep n → n+1, the trapezoidal rule yields:
//
//     i_{n+1} − i_n = (dt / 2L) · (v_n + v_{n+1})
//
// Rearranging:
//
//     i_{n+1}  =  i_n  +  (dt/2L) · v_n  +  (dt/2L) · v_{n+1}
//              =  G_L,eq · v_{n+1}  +  I_hist,L
//
//     G_L,eq    = dt / (2L)
//     I_hist,L  = i_n + G_L,eq · v_n
//
// V2 uses the **branch-current formulation**: the inductor's
// branch current `i_L` is an explicit MNA unknown (just like a
// voltage source), and the constraint row reads (after scaling
// by 2L/dt to keep the +1/−1 voltage-column pattern):
//
//     v_{n+1, from} − v_{n+1, to} − (2L/dt) · i_{n+1}
//                                          + (2L/dt) · I_hist,L = 0
//
// This composes naturally with the existing branch-current
// plumbing for voltage sources (DevicePool already counts them).
//
// The companion contributions are decomposed between
// "assembled" (J + b_constant) and "history" (b_extra at solve
// time):
//   * J  has +1, -1 on the voltage columns; −(2L/dt) on the
//     branch_var_id column of the constraint row; +1, -1 from
//     the constraint row's terminal columns; and the KCL
//     contributions from the branch current (the +i_L on the
//     from-row and −i_L on the to-row, identical to a voltage
//     source).
//   * b_constant is 0 — assembly emits no constants for the
//     inductor.
//   * b_extra(branch_var_id_row) = −(2L/dt) · I_hist,L at solve
//     time.

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::models {

struct Inductor {
    struct Params {
        Real L;     // henries
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::PassiveLinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;
    static constexpr bool is_dynamic = true;
    static constexpr bool needs_branch_unknown = true;

    /// Conductance reciprocal used in the constraint row.
    /// g_eq_inv = dt / (2L). The constraint row stamps
    /// −(2L/dt) = −1 / g_eq_inv on the inductor's branch_var_id
    /// diagonal entry.
    [[nodiscard]] static Real g_eq_inv(Real dt,
                                        const Params& p) noexcept {
        return dt / (Real{2} * p.L);
    }

    /// History term at step n+1: I_hist,L = i_n + (dt/2L) · v_n.
    /// Layer 5's HistoryState tracks (v_n, i_n) and computes this
    /// each step; the contribution to b_extra (the `b` portion of
    /// the constraint row's residual) is
    ///   b_extra(branch_var_id_row) += +(2L/dt) · I_hist,L
    /// (sign-consistent with cache.solve doing rhs = -b_total).
    [[nodiscard]] static Real history_term(Real v_prev, Real i_prev,
                                            Real dt,
                                            const Params& p) noexcept {
        return i_prev + g_eq_inv(dt, p) * v_prev;
    }

    /// Static current contract — inductors don't fit the
    /// "current as a function of voltage" model. Layer 3
    /// stamping uses companion-stamp helpers.
    template <numeric::FloatingPoint S>
    [[nodiscard]] static constexpr S current(const S* /*v*/,
                                              const Params& /*p*/) noexcept {
        return S{0};
    }
};

}  // namespace pulsim::models
