#pragma once

// =============================================================================
// Pulsim — Layer 3: Fixed-state switch stamper
// =============================================================================
//
// `pulsim-v2-generic-stamping-pipeline` Phase 4.
//
// Stamps a switch as a fixed conductance (`g_on` if closed,
// `g_off` if open). Used by Layer 4's PWL state-space cache to
// materialise per-segment matrices: for each switch combination
// in the `SwitchStateMask`, iterate Switch-kind branches and call
// `stamp_switch_fixed` with the corresponding bit.
//
// `v_drop_on` (SwitchedDiode V_f) adds a series source while closed;
// J is unchanged, only the residual / b gets −g_on·V_f on the anode.
//
// Mathematically identical to a Resistor stamp with `G = closed ?
// g_on : g_off`. The function exists as a separate entry point
// because Layer 4 dispatches by `BranchKind::Switch` and the
// switch-state mask, NOT by device-type concept — it's clearer to
// have a single-purpose call than to construct a fake Resistor
// model.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"

namespace pulsim::stamping {

// -----------------------------------------------------------------------------
// stamp_switch_fixed — 2-terminal binary-conductance stamper.
//
// The fixed-conductance form means Layer 4 builds ONE per-segment
// matrix per switch combination, with the appropriate conductance
// per switch. Newton iteration only kicks in for the Nonlinear-kind
// devices stamped via `stamp_device<T>`; the switch contribution
// is exact / linear.
// -----------------------------------------------------------------------------
inline void stamp_switch_fixed(sparse::Matrix& J, Vector& f,
                                const Vector& x,
                                const BranchCoord& coord,
                                bool closed,
                                Real g_on,
                                Real g_off,
                                Real v_drop_on = Real{0}) noexcept {
    const Real G = closed ? g_on : g_off;

    const Real v_from = read_node_voltage(x, coord.from);
    const Real v_to   = read_node_voltage(x, coord.to);
    // A closed SwitchedDiode is g_on in series with its forward drop:
    // i = g_on·(v − V_f). Skipped at 0 so the V_f = 0 stamp (and the
    // sign of zero in b) is bit-identical to the plain conductance.
    Real v_br = v_from - v_to;
    if (closed && v_drop_on != Real{0}) v_br -= v_drop_on;
    const Real i = G * v_br;

    const bool from_active = node_is_active(coord.from);
    const bool to_active   = node_is_active(coord.to);

    if (from_active) f[coord.from] += i;
    if (to_active)   f[coord.to]   -= i;

    if (from_active) {
        J.coeffRef(coord.from, coord.from) += G;
        if (to_active) {
            J.coeffRef(coord.from, coord.to) -= G;
        }
    }
    if (to_active) {
        if (from_active) {
            J.coeffRef(coord.to, coord.from) -= G;
        }
        J.coeffRef(coord.to, coord.to) += G;
    }
}

}  // namespace pulsim::stamping
