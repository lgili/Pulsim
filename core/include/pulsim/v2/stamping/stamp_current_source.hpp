#pragma once

// =============================================================================
// Pulsim v2 — Layer 3: CurrentSource MNA stamping
// =============================================================================
//
// `pulsim-v2-current-source` Phase 2.
//
// Stamp a constant DC current source into the MNA system.
//
// EE convention: the user calls
// `add_current_source(name, from, to, I)` where:
//   * `from` is the source's POSITIVE terminal (current
//     EXITS the source here, INTO the external circuit).
//   * `to` is the NEGATIVE terminal.
//   * `I` is the magnitude flowing OUT of `from` into the
//     external circuit, then back to `to`.
//
// In v2's solve convention `J · x = −b_constant` (where
// b_constant captures the constant residual term), this
// translates to:
//   b_constant[from] −= I
//   b_constant[to]   += I
//
// Verification: a current source from `n0` to `gnd` with
// I = 10 mA in parallel with R = 1 kΩ from `n0` to `gnd`:
//   KCL at n0:  v_n0 / R = +I  →  v_n0 = +10 V.
// The signs above produce this.
//
// (v1 uses the same EE convention but solves N · x = +b,
// so v1's stamp is `b[npos] += I`. The signs differ
// because the solve-convention differs; the user-facing
// behaviour is identical.)
//
// Ground terminals are silently skipped (their KCL row
// doesn't exist in the active state vector).

#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"

namespace pulsim::v2::stamping {

inline void stamp_current_source(Vector& f,
                                   const BranchCoord& coord,
                                   Real I) noexcept {
    const bool from_active = node_is_active(coord.from);
    const bool to_active   = node_is_active(coord.to);

    // EE convention: `from` is + terminal (current EXITS
    // the source here into the external circuit).
    // v2's solve does J·x = −b_constant, so to get
    // J·x[from] = +I (i.e. external loads carry I OUT of
    // the from-node), we need b_constant[from] = −I.
    if (from_active) {
        f[coord.from] -= I;
    }
    if (to_active) {
        f[coord.to] += I;
    }
}

}  // namespace pulsim::v2::stamping
