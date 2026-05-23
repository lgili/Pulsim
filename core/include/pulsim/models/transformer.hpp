#pragma once

// =============================================================================
// Pulsim v2 — Layer 2 V2: Two-winding linear transformer
// =============================================================================
//
// `pulsim-v2-transformer` Phase 1.
//
// Coupled-inductors model:
//
//     v_p(t) = L_p · di_p/dt + M · di_s/dt
//     v_s(t) = M  · di_p/dt + L_s · di_s/dt
//
// with mutual inductance
//
//     M = k · √(L_p · L_s)
//
// and coupling coefficient `k ∈ [0, 1]`. k = 0 → independent
// inductors. k = 1 → ideal coupling (no leakage).
//
// The model does NOT carry a `current(v)` function — the
// constitutive relations are stamped via cross-coupling
// terms in the MNA matrix, not per-branch I-V.
//
// Trap-companion stamping pattern (added on top of each
// inductor's existing self-stamping):
//
//   J[p_row, s_branch_var_col] += -(2M/dt)
//   J[s_row, p_branch_var_col] += -(2M/dt)
//   b_extra[p_row] += (2M/dt) · i_s_prev
//   b_extra[s_row] += (2M/dt) · i_p_prev
//
// where `p_row` and `s_row` are the constraint-row indices
// for the two inductors (which equal their branch-var
// column indices in v2's MNA layout).

#include "pulsim/numeric/types.hpp"

#include <cmath>

namespace pulsim::models {

struct TwoWindingTransformer {
    struct Params {
        Real L_p;            // primary self-inductance [H]
        Real L_s;            // secondary self-inductance [H]
        Real k = Real{1};    // coupling coefficient ∈ [0,1]
    };

    /// Mutual inductance M = k · √(L_p · L_s) [H]. Clamps
    /// k to [0, 1] defensively; negative L is undefined
    /// behaviour and we don't guard it (the user's fault).
    [[nodiscard]] static Real mutual_inductance(
        const Params& p) noexcept {
        const Real k_clamped = (p.k < Real{0}) ? Real{0}
                              : (p.k > Real{1}) ? Real{1}
                              : p.k;
        return k_clamped * std::sqrt(p.L_p * p.L_s);
    }

    /// Cross-coupling matrix entry magnitude:
    ///     cross_dt = 2·M / dt
    ///
    /// Used by `assemble.hpp` to stamp the off-diagonal
    /// J entries between the two winding branch-current
    /// rows, and by `HistoryState::compute_b_extra` to add
    /// the cross-current history term to b_extra.
    [[nodiscard]] static Real cross_dt(
        const Params& p, Real dt) noexcept {
        return Real{2} * mutual_inductance(p) / dt;
    }
};

}  // namespace pulsim::models
