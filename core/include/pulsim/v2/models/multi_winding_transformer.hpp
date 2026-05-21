#pragma once

// =============================================================================
// Pulsim v2 — Layer 2 V16: Multi-winding (N=2..6) linear transformer
// =============================================================================
//
// Generalisation of TwoWindingTransformer (V2) to N windings:
//
//     v_i(t) = Σ_j M_ij · di_j/dt    for i = 0..N-1
//
// where M_ij = k_ij · √(L_i · L_j) is the mutual inductance
// between windings i and j, and M_ii = L_i (self-inductance).
//
// Use cases:
//   * 3-winding flyback (primary + main secondary + auxiliary
//     bias winding for the controller)
//   * Multi-output forward converters
//   * Centre-tapped push-pull (treated as 3 windings: primary,
//     sec-top, sec-bottom)
//   * Up to 6 windings supported in V0 (compile-time bound).
//
// Like the 2-winding model, this is a "metadata" device — the
// individual winding inductors are added separately via
// `add_inductor`, and the multi-winding transformer entry
// just stamps the CROSS-COUPLING terms M_ij·di_j/dt in the
// MNA matrix.
//
// Trap-companion cross-coupling pattern:
//   J[i_row, j_branch_var_col] += -(2·M_ij/dt)    for i ≠ j
//   b_extra[i_row] += (2·M_ij/dt) · i_j_prev      for i ≠ j

#include "pulsim/v2/numeric/types.hpp"

#include <array>
#include <cmath>

namespace pulsim::v2::models {

struct MultiWindingTransformer {
    // Compile-time maximum number of windings. Increase if
    // needed; the cost is a 6x6 → 8x8 coupling matrix in the
    // params struct (still small).
    static constexpr Size N_MAX = 6;

    struct Params {
        // Self-inductances per winding [H]. Only the first
        // `n_windings` entries are valid.
        std::array<Real, N_MAX> L_i{};
        // Coupling matrix k_ij ∈ [0, 1]. Symmetric: k_ij == k_ji.
        // Diagonal entries are unused (mutual_inductance(i,i)
        // returns L_i directly).
        std::array<std::array<Real, N_MAX>, N_MAX> k_ij{};
        Size n_windings = 0;
    };

    /// Mutual inductance M_ij = k_ij · √(L_i · L_j) for i ≠ j,
    /// or L_i for i == j. Clamps k_ij to [0, 1] defensively.
    [[nodiscard]] static Real mutual_inductance(
        const Params& p, Size i, Size j) noexcept {
        if (i == j) return p.L_i[i];
        Real k = p.k_ij[i][j];
        if (k < Real{0}) k = Real{0};
        if (k > Real{1}) k = Real{1};
        return k * std::sqrt(p.L_i[i] * p.L_i[j]);
    }

    /// Cross-coupling matrix entry: 2·M_ij / dt. Used by
    /// `assemble.hpp` to stamp the off-diagonal J entries for
    /// every (i, j) pair with i ≠ j.
    [[nodiscard]] static Real cross_dt(
        const Params& p, Size i, Size j,
        Real dt) noexcept {
        return Real{2} * mutual_inductance(p, i, j) / dt;
    }
};

}  // namespace pulsim::v2::models
