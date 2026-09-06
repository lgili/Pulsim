#pragma once

// =============================================================================
// Pulsim — Layer 2 V16: Multi-winding (N ≥ 2) linear transformer
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
//   * Any number of windings (the coupling registry is unbounded).
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

#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace pulsim::models {

struct MultiWindingTransformer {
    struct Params {
        //! Self-inductances per winding [H]; size() is N.
        std::vector<Real> L_i{};
        //! Coupling matrix k_ij ∈ [0, 1], N×N; only i < j is read.
        //! Empty means k = 1 everywhere off the diagonal.
        std::vector<std::vector<Real>> k_ij{};
        [[nodiscard]] Size n_windings() const noexcept { return L_i.size(); }
    };

    [[nodiscard]] static Real coupling(const Params& p, Size i, Size j) noexcept {
        if (i == j) return Real{1};
        if (i > j) std::swap(i, j);
        if (p.k_ij.empty() || i >= p.k_ij.size() || j >= p.k_ij[i].size()) return Real{1};
        return std::clamp(p.k_ij[i][j], Real{0}, Real{1});
    }

    /// Mutual inductance M_ij = k_ij · √(L_i · L_j) for i ≠ j,
    /// or L_i for i == j.
    [[nodiscard]] static Real mutual_inductance(
        const Params& p, Size i, Size j) noexcept {
        if (i == j) return p.L_i[i];
        return coupling(p, i, j) * std::sqrt(p.L_i[i] * p.L_i[j]);
    }

    /// Cross-coupling matrix entry: 2·M_ij / dt.
    [[nodiscard]] static Real cross_dt(
        const Params& p, Size i, Size j, Real dt) noexcept {
        return Real{2} * mutual_inductance(p, i, j) / dt;
    }

    /// REALISABILITY. Any three windings with k₁₂ = k₁₃ = 1 and
    /// k₂₃ = 0 describe a transformer that cannot exist: the
    /// inductance matrix [M_ij] is then indefinite, the stored
    /// energy ½ iᵀ M i can go NEGATIVE, and a simulation of it is
    /// an oscillator with no physics behind it. Pair-wise couplings
    /// let a user write that down without noticing. A set of
    /// windings is realisable exactly when M is positive
    /// semi-definite; this returns the smallest Cholesky pivot
    /// (relative to the largest L), negative or zero when it is not.
    [[nodiscard]] static Real min_relative_pivot(const Params& p) noexcept {
        const Size n = p.n_windings();
        if (n == 0) return Real{0};
        std::vector<std::vector<Real>> a(n, std::vector<Real>(n));
        Real L_max = Real{0};
        for (Size i = 0; i < n; ++i) {
            L_max = std::max(L_max, p.L_i[i]);
            for (Size j = 0; j < n; ++j) a[i][j] = mutual_inductance(p, i, j);
        }
        // Cholesky without the square roots (LDLᵀ): D_k are the
        // pivots; the first non-positive one ends it.
        Real worst = std::numeric_limits<Real>::infinity();
        std::vector<std::vector<Real>> l(n, std::vector<Real>(n, Real{0}));
        std::vector<Real> d(n, Real{0});
        for (Size j = 0; j < n; ++j) {
            Real sum = a[j][j];
            for (Size k = 0; k < j; ++k) sum -= l[j][k] * l[j][k] * d[k];
            d[j] = sum;
            worst = std::min(worst, sum / L_max);
            if (!(sum > Real{0})) return worst;
            for (Size i = j + 1; i < n; ++i) {
                Real s2 = a[i][j];
                for (Size k = 0; k < j; ++k) s2 -= l[i][k] * l[j][k] * d[k];
                l[i][j] = s2 / d[j];
            }
        }
        return worst;
    }
};

}  // namespace pulsim::models
