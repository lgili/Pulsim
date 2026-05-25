#pragma once

// =============================================================================
// Pulsim — Layer 0: PulsimSparseLuSolver (in-house sparse LU)
// =============================================================================
//
// `openspec/changes/replace-klu-with-pulsim-sparse-lu` Sections 2-5 (v1.3.0).
// Templatized on `Scalar` in v1.4.0 via
// `openspec/changes/add-pulsim-complex-sparse-lu` so the same code path
// drives both real-valued PWL state-space MNA and complex-valued AC sweeps.
//
// In-house C++23 sparse LU implementation. Uses `Eigen::SparseMatrix` only as
// a passive matrix container — neither `Eigen::SparseLU` nor any third-party
// LU library is invoked.
//
// Implementation lands in stages:
//
//   * Section 2 (this header, initial version): symbolic analysis —
//     RCM column ordering (George 1971), elimination tree (Liu 1986 /
//     Davis 2006 §4.10), symbolic L+U pattern from the etree.
//     `analyze()` is the public entry; `factorize()` / `solve()` /
//     `partial_refactor()` return false / throw `not_implemented`
//     stubs until Sections 3-5 land.
//   * Section 3: numeric factorization with partial pivoting
//     (Gilbert & Peierls, *SIAM J. Sci. Stat. Comput.* 9, 1988)
//   * Section 4: triangular solve forward/back substitution
//     (Davis 2006 §3)
//   * Section 5: path-based partial refactor with pivot-fault
//     detection (Chan/Brandwajn/Tinney 1986, Dinkelbach 2021 §3)
//
// Complex template (v1.4.0):
//   The whole class body is `template <typename Scalar = Real>
//   class PulsimSparseLuSolverT`. The pivoting / threshold logic uses
//   `std::abs(Scalar)` which returns the underlying magnitude type
//   (`Real`) for both `double` and `std::complex<double>` — so the
//   existing comparisons keep meaning "largest magnitude column entry".
//   Zero comparisons use `Scalar{0}`.
//
//   Backward-compat aliases (following the `MatrixT`/`Matrix` and
//   `DirectSolverT`/`DirectSolver` pattern):
//     * `PulsimSparseLuSolver`        = `PulsimSparseLuSolverT<Real>`
//     * `PulsimComplexSparseLuSolver` = `PulsimSparseLuSolverT<std::complex<Real>>`
//   Every Layer 1-9 consumer that writes `PulsimSparseLuSolver solver;`
//   keeps compiling — the alias points back at the same instantiation.
//
// References cited throughout:
//   [1] Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006.
//   [2] George, "Computer Implementation of the Finite Element
//       Method," Stanford STAN-CS-71-208, 1971 — RCM ordering.
//   [3] Liu, "A Compact Row Storage Scheme for Cholesky Factors
//       Using Elimination Trees," ACM TOMS 12(2), 1986.
//   [4] Gilbert & Peierls, "Sparse Partial Pivoting in Time
//       Proportional to Arithmetic Operations," SIAM J. Sci. Stat.
//       Comput. 9(5), 1988.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>
#include <queue>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

namespace pulsim::sparse {

// -----------------------------------------------------------------------------
// MAX_PATH_LENGTH_RATIO (v1.4.0 — openspec/changes/add-generalised-path-refactor).
//
// Compile-time tunable. When `partial_refactor_count_path(changed_cols)`
// returns a path length L such that `L / n > MAX_PATH_LENGTH_RATIO`, the
// caller should prefer a fresh `factorize()` because the path-based
// update would cost approximately the same as a full factorisation
// (which also avoids potential floating-point drift from accumulating
// path-based updates).
//
// 0.6 is the empirical break-even on the chapter 8 microbench data;
// see `openspec/changes/add-generalised-path-refactor/design.md` Decision 2.
// Tune in the same file + recompile if benchmark data warrants.
// -----------------------------------------------------------------------------
inline constexpr Real MAX_PATH_LENGTH_RATIO = Real{0.6};

// The `Scalar = Real` default lives on the forward declaration in
// `solver.hpp`; C++ forbids restating it here.
template <typename Scalar>
class PulsimSparseLuSolverT final : public DirectSolverT<Scalar> {
public:
    using MatrixType = typename DirectSolverT<Scalar>::MatrixType;
    using VectorType = typename DirectSolverT<Scalar>::VectorType;

    PulsimSparseLuSolverT() noexcept = default;
    ~PulsimSparseLuSolverT() override = default;

    PulsimSparseLuSolverT(const PulsimSparseLuSolverT&) = delete;
    PulsimSparseLuSolverT& operator=(const PulsimSparseLuSolverT&) = delete;
    PulsimSparseLuSolverT(PulsimSparseLuSolverT&&) = delete;
    PulsimSparseLuSolverT& operator=(PulsimSparseLuSolverT&&) = delete;

    /// Symbolic factorization: computes the fill-reducing column
    /// permutation, the elimination tree, and the symbolic non-zero
    /// pattern of the upcoming L and U factors.
    ///
    /// Sections covered in this revision:
    ///   * Section 2 — RCM column ordering, etree, symbolic L+U
    ///     pattern. Returns true on valid input; false on
    ///     `M.rows() != M.cols()` or `M.rows() == 0`.
    [[nodiscard]] bool analyze(const MatrixType& M) override {
        analyzed_   = false;
        factorized_ = false;
        l_col_ptr_.clear();
        l_row_idx_.clear();
        u_col_ptr_.clear();
        u_row_idx_.clear();
        // Section 5: analyze() invalidates the path cache (etree itself
        // changes since the column permutation may change).
        invalidate_path_cache_();

        if (M.rows() != M.cols() || M.rows() == 0) {
            return false;
        }
        n_ = static_cast<Index>(M.rows());

        // 1. Build the symmetric adjacency of |M| + |M^T| — used by both
        //    the RCM ordering and the symbolic factorization. Each row/col
        //    pair (i, j) with `i != j` produces edges in both directions.
        const auto adj = build_symmetric_adjacency_(M);

        // 2. Fill-reducing column permutation via RCM.
        Pcol_     = compute_rcm_ordering_(adj);
        Pinv_col_.assign(static_cast<std::size_t>(n_), 0);
        for (Index k = 0; k < n_; ++k) {
            Pinv_col_[static_cast<std::size_t>(Pcol_[k])] = k;
        }

        // 3. Elimination tree on the permuted symmetric structure.
        etree_parent_ = compute_etree_(adj);

        // 4. Symbolic L+U pattern from the etree.
        compute_symbolic_pattern_(adj);

        analyzed_ = true;
        return true;
    }

    /// Numeric factorization via Gilbert-Peierls left-looking sparse LU
    /// (Gilbert & Peierls, *SIAM J. Sci. Stat. Comput.* 9, 1988; Davis
    /// 2006 §3). For each permuted column k = 0..n-1:
    ///   1. Initialize a dense workspace `x` from column `Pcol_[k]` of M
    ///      after applying the current row permutation `Prow_`.
    ///   2. Apply L-updates from previously-factored columns j < k:
    ///      for each j in the symbolic U[:, k] pattern (excluding the
    ///      diagonal), subtract `L[i, j] * x[j]` from every i in
    ///      L[:, j]'s pattern.
    ///   3. The diagonal entry x[k] becomes the pivot U[k, k]. Above-
    ///      diagonal x[i] (i < k) become U[i, k]. Below-diagonal entries
    ///      x[i] (i > k) divided by the pivot become L[i, k].
    ///
    /// Includes **partial pivoting** (column-by-column row swap to the
    /// largest-magnitude candidate). The row swap relabels indices in
    /// the already-stored L columns 0..k-1 (no new storage slots are
    /// needed — the symbolic |M|+|M^T| pattern computed in Section 2
    /// is invariant under row-only permutations). Required for circuit
    /// MNA matrices whose voltage-source constraint rows produce zero
    /// diagonals: without pivoting, Gilbert-Peierls hits zero pivot at
    /// the source's branch-current variable.
    ///
    /// Pivot magnitudes below `PIVOT_TOL` (1e-14) trigger
    /// `numeric_singular_ = true` and a `false` return; this signals
    /// genuine rank deficiency (no row swap can rescue it) and the
    /// caller is expected to surface the failure.
    ///
    /// Complex specialisation: `std::abs(Scalar)` returns `Real` for
    /// both `double` and `std::complex<double>`, so the pivot-magnitude
    /// comparisons keep semantics across both Scalar types. Pivot
    /// extraction stores `Scalar` values; `1 / pivot` is well-defined
    /// for the complex specialisation as long as the magnitude check
    /// just above guarantees `|pivot| > 0`.
    [[nodiscard]] bool factorize(const MatrixType& M) override {
        if (!analyzed_) {
            throw std::logic_error(
                "PulsimSparseLuSolverT::factorize called before analyze");
        }
        factorized_       = false;
        numeric_singular_ = false;
        if (M.rows() != n_ || M.cols() != n_) {
            return false;  // dimensions changed since analyze
        }

        // Initialize the row permutation to the SAME order as the column
        // permutation. Circuit MNA matrices are structurally near-symmetric
        // (every off-diagonal nonzero typically has a transpose partner),
        // so reordering both rows AND columns the same way keeps the
        // diagonal entries on the diagonal of M_perm — which is required
        // for Gilbert-Peierls without partial pivoting to find a non-zero
        // pivot at column k. Partial pivoting (deferred) would further
        // mutate Prow_ on top of this base ordering.
        Prow_     = std::vector<Index>(Pcol_.begin(), Pcol_.end());
        Pinv_row_ = std::vector<Index>(Pinv_col_.begin(), Pinv_col_.end());

        // Reset L and U storage to empty — factorize() recomputes the
        // pattern DYNAMICALLY from x's actual nonzeros per column. This
        // overrides any pattern previously populated by analyze()'s
        // symbolic step; the symbolic pattern is just a hint for
        // diagnostics, not used for storage allocation.
        //
        // Rationale: Section 2's symbolic pattern was computed against
        // the |M|+|M^T| symmetric structure under the assumption
        // Prow == Pcol. Partial pivoting in Section 3 mutates Prow,
        // which can introduce L/U entries at permuted rows that the
        // pre-pivot symbolic pattern didn't anticipate. Dynamic
        // discovery avoids the issue: we record every nonzero x[i]
        // post-elimination as an L or U entry.
        l_col_ptr_.assign(static_cast<std::size_t>(n_ + 1), Index{0});
        l_row_idx_.clear();
        l_values_.clear();
        u_col_ptr_.assign(static_cast<std::size_t>(n_ + 1), Index{0});
        u_row_idx_.clear();
        u_values_.clear();

        // Dense workspace for the current column. Reused across the k loop.
        std::vector<Scalar> x(static_cast<std::size_t>(n_), Scalar{0});

        const Index*  Ap = M.outerIndexPtr();
        const Index*  Ai = M.innerIndexPtr();
        const Scalar* Ax = M.valuePtr();

        constexpr Real PIVOT_TOL = Real{1e-14};

        for (Index k = 0; k < n_; ++k) {
            // ---- Step 1: load column k of M[Prow_, Pcol_] into x --------
            std::fill(x.begin(), x.end(), Scalar{0});
            const Index orig_col = Pcol_[static_cast<std::size_t>(k)];
            for (Index p = Ap[orig_col]; p < Ap[orig_col + 1]; ++p) {
                const Index orig_row = Ai[p];
                const Index perm_row = Pinv_row_[static_cast<std::size_t>(orig_row)];
                x[static_cast<std::size_t>(perm_row)] = Ax[p];
            }

            // ---- Step 2: apply L-updates from ALL prior columns where x[j] != 0
            // Dense workspace makes this O(k) per column for the j-loop
            // (with the inner work scaling with L[:, j]'s nnz). The total
            // O(n²) iteration cost is acceptable for circuit MNA at
            // n ≤ a few hundred; for larger n a Davis 2006 §3-style
            // reachability-based sparse triangular solve is the next
            // optimization.
            for (Index j = 0; j < k; ++j) {
                const Scalar xj = x[static_cast<std::size_t>(j)];
                if (xj == Scalar{0}) continue;
                for (Index q = l_col_ptr_[static_cast<std::size_t>(j)];
                     q < l_col_ptr_[static_cast<std::size_t>(j + 1)]; ++q) {
                    const Index i = l_row_idx_[static_cast<std::size_t>(q)];
                    x[static_cast<std::size_t>(i)] -=
                        l_values_[static_cast<std::size_t>(q)] * xj;
                }
            }

            // ---- Step 3a: partial pivoting --------------------------------
            // Find argmax |x[i]| for i ∈ [k, n_). If the largest is not
            // at position k, swap logical rows i_max ↔ k. The swap
            // relabels row indices in the already-stored L columns
            // 0..k-1 (no new storage slots needed — the SET of nonzero
            // logical rows per column is unchanged, only the labels
            // permute). The symbolic L+U pattern from Section 2 was
            // computed against the SYMMETRIC |M|+|M^T| structure, which
            // is an over-estimate that remains valid under row-only
            // permutations introduced by pivoting.
            //
            // Required for circuit MNA matrices with voltage-source
            // constraint rows (zero diagonal at the source's branch-
            // current variable). Without pivoting, M_perm has zero on
            // the diagonal at that position and factorization fails.
            //
            // Complex specialisation: `std::abs` of `std::complex<Real>`
            // returns the magnitude `Real`, so "largest magnitude"
            // remains the well-defined pivot criterion (matches LAPACK
            // ZGETRF semantics).
            Index i_max     = k;
            Real  max_abs   = std::abs(x[static_cast<std::size_t>(k)]);
            for (Index i = k + 1; i < n_; ++i) {
                const Real abs_xi =
                    std::abs(x[static_cast<std::size_t>(i)]);
                if (abs_xi > max_abs) {
                    max_abs = abs_xi;
                    i_max   = i;
                }
            }
            if (i_max != k) {
                // (a) workspace
                std::swap(x[static_cast<std::size_t>(k)],
                           x[static_cast<std::size_t>(i_max)]);
                // (b) stored L columns 0..k-1: relabel row indices
                for (Index j = 0; j < k; ++j) {
                    for (Index p = l_col_ptr_[static_cast<std::size_t>(j)];
                         p < l_col_ptr_[static_cast<std::size_t>(j + 1)];
                         ++p) {
                        const Index r = l_row_idx_[static_cast<std::size_t>(p)];
                        if (r == k) {
                            l_row_idx_[static_cast<std::size_t>(p)] = i_max;
                        } else if (r == i_max) {
                            l_row_idx_[static_cast<std::size_t>(p)] = k;
                        }
                    }
                }
                // (c) row permutation tracker
                const Index orig_k    = Prow_[static_cast<std::size_t>(k)];
                const Index orig_imax = Prow_[static_cast<std::size_t>(i_max)];
                std::swap(Prow_[static_cast<std::size_t>(k)],
                           Prow_[static_cast<std::size_t>(i_max)]);
                Pinv_row_[static_cast<std::size_t>(orig_k)]    = i_max;
                Pinv_row_[static_cast<std::size_t>(orig_imax)] = k;
            }

            // ---- Step 3b: pivot check + numeric extraction ---------------
            const Scalar pivot = x[static_cast<std::size_t>(k)];
            if (std::abs(pivot) < PIVOT_TOL) {
                numeric_singular_ = true;
                return false;
            }

            // Store U[:, k] — dynamically discover nonzero rows
            // i ∈ [0, k] in x. Diagonal is `pivot`. Row indices end
            // up sorted automatically (we iterate i in increasing
            // order). We update `u_col_ptr_[k+1]` at the END of this
            // column's push so that the NEXT iteration's L-update
            // loop sees the correct slice for j ≤ k.
            for (Index i = 0; i < k; ++i) {
                const Scalar xi = x[static_cast<std::size_t>(i)];
                if (xi != Scalar{0}) {
                    u_row_idx_.push_back(i);
                    u_values_.push_back(xi);
                }
            }
            u_row_idx_.push_back(k);
            u_values_.push_back(pivot);
            u_col_ptr_[static_cast<std::size_t>(k + 1)] =
                static_cast<Index>(u_row_idx_.size());

            // Store L[:, k] — dynamically discover nonzero rows
            // i ∈ (k, n). Values scaled by 1/pivot to give L unit-
            // lower-triangular form. Same end-of-push col_ptr update.
            const Scalar inv_pivot = Scalar{1} / pivot;
            for (Index i = k + 1; i < n_; ++i) {
                const Scalar xi = x[static_cast<std::size_t>(i)];
                if (xi != Scalar{0}) {
                    l_row_idx_.push_back(i);
                    l_values_.push_back(xi * inv_pivot);
                }
            }
            l_col_ptr_[static_cast<std::size_t>(k + 1)] =
                static_cast<Index>(l_row_idx_.size());
        }

        factorized_ = true;
        return true;
    }

    /// Triangular solve: `M · x = b` via the cached factor `L · U = P_row · M · P_col`.
    /// Three steps (Davis 2006 §3):
    ///   1. y ← P_row · b           (apply row permutation)
    ///   2. y ← L \ y               (forward substitution; L is unit-lower)
    ///   3. y ← U \ y               (back substitution; U is upper, diagonal stored last)
    ///   4. x[Pcol[k]] ← y[k]       (apply inverse column permutation to recover x)
    /// Throws `std::logic_error` if called before a successful `factorize`.
    void solve(const VectorType& b, VectorType& x) const override {
        if (!factorized_) {
            throw std::logic_error(
                "PulsimSparseLuSolverT::solve called before factorize "
                "(or factorize returned false). Call factorize(M) first "
                "and check its return value.");
        }

        // Apply row permutation: y[i] = b[Prow[i]].
        VectorType y(static_cast<Index>(n_));
        for (Index i = 0; i < n_; ++i) {
            y[i] = b[Prow_[static_cast<std::size_t>(i)]];
        }

        // ---- Step 2: forward substitution (L unit-lower triangular) ---
        // For each column k of L, propagate the now-known y[k] downward:
        // y[i] -= L[i, k] * y[k] for every (i, val) stored in L[:, k].
        for (Index k = 0; k < n_; ++k) {
            const Scalar yk = y[k];
            for (Index q = l_col_ptr_[static_cast<std::size_t>(k)];
                 q < l_col_ptr_[static_cast<std::size_t>(k + 1)]; ++q) {
                const Index i = l_row_idx_[static_cast<std::size_t>(q)];
                y[i] -= l_values_[static_cast<std::size_t>(q)] * yk;
            }
        }

        // ---- Step 3: back substitution (U upper triangular) ---------
        // U[:, k]'s storage convention from Section 3: entries with row
        // i < k come first (in ascending order), the diagonal entry
        // (row k) is the LAST slot. We divide by U[k, k] then propagate
        // y[k] backwards through column k's above-diagonal entries.
        for (Index k = n_ - 1; k >= 0; --k) {
            const Index diag_slot = u_col_ptr_[static_cast<std::size_t>(k + 1)] - 1;
            const Scalar ukk      = u_values_[static_cast<std::size_t>(diag_slot)];
            y[k] /= ukk;
            const Scalar yk = y[k];
            for (Index q = u_col_ptr_[static_cast<std::size_t>(k)];
                 q < diag_slot; ++q) {
                const Index i = u_row_idx_[static_cast<std::size_t>(q)];
                y[i] -= u_values_[static_cast<std::size_t>(q)] * yk;
            }
        }

        // ---- Step 4: apply inverse column permutation ----------------
        // The solved system is U·z = y in permuted column space; the
        // original solution x satisfies x[Pcol[k]] = z[k] = y[k].
        x.resize(static_cast<Index>(n_));
        for (Index k = 0; k < n_; ++k) {
            x[Pcol_[static_cast<std::size_t>(k)]] = y[k];
        }
    }

    [[nodiscard]] bool is_analyzed()   const noexcept override { return analyzed_; }
    [[nodiscard]] bool is_factorized() const noexcept override { return factorized_; }

    // -------------------------------------------------------------------------
    // Section 5 — path-based partial_refactor (Chan/Brandwajn/Tinney 1986;
    // Dinkelbach et al., *Energies* 14:7989, 2021, §3)
    // -------------------------------------------------------------------------

    [[nodiscard]] bool supports_partial_refactor() const noexcept override {
        return true;
    }

    /// Re-eliminate only the columns of the LU factor that depend on
    /// `changed_cols` (the columns of `new_M` whose values have changed
    /// since the most recent `factorize`). The sparsity pattern of M
    /// MUST be unchanged.
    ///
    /// Algorithm (Dinkelbach 2021 §3):
    ///   1. Update the lazy union `varying_set_` with `changed_cols`.
    ///   2. If the union grew, recompute the etree path from each
    ///      varying column up to the root. Walk caches in `path_`.
    ///   3. For each column k in `path_` (ascending order),
    ///      re-run Gilbert-Peierls's column step against `new_M`'s
    ///      values, using the EXISTING `Prow_` (no re-pivoting):
    ///        - Apply L-updates from j < k (uses both updated columns
    ///          earlier in path_ AND unchanged columns NOT in path_)
    ///        - Pivot-fault check: if |x[k]| < `PIVOT_TOL` OR if some
    ///          x[i] for i > k is significantly larger than x[k]
    ///          (within factor 1.1), the original pivot order is no
    ///          longer optimal — invalidate cache + return false
    ///        - Pattern check: if x has nonzero at a row not in the
    ///          existing L+U pattern for column k, the symbolic
    ///          structure changed — invalidate cache + return false
    ///        - Update L+U values in the existing CSC slots
    ///
    /// On any failure mode, invalidates the path cache and returns
    /// `false`. The caller then falls back to a full `factorize(new_M)`.
    [[nodiscard]] bool partial_refactor(
        const MatrixType& new_M,
        std::span<const Index> changed_cols) override {
        if (!factorized_) {
            return false;  // need a prior factor to refactor over
        }
        if (new_M.rows() != n_ || new_M.cols() != n_) {
            return false;
        }
        if (changed_cols.empty()) {
            return true;  // nothing to do
        }

        // 1. Update lazy union of varying columns (in ORIGINAL coords)
        bool need_recompute = !path_valid_;
        for (Index c : changed_cols) {
            if (c < 0 || c >= n_) {
                invalidate_path_cache_();
                return false;
            }
            auto [_, inserted] = varying_set_.insert(c);
            if (inserted) {
                need_recompute = true;
            }
        }

        // 2. Recompute path if union grew
        if (need_recompute) {
            compute_path_();
            path_valid_ = true;
        }

        // 3. Re-eliminate path columns
        std::vector<Scalar> x(static_cast<std::size_t>(n_), Scalar{0});
        std::vector<bool> in_pattern(static_cast<std::size_t>(n_), false);
        const Index*  Ap = new_M.outerIndexPtr();
        const Index*  Ai = new_M.innerIndexPtr();
        const Scalar* Ax = new_M.valuePtr();

        constexpr Real PIVOT_TOL        = Real{1e-14};
        // Threshold-pivoting tolerance: the cached pivot is acceptable
        // as long as its magnitude is at least PIVOT_THRESH × the
        // column infinity-norm. KLU's default is 0.001 (0.1%), giving
        // generous headroom to absorb the wide pivot-magnitude swings
        // common in circuit MNA between switch-state changes. Stricter
        // values cause excess fallback to full factorize without much
        // numerical benefit.
        constexpr Real PIVOT_THRESH     = Real{1e-3};

        for (Index k : path_) {
            // ---- Load x = new_M[Prow, Pcol[k]] -------------------------
            std::fill(x.begin(), x.end(), Scalar{0});
            const Index orig_col = Pcol_[static_cast<std::size_t>(k)];
            for (Index p = Ap[orig_col]; p < Ap[orig_col + 1]; ++p) {
                const Index orig_row = Ai[p];
                const Index perm_row = Pinv_row_[static_cast<std::size_t>(orig_row)];
                x[static_cast<std::size_t>(perm_row)] = Ax[p];
            }

            // ---- L-updates from j < k ---------------------------------
            for (Index j = 0; j < k; ++j) {
                const Scalar xj = x[static_cast<std::size_t>(j)];
                if (xj == Scalar{0}) continue;
                for (Index q = l_col_ptr_[static_cast<std::size_t>(j)];
                     q < l_col_ptr_[static_cast<std::size_t>(j + 1)]; ++q) {
                    const Index i = l_row_idx_[static_cast<std::size_t>(q)];
                    x[static_cast<std::size_t>(i)] -=
                        l_values_[static_cast<std::size_t>(q)] * xj;
                }
            }

            // ---- Pivot-fault check ------------------------------------
            const Scalar pivot = x[static_cast<std::size_t>(k)];
            const Real pivot_abs = std::abs(pivot);
            if (pivot_abs < PIVOT_TOL) {
                invalidate_path_cache_();
                return false;
            }
            // Threshold pivoting: reject if |x[k]| < PIVOT_THRESH ×
            // column infinity norm, i.e. some row's magnitude is more
            // than 1/PIVOT_THRESH × the current pivot. KLU-style; lets
            // typical switch-state swings through while catching true
            // pivot-order collapses.
            Real col_max = pivot_abs;
            for (Index i = k + 1; i < n_; ++i) {
                col_max = std::max(col_max,
                    std::abs(x[static_cast<std::size_t>(i)]));
            }
            if (pivot_abs < PIVOT_THRESH * col_max) {
                invalidate_path_cache_();
                return false;
            }

            // ---- Pattern check + value update -------------------------
            // Build a marker set of the existing L+U pattern for column k.
            for (Index q = u_col_ptr_[static_cast<std::size_t>(k)];
                 q < u_col_ptr_[static_cast<std::size_t>(k + 1)]; ++q) {
                in_pattern[static_cast<std::size_t>(
                    u_row_idx_[static_cast<std::size_t>(q)])] = true;
            }
            for (Index q = l_col_ptr_[static_cast<std::size_t>(k)];
                 q < l_col_ptr_[static_cast<std::size_t>(k + 1)]; ++q) {
                in_pattern[static_cast<std::size_t>(
                    l_row_idx_[static_cast<std::size_t>(q)])] = true;
            }
            // Verify no x[i] != 0 falls outside the existing pattern.
            bool pattern_ok = true;
            for (Index i = 0; i < n_; ++i) {
                if (x[static_cast<std::size_t>(i)] != Scalar{0} &&
                    !in_pattern[static_cast<std::size_t>(i)]) {
                    pattern_ok = false;
                    break;
                }
            }
            // Reset the marker for the next column iteration. We
            // touched only the existing-pattern positions; reset just
            // those (cheaper than std::fill over the whole vector).
            for (Index q = u_col_ptr_[static_cast<std::size_t>(k)];
                 q < u_col_ptr_[static_cast<std::size_t>(k + 1)]; ++q) {
                in_pattern[static_cast<std::size_t>(
                    u_row_idx_[static_cast<std::size_t>(q)])] = false;
            }
            for (Index q = l_col_ptr_[static_cast<std::size_t>(k)];
                 q < l_col_ptr_[static_cast<std::size_t>(k + 1)]; ++q) {
                in_pattern[static_cast<std::size_t>(
                    l_row_idx_[static_cast<std::size_t>(q)])] = false;
            }
            if (!pattern_ok) {
                invalidate_path_cache_();
                return false;
            }

            // Update U[:, k]'s values in-place at the existing slots.
            // x[u_row_idx_[q]] may be 0 — that's a sparse zero, fine.
            for (Index q = u_col_ptr_[static_cast<std::size_t>(k)];
                 q < u_col_ptr_[static_cast<std::size_t>(k + 1)]; ++q) {
                const Index i = u_row_idx_[static_cast<std::size_t>(q)];
                u_values_[static_cast<std::size_t>(q)] =
                    x[static_cast<std::size_t>(i)];
            }
            // Update L[:, k]'s values (scaled by 1/pivot).
            const Scalar inv_pivot = Scalar{1} / pivot;
            for (Index q = l_col_ptr_[static_cast<std::size_t>(k)];
                 q < l_col_ptr_[static_cast<std::size_t>(k + 1)]; ++q) {
                const Index i = l_row_idx_[static_cast<std::size_t>(q)];
                l_values_[static_cast<std::size_t>(q)] =
                    x[static_cast<std::size_t>(i)] * inv_pivot;
            }
        }

        return true;
    }

    /// Test-only: how many times has `compute_path_()` been invoked?
    /// Used by 5.7.2 to verify path caching across identical
    /// changed_cols calls.
    [[nodiscard]] std::uint64_t path_compute_count() const noexcept {
        return path_compute_count_;
    }

    /// Query: how many columns of the LU factor would
    /// `partial_refactor(new_M, changed_cols)` re-eliminate?
    ///
    /// Returns the length of the union path that `partial_refactor`
    /// would walk **without executing the refactor**. Pure read-only
    /// query — does not mutate `varying_set_`, `path_`, or any solver
    /// state. Callers use this with `MAX_PATH_LENGTH_RATIO` to decide
    /// between path-based update and a fresh `factorize()`.
    ///
    /// Algorithm: simulate the insertion of every column in
    /// `changed_cols` into `varying_set_` (without committing), then
    /// walk each member's etree path up to the root. Dedupe via an
    /// `in_path` bitmap so each path column is counted exactly once.
    ///
    /// Edge cases:
    ///   * `changed_cols` empty AND `varying_set_` empty → returns 0.
    ///   * Already-cached path (`changed_cols` ⊆ `varying_set_` and
    ///     `path_valid_`) → returns `path_.size()` directly without
    ///     re-walking the etree.
    ///   * Out-of-range column index → silently skipped (matches
    ///     `partial_refactor`'s pattern of returning false on
    ///     out-of-range, but the query is non-mutating so we return
    ///     a best-effort count over the in-range entries).
    [[nodiscard]] Index partial_refactor_count_path(
        std::span<const Index> changed_cols) const noexcept override {
        if (!factorized_ || n_ == 0) {
            return Index{0};
        }
        // Check whether changed_cols would expand varying_set_.
        bool would_grow = !path_valid_;
        for (Index c : changed_cols) {
            if (c < 0 || c >= n_) continue;
            if (!varying_set_.contains(c)) {
                would_grow = true;
                break;
            }
        }
        if (!would_grow) {
            return static_cast<Index>(path_.size());
        }
        // Walk the hypothetical union path WITHOUT mutating state.
        std::vector<bool> in_path(static_cast<std::size_t>(n_), false);
        Index count = 0;
        auto walk_from = [&](Index orig_c) {
            if (orig_c < 0 || orig_c >= n_) return;
            Index k = Pinv_col_[static_cast<std::size_t>(orig_c)];
            while (k != Index{-1} &&
                   !in_path[static_cast<std::size_t>(k)]) {
                in_path[static_cast<std::size_t>(k)] = true;
                ++count;
                k = etree_parent_[static_cast<std::size_t>(k)];
            }
        };
        for (Index c : varying_set_) walk_from(c);
        for (Index c : changed_cols)   walk_from(c);
        return count;
    }

    /// Convenience: returns the ratio
    /// `partial_refactor_count_path(cols) / n`. Equivalent to the
    /// `MAX_PATH_LENGTH_RATIO` comparison expression in caller code.
    [[nodiscard]] Real partial_refactor_path_ratio(
        std::span<const Index> changed_cols) const noexcept {
        if (n_ == 0) return Real{0};
        return static_cast<Real>(partial_refactor_count_path(changed_cols))
               / static_cast<Real>(n_);
    }

    // -------------------------------------------------------------------------
    // Test-only / introspection accessors
    // -------------------------------------------------------------------------

    [[nodiscard]] Index n() const noexcept { return n_; }

    /// Column permutation produced by `analyze()`. `column_permutation()[k]`
    /// is the index of the ORIGINAL column that becomes the k-th column
    /// after RCM reordering.
    [[nodiscard]] std::span<const Index> column_permutation() const noexcept {
        return Pcol_;
    }

    /// Elimination tree parent array. `etree_parent()[k] == -1` means k is
    /// a root of the elimination forest.
    [[nodiscard]] std::span<const Index> etree_parent() const noexcept {
        return etree_parent_;
    }

    /// Symbolic non-zero count in the L factor (strictly below the
    /// diagonal — the unit diagonal of L is implicit).
    [[nodiscard]] Index l_nnz() const noexcept {
        return l_row_idx_.empty() ? Index{0}
                                    : static_cast<Index>(l_row_idx_.size());
    }

    /// Symbolic non-zero count in the U factor (on and above the diagonal).
    [[nodiscard]] Index u_nnz() const noexcept {
        return u_row_idx_.empty() ? Index{0}
                                    : static_cast<Index>(u_row_idx_.size());
    }

    /// True after a failed `factorize` call when the numerical pivot fell
    /// below `PIVOT_TOL` (1e-14 of column infinity-norm). Caller can use
    /// this to distinguish numerical singularity from "didn't call analyze
    /// first" (which throws std::logic_error instead).
    [[nodiscard]] bool numeric_singular() const noexcept {
        return numeric_singular_;
    }

    /// Row permutation produced by the most recent `factorize`. Identity
    /// in Section 3 V0 (no partial pivoting yet). `row_permutation()[i]`
    /// is the index of the ORIGINAL row that becomes the i-th row after
    /// the eventual partial pivoting.
    [[nodiscard]] std::span<const Index> row_permutation() const noexcept {
        return Prow_;
    }

    /// Extract the strictly lower-triangular L factor as a dense-allocated
    /// `Eigen::SparseMatrix`. The implicit unit diagonal is NOT included
    /// (caller should add identity for `L * U == P_row · M · P_col`
    /// checks). Returns an n×n matrix; empty if not factorized.
    [[nodiscard]] MatrixType extract_L_matrix() const {
        MatrixType L(static_cast<Index>(n_), static_cast<Index>(n_));
        if (!factorized_) return L;
        std::vector<TripletT<Scalar>> trips;
        trips.reserve(l_row_idx_.size());
        for (Index k = 0; k < n_; ++k) {
            for (Index p = l_col_ptr_[static_cast<std::size_t>(k)];
                 p < l_col_ptr_[static_cast<std::size_t>(k + 1)]; ++p) {
                trips.emplace_back(l_row_idx_[static_cast<std::size_t>(p)], k,
                                    l_values_[static_cast<std::size_t>(p)]);
            }
        }
        L.setFromTriplets(trips.begin(), trips.end());
        L.makeCompressed();
        return L;
    }

    /// Extract the upper-triangular U factor (including the diagonal).
    [[nodiscard]] MatrixType extract_U_matrix() const {
        MatrixType U(static_cast<Index>(n_), static_cast<Index>(n_));
        if (!factorized_) return U;
        std::vector<TripletT<Scalar>> trips;
        trips.reserve(u_row_idx_.size());
        for (Index k = 0; k < n_; ++k) {
            for (Index p = u_col_ptr_[static_cast<std::size_t>(k)];
                 p < u_col_ptr_[static_cast<std::size_t>(k + 1)]; ++p) {
                trips.emplace_back(u_row_idx_[static_cast<std::size_t>(p)], k,
                                    u_values_[static_cast<std::size_t>(p)]);
            }
        }
        U.setFromTriplets(trips.begin(), trips.end());
        U.makeCompressed();
        return U;
    }

private:
    // -------------------------------------------------------------------------
    // 1. Symmetric-adjacency builder
    // -------------------------------------------------------------------------
    //
    // For each `(row_i, col_j)` non-zero in M (with `i != j`) emit edges
    // `i ~ j` AND `j ~ i` into the adjacency list. The result is the
    // adjacency graph of |M| + |M^T| with each edge stored once per
    // endpoint, sorted + deduplicated.
    //
    // This is the standard input format for both RCM and Davis 2006 §4.10
    // etree computation when M is structurally asymmetric (typical for
    // circuit MNA with controlled sources).
    //
    // Note: the adjacency depends only on the SPARSITY PATTERN of M, not
    // on its values, so this routine is Scalar-agnostic — we only ever
    // touch `outerIndexPtr()` and `innerIndexPtr()`.

    using AdjacencyList = std::vector<std::vector<Index>>;

    [[nodiscard]] AdjacencyList build_symmetric_adjacency_(const MatrixType& M) const {
        const Index n = static_cast<Index>(M.rows());
        AdjacencyList adj(static_cast<std::size_t>(n));
        const Index* Ap = M.outerIndexPtr();
        const Index* Ai = M.innerIndexPtr();

        for (Index j = 0; j < n; ++j) {
            for (Index p = Ap[j]; p < Ap[j + 1]; ++p) {
                const Index i = Ai[p];
                if (i != j) {
                    adj[static_cast<std::size_t>(i)].push_back(j);
                    adj[static_cast<std::size_t>(j)].push_back(i);
                }
            }
        }
        for (auto& nbr : adj) {
            std::sort(nbr.begin(), nbr.end());
            nbr.erase(std::unique(nbr.begin(), nbr.end()), nbr.end());
        }
        return adj;
    }

    // -------------------------------------------------------------------------
    // 2. Reverse Cuthill-McKee ordering (George 1971)
    // -------------------------------------------------------------------------
    //
    // Bandwidth-reducing column permutation. Loops over connected
    // components (always 1 for circuit MNA in practice but handled for
    // safety): start at the min-degree unvisited vertex, BFS visiting
    // neighbours in ascending-degree order, reverse the resulting sequence.

    [[nodiscard]] std::vector<Index> compute_rcm_ordering_(
        const AdjacencyList& adj) const {
        const Index n = static_cast<Index>(adj.size());
        if (n == 0) return {};

        std::vector<Index> degree(static_cast<std::size_t>(n));
        for (Index i = 0; i < n; ++i) {
            degree[static_cast<std::size_t>(i)] =
                static_cast<Index>(adj[static_cast<std::size_t>(i)].size());
        }

        std::vector<Index> result;
        result.reserve(static_cast<std::size_t>(n));
        std::vector<bool> visited(static_cast<std::size_t>(n), false);

        while (static_cast<Index>(result.size()) < n) {
            // Min-degree unvisited starting vertex
            Index start = -1;
            Index best  = std::numeric_limits<Index>::max();
            for (Index i = 0; i < n; ++i) {
                if (!visited[static_cast<std::size_t>(i)] &&
                    degree[static_cast<std::size_t>(i)] < best) {
                    start = i;
                    best  = degree[static_cast<std::size_t>(i)];
                }
            }
            if (start < 0) break;

            std::queue<Index> q;
            q.push(start);
            visited[static_cast<std::size_t>(start)] = true;
            while (!q.empty()) {
                const Index v = q.front();
                q.pop();
                result.push_back(v);

                // Collect unvisited neighbours and sort by ascending degree
                std::vector<Index> nbrs;
                for (Index u : adj[static_cast<std::size_t>(v)]) {
                    if (!visited[static_cast<std::size_t>(u)]) {
                        nbrs.push_back(u);
                        visited[static_cast<std::size_t>(u)] = true;
                    }
                }
                std::sort(nbrs.begin(), nbrs.end(),
                          [&](Index a, Index b) {
                              return degree[static_cast<std::size_t>(a)] <
                                     degree[static_cast<std::size_t>(b)];
                          });
                for (Index u : nbrs) q.push(u);
            }
        }

        std::reverse(result.begin(), result.end());
        return result;
    }

    // -------------------------------------------------------------------------
    // 3. Elimination tree (Davis 2006 §4.10 / Liu 1986)
    // -------------------------------------------------------------------------
    //
    // Computes the elimination-tree parent array on the column-permuted
    // symmetric structure. `parent[k] == -1` means column k has no
    // below-diagonal nonzero (root of the etree forest). Uses Liu's
    // disjoint-set "ancestor compression" variant — O(α(n)·nnz) per
    // Davis 2006 §4.10.

    [[nodiscard]] std::vector<Index> compute_etree_(
        const AdjacencyList& adj) const {
        const Index n = n_;
        std::vector<Index> parent(static_cast<std::size_t>(n), Index{-1});
        std::vector<Index> ancestor(static_cast<std::size_t>(n), Index{-1});

        // Process columns in PERMUTED order: at permuted column k we look
        // at adj[Pcol_[k]] (the original neighbours) and for each
        // neighbour map its column to its permuted position.
        for (Index k = 0; k < n; ++k) {
            const Index orig = Pcol_[static_cast<std::size_t>(k)];
            parent[static_cast<std::size_t>(k)]   = Index{-1};
            ancestor[static_cast<std::size_t>(k)] = Index{-1};
            for (Index orig_nbr : adj[static_cast<std::size_t>(orig)]) {
                Index i = Pinv_col_[static_cast<std::size_t>(orig_nbr)];
                while (i != Index{-1} && i < k) {
                    const Index next = ancestor[static_cast<std::size_t>(i)];
                    ancestor[static_cast<std::size_t>(i)] = k;
                    if (next == Index{-1}) {
                        parent[static_cast<std::size_t>(i)] = k;
                    }
                    i = next;
                }
            }
        }
        return parent;
    }

    // -------------------------------------------------------------------------
    // 4. Symbolic L+U pattern (Davis 2006 §4)
    // -------------------------------------------------------------------------
    //
    // For each permuted column k, the L pattern below the diagonal is the
    // union of:
    //   (a) direct nonzeros: permuted rows i > k that have an entry in
    //       the symmetric adjacency of column k
    //   (b) inherited fill: for each direct nonzero permuted row i with
    //       i < k, walk up the etree from i, gathering every row > k
    //       encountered (those become fill-in entries in L's column k)
    //
    // U above the diagonal mirrors the same set on the row dimension
    // (we treat the structure as symmetric per the |M|+|M^T| convention).
    // Together they give us a CSC pattern for L (strictly lower) and U
    // (upper + diagonal) ready for Section 3 numeric factorization.

    void compute_symbolic_pattern_(const AdjacencyList& adj) {
        const Index n = n_;
        l_col_ptr_.assign(static_cast<std::size_t>(n + 1), Index{0});
        u_col_ptr_.assign(static_cast<std::size_t>(n + 1), Index{0});

        // Per-column scratch: marker-array technique to avoid set ops.
        // `seen[i] == k` means row i has already been added to column k.
        std::vector<Index> seen(static_cast<std::size_t>(n), Index{-1});
        std::vector<Index> col_l_rows;
        std::vector<Index> col_u_rows;

        // First pass: gather row-index lists per column AND tally col_ptr
        // counts. We materialise the indices into l_row_idx_/u_row_idx_
        // immediately in their final CSC order.
        l_row_idx_.clear();
        u_row_idx_.clear();
        l_row_idx_.reserve(static_cast<std::size_t>(n));
        u_row_idx_.reserve(static_cast<std::size_t>(n));

        for (Index k = 0; k < n; ++k) {
            const Index orig = Pcol_[static_cast<std::size_t>(k)];
            col_l_rows.clear();
            col_u_rows.clear();

            // (a) direct neighbours in the symmetric structure
            for (Index orig_nbr : adj[static_cast<std::size_t>(orig)]) {
                const Index i = Pinv_col_[static_cast<std::size_t>(orig_nbr)];
                if (seen[static_cast<std::size_t>(i)] == k) continue;
                seen[static_cast<std::size_t>(i)] = k;
                if (i > k) {
                    col_l_rows.push_back(i);
                } else if (i < k) {
                    col_u_rows.push_back(i);
                }
                // i == k is the diagonal; handled by U's diagonal entry below.
            }

            // (b) inherited fill via etree walk: for each i < k that's
            // marked seen for column k, walk up the etree and add every
            // node encountered as a fill-in (these are L-side fills above k
            // in U-space, and the etree ensures we don't miss any).
            // We iterate a COPY of col_u_rows because the walk may extend it.
            std::vector<Index> walk_seed = col_u_rows;
            for (Index i : walk_seed) {
                Index node = etree_parent_[static_cast<std::size_t>(i)];
                while (node != Index{-1} && node < k) {
                    if (seen[static_cast<std::size_t>(node)] != k) {
                        seen[static_cast<std::size_t>(node)] = k;
                        col_u_rows.push_back(node);
                    }
                    node = etree_parent_[static_cast<std::size_t>(node)];
                }
                // The path also touches column k itself if reached — that's
                // the diagonal, no need to record.
            }

            // Sort + flush into the CSC arrays. CSC convention: row
            // indices within a column may be unsorted, but for solver
            // correctness and reproducibility we keep them sorted.
            std::sort(col_l_rows.begin(), col_l_rows.end());
            std::sort(col_u_rows.begin(), col_u_rows.end());

            // Diagonal entry always lives in U.
            col_u_rows.push_back(k);

            l_col_ptr_[static_cast<std::size_t>(k)] =
                static_cast<Index>(l_row_idx_.size());
            for (Index r : col_l_rows) l_row_idx_.push_back(r);

            u_col_ptr_[static_cast<std::size_t>(k)] =
                static_cast<Index>(u_row_idx_.size());
            for (Index r : col_u_rows) u_row_idx_.push_back(r);
        }
        l_col_ptr_[static_cast<std::size_t>(n)] =
            static_cast<Index>(l_row_idx_.size());
        u_col_ptr_[static_cast<std::size_t>(n)] =
            static_cast<Index>(u_row_idx_.size());
    }

    // -------------------------------------------------------------------------
    // 5. Path-based partial refactor helpers
    // -------------------------------------------------------------------------

    /// Compute the union of etree paths from each varying column up to
    /// the root. Stored ascending in `path_`. Each call increments
    /// `path_compute_count_` for diagnostic purposes.
    void compute_path_() {
        path_.clear();
        std::vector<bool> in_path(static_cast<std::size_t>(n_), false);
        for (Index orig_c : varying_set_) {
            Index k = Pinv_col_[static_cast<std::size_t>(orig_c)];
            while (k != Index{-1} &&
                   !in_path[static_cast<std::size_t>(k)]) {
                in_path[static_cast<std::size_t>(k)] = true;
                path_.push_back(k);
                k = etree_parent_[static_cast<std::size_t>(k)];
            }
        }
        std::sort(path_.begin(), path_.end());
        ++path_compute_count_;
    }

    /// Drop the cached path + varying-set state. Called from `analyze()`
    /// (symbolic factor invalidated by topology change) and on any
    /// pivot/pattern fault inside `partial_refactor`.
    void invalidate_path_cache_() {
        varying_set_.clear();
        path_.clear();
        path_valid_ = false;
    }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    Index n_ = 0;

    // Column permutation + inverse (size n). `Pcol_[k]` = original col at
    // new position k; `Pinv_col_[j]` = new position of original col j.
    std::vector<Index> Pcol_;
    std::vector<Index> Pinv_col_;

    // Elimination tree parent (size n; -1 = root).
    std::vector<Index> etree_parent_;

    // Symbolic L pattern (strictly lower triangular) in CSC:
    //   l_col_ptr_[k] .. l_col_ptr_[k+1]-1  →  l_row_idx_[...]
    std::vector<Index> l_col_ptr_;
    std::vector<Index> l_row_idx_;

    // Symbolic U pattern (upper triangular + diagonal) in CSC.
    std::vector<Index> u_col_ptr_;
    std::vector<Index> u_row_idx_;

    // Section 3 — numeric storage parallel to the symbolic CSC pattern.
    // Templatized on Scalar (v1.4.0): real-valued for PWL state-space MNA,
    // complex-valued for AC sweeps. Lengths still mirror l_row_idx_ /
    // u_row_idx_ exactly.
    std::vector<Scalar> l_values_;   // same length as l_row_idx_
    std::vector<Scalar> u_values_;   // same length as u_row_idx_

    // Section 5 — path-based partial refactor state.
    // `varying_set_` holds the union of all ORIGINAL column indices ever
    // passed as `changed_cols`. `path_` is the (sorted ascending) set
    // of PERMUTED column indices that depend on the varying set via the
    // elimination tree. Both are mutated by `partial_refactor`, cleared
    // by `analyze` and on any pivot/pattern fault.
    std::set<Index>    varying_set_;
    std::vector<Index> path_;
    bool               path_valid_         = false;
    std::uint64_t      path_compute_count_ = 0;

    // Section 3 — row permutation from partial pivoting. V0 of this
    // header initializes both to identity and never mutates them
    // (partial pivoting is deferred). The data structures are in place
    // so a follow-up commit can add pivoting without changing the
    // factorize() API.
    std::vector<Index> Prow_;       // size n; Prow_[i] = original row at new position i
    std::vector<Index> Pinv_row_;   // size n; inverse of Prow_

    bool analyzed_         = false;
    bool factorized_       = false;
    bool numeric_singular_ = false;
};

// -----------------------------------------------------------------------------
// Backward-compat type aliases (v1.4.0+).
//
// Following the codebase pattern (`MatrixT<Scalar>` + `Matrix = MatrixT<Real>`;
// `DirectSolverT<Scalar>` + `DirectSolver = DirectSolverT<Real>`):
//
//   * `PulsimSparseLuSolver`        — real-valued specialisation. Every
//     Layer 1-9 consumer that writes `PulsimSparseLuSolver solver;`
//     or `std::make_unique<PulsimSparseLuSolver>()` compiles unchanged.
//   * `PulsimComplexSparseLuSolver` — new complex specialisation,
//     consumed by `core/include/pulsim/analysis/mna_sweep.hpp` for AC
//     sweeps.
// -----------------------------------------------------------------------------
using PulsimSparseLuSolver        = PulsimSparseLuSolverT<Real>;
using PulsimComplexSparseLuSolver = PulsimSparseLuSolverT<std::complex<Real>>;

// -----------------------------------------------------------------------------
// Pulsim-aware factory implementation.
//
// Declared in `solver.hpp`; defined here at the bottom of
// `pulsim_lu_solver.hpp` so `PulsimSparseLuSolverT` is a complete type at
// the point we call `std::make_unique<PulsimSparseLuSolverT<...>>()`. Same
// pattern V0 used for KLU; ODR-safe because only ONE definition of the
// 2-arg overload exists per build.
//
// `Backend::Auto` returns `PulsimSparseLuSolverT<Scalar>` since v1.3.0 —
// the in-house solver is the default for the rank-1 PWL cache fast-path
// because it implements `partial_refactor`. `Backend::Eigen` remains
// available for parity testing + as a non-rank-1 baseline.
//
// Two factory entry points (v1.4.0+):
//   * `make_default_solver_t<Scalar>(n, hint)` — template factory used by
//     `mna_sweep.hpp` for the complex case.
//   * `make_default_solver(n, hint)` — non-template legacy shim that
//     dispatches to `make_default_solver_t<Real>(n, hint)`. Every Layer
//     1-9 consumer continues to call this overload unchanged.
// -----------------------------------------------------------------------------
template <typename Scalar>
[[nodiscard]] inline std::unique_ptr<DirectSolverT<Scalar>>
make_default_solver_t([[maybe_unused]] Size n, Backend hint) {
    switch (hint) {
        case Backend::Eigen:
            return std::make_unique<SparseLuSolverT<Scalar>>();
        case Backend::Pulsim:
        case Backend::Auto:
        default:
            return std::make_unique<PulsimSparseLuSolverT<Scalar>>();
    }
}

inline std::unique_ptr<DirectSolver> make_default_solver(Size n, Backend hint) {
    return make_default_solver_t<Real>(n, hint);
}

}  // namespace pulsim::sparse
