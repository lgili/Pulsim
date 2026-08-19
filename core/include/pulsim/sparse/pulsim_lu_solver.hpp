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

#include <Eigen/OrderingMethods>

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

/// Fill-reducing ordering used by `analyze()` (Phase-1 LU core).
///
/// * `Colamd` (default since Phase 1) — Eigen's COLAMD, the ordering
///   designed for partial-pivoting LU (same family KLU/SuperLU use).
///   On circuit MNA it typically produces a small fraction of RCM's
///   fill: RCM minimises BANDWIDTH, which is the wrong objective for
///   factorisation fill (audit finding rcm-not-amd-btf, CONFIRMED).
/// * `Rcm` — the historical v1.3-v1.7 ordering, kept for A/B
///   benchmarking and regression comparison.
enum class LuOrdering { Colamd, Rcm };

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

        // 2. Fill-reducing column permutation.
        if (ordering_ == LuOrdering::Colamd) {
            // Eigen COLAMD convention (verified empirically on an
            // arrow matrix — hub must land LAST for near-zero fill):
            // P.indices()[orig] == new position of original column.
            Eigen::COLAMDOrdering<Index> colamd;
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic,
                                      Index> perm;
            colamd(M, perm);
            Pinv_col_.assign(static_cast<std::size_t>(n_), 0);
            Pcol_.assign(static_cast<std::size_t>(n_), 0);
            for (Index j = 0; j < n_; ++j) {
                const Index newpos = perm.indices()[j];
                Pinv_col_[static_cast<std::size_t>(j)] = newpos;
                Pcol_[static_cast<std::size_t>(newpos)] = j;
            }
        } else {
            Pcol_ = compute_rcm_ordering_(adj);
            Pinv_col_.assign(static_cast<std::size_t>(n_), 0);
            for (Index k = 0; k < n_; ++k) {
                Pinv_col_[static_cast<std::size_t>(Pcol_[k])] = k;
            }
        }

        // 3. Elimination tree on the permuted symmetric structure.
        etree_parent_ = compute_etree_(adj);

        // 4. Symbolic L+U pattern from the etree.
        compute_symbolic_pattern_(adj);

        analyzed_ = true;
        return true;
    }

    /// Numeric factorization — TRUE Gilbert-Peierls left-looking sparse
    /// LU with partial pivoting, O(flops) total (Gilbert & Peierls 1988;
    /// Davis 2006 §6, `cs_lu`). Phase-1 rewrite of the v1.3-v1.7
    /// implementation, whose inner loops were dense:
    ///   * the L-update loop scanned ALL prior columns (`for j < k`) —
    ///     O(n²) even for a tridiagonal matrix;
    ///   * pivot search scanned rows k..n densely;
    ///   * each pivot swap relabelled every stored L column — O(nnz(L))
    ///     per swap.
    /// The rewrite keeps the exact same public contract and factor
    /// conventions (L strictly-lower unit, pivotal row ids, ascending;
    /// U upper with the DIAGONAL STORED LAST per column) so `solve`,
    /// `partial_refactor` and every introspection accessor are
    /// unchanged.
    ///
    /// Per permuted column k:
    ///   1. Scatter A(:, Pcol_[k]) into the sparse workspace by
    ///      ORIGINAL row id; DFS each entry through the partially-built
    ///      L (rows stored by original id during elimination, resolved
    ///      via `pinv`: original row -> pivotal position) to obtain the
    ///      reach pattern in topological order — the nonzero pattern of
    ///      L \ A(:,col).
    ///   2. Eliminate along the topological order only (O(flops)).
    ///   3. Pivot = largest-magnitude candidate among NOT-yet-pivotal
    ///      pattern rows (ties: diagonal first, then smallest original
    ///      row id). |pivot| < 1e-14 → numeric_singular_. The "swap" is
    ///      O(1): assign `pinv[ipiv] = k` — no stored-column relabel.
    ///   4. Pattern rows already pivotal become U(:,k); the rest
    ///      become L(:,k) scaled by 1/pivot.
    /// After the loop one O(nnz) pass relabels L's row ids from
    /// original to pivotal coordinates and sorts each column — restoring
    /// the storage convention the rest of the class expects.
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
        const std::size_t nz = static_cast<std::size_t>(n_);

        l_col_ptr_.assign(nz + 1, Index{0});
        l_row_idx_.clear();
        l_values_.clear();
        u_col_ptr_.assign(nz + 1, Index{0});
        u_row_idx_.clear();
        u_values_.clear();

        // pinv[orig_row] = pivotal position (column that chose this row
        // as pivot), or -1 while the row is still un-pivoted.
        std::vector<Index> pinv(nz, Index{-1});
        Prow_.assign(nz, Index{0});

        // Sparse workspace (values by ORIGINAL row id) + DFS scratch.
        std::vector<Scalar> x(nz, Scalar{0});
        std::vector<Index>  xi(nz, Index{0});      // reach pattern stack
        std::vector<Index>  rstack(nz, Index{0});  // DFS node stack
        std::vector<Index>  pstack(nz, Index{0});  // DFS position stack
        std::vector<Index>  visit(nz, Index{-1});  // stamp = column k

        const Index*  Ap = M.outerIndexPtr();
        const Index*  Ai = M.innerIndexPtr();
        const Scalar* Ax = M.valuePtr();

        constexpr Real PIVOT_TOL = Real{1e-14};

        for (Index k = 0; k < n_; ++k) {
            Index top = n_;   // xi[top..n) will hold the reach pattern

            // ---- Step 1: scatter + DFS reach ------------------------
            const Index orig_col = Pcol_[static_cast<std::size_t>(k)];
            for (Index p = Ap[orig_col]; p < Ap[orig_col + 1]; ++p) {
                const Index r0 = Ai[p];
                x[static_cast<std::size_t>(r0)] = Ax[p];
                if (visit[static_cast<std::size_t>(r0)] == k) continue;
                // Iterative DFS from r0 over the partially-built L.
                Index head = 0;
                rstack[0] = r0;
                while (head >= 0) {
                    const Index r = rstack[static_cast<std::size_t>(head)];
                    const Index j = pinv[static_cast<std::size_t>(r)];
                    if (visit[static_cast<std::size_t>(r)] != k) {
                        visit[static_cast<std::size_t>(r)] = k;
                        pstack[static_cast<std::size_t>(head)] =
                            (j < 0) ? Index{0}
                                     : l_col_ptr_[static_cast<std::size_t>(j)];
                    }
                    bool done = true;
                    if (j >= 0) {
                        const Index pend =
                            l_col_ptr_[static_cast<std::size_t>(j + 1)];
                        for (Index q = pstack[static_cast<std::size_t>(head)];
                             q < pend; ++q) {
                            const Index rr =
                                l_row_idx_[static_cast<std::size_t>(q)];
                            if (visit[static_cast<std::size_t>(rr)] != k) {
                                pstack[static_cast<std::size_t>(head)] = q + 1;
                                ++head;
                                rstack[static_cast<std::size_t>(head)] = rr;
                                done = false;
                                break;
                            }
                        }
                    }
                    if (done) {
                        xi[static_cast<std::size_t>(--top)] = r;
                        --head;
                    }
                }
            }

            // ---- Step 2: sparse elimination in topological order ----
            for (Index p = top; p < n_; ++p) {
                const Index r = xi[static_cast<std::size_t>(p)];
                const Index j = pinv[static_cast<std::size_t>(r)];
                if (j < 0) continue;              // un-pivoted row: leaf
                const Scalar xj = x[static_cast<std::size_t>(r)];
                if (xj == Scalar{0}) continue;
                for (Index q = l_col_ptr_[static_cast<std::size_t>(j)];
                     q < l_col_ptr_[static_cast<std::size_t>(j + 1)]; ++q) {
                    x[static_cast<std::size_t>(
                        l_row_idx_[static_cast<std::size_t>(q)])] -=
                        l_values_[static_cast<std::size_t>(q)] * xj;
                }
            }

            // ---- Step 3: pivot among un-pivoted pattern rows --------
            Index ipiv    = Index{-1};
            Real  max_abs = Real{0};
            for (Index p = top; p < n_; ++p) {
                const Index r = xi[static_cast<std::size_t>(p)];
                if (pinv[static_cast<std::size_t>(r)] >= 0) continue;
                const Real a = std::abs(x[static_cast<std::size_t>(r)]);
                if (a > max_abs ||
                    (a == max_abs && ipiv >= 0 &&
                     (r == orig_col || (ipiv != orig_col && r < ipiv)))) {
                    max_abs = a;
                    ipiv    = r;
                }
            }
            // Diagonal preference on exact ties handled above; plain
            // largest-magnitude otherwise (matches the historical
            // pivoting criterion).
            if (ipiv < 0 || max_abs < PIVOT_TOL) {
                numeric_singular_ = true;
                // Clear workspace for a clean retry after refactor.
                for (Index p = top; p < n_; ++p) {
                    x[static_cast<std::size_t>(
                        xi[static_cast<std::size_t>(p)])] = Scalar{0};
                }
                return false;
            }
            const Scalar pivot     = x[static_cast<std::size_t>(ipiv)];
            const Scalar inv_pivot = Scalar{1} / pivot;

            // ---- Step 4: harvest U(:,k) and L(:,k) ------------------
            for (Index p = top; p < n_; ++p) {
                const Index r = xi[static_cast<std::size_t>(p)];
                const Index j = pinv[static_cast<std::size_t>(r)];
                const Scalar xr = x[static_cast<std::size_t>(r)];
                if (j >= 0) {
                    // already pivotal → U entry at pivotal row j
                    if (xr != Scalar{0}) {
                        u_row_idx_.push_back(j);
                        u_values_.push_back(xr);
                    }
                } else if (r != ipiv) {
                    // stays below the diagonal → L entry (ORIGINAL row
                    // id for now; relabelled after the k-loop)
                    if (xr != Scalar{0}) {
                        l_row_idx_.push_back(r);
                        l_values_.push_back(xr * inv_pivot);
                    }
                }
                x[static_cast<std::size_t>(r)] = Scalar{0};  // clear
            }
            // U's diagonal entry is stored LAST (solve() relies on it),
            // and the above-diagonal entries are sorted ascending for
            // deterministic storage.
            {
                const auto ub =
                    static_cast<std::size_t>(u_col_ptr_[nzk_(k)]);
                sort_col_pairs_(u_row_idx_, u_values_, ub,
                                 u_row_idx_.size());
            }
            u_row_idx_.push_back(k);
            u_values_.push_back(pivot);
            u_col_ptr_[static_cast<std::size_t>(k + 1)] =
                static_cast<Index>(u_row_idx_.size());
            l_col_ptr_[static_cast<std::size_t>(k + 1)] =
                static_cast<Index>(l_row_idx_.size());

            pinv[static_cast<std::size_t>(ipiv)] = k;
            Prow_[static_cast<std::size_t>(k)]   = ipiv;
        }

        // ---- Final pass: relabel L rows original → pivotal and sort.
        // O(nnz(L) + n·col·log) — once per factorize, this is what
        // buys the O(1) pivot "swap" inside the k-loop.
        for (auto& r : l_row_idx_) {
            r = pinv[static_cast<std::size_t>(r)];
        }
        for (Index k = 0; k < n_; ++k) {
            sort_col_pairs_(l_row_idx_, l_values_,
                static_cast<std::size_t>(
                    l_col_ptr_[static_cast<std::size_t>(k)]),
                static_cast<std::size_t>(
                    l_col_ptr_[static_cast<std::size_t>(k + 1)]));
        }

        Pinv_row_.assign(nz, Index{0});
        for (Index i = 0; i < n_; ++i) {
            Pinv_row_[static_cast<std::size_t>(
                Prow_[static_cast<std::size_t>(i)])] = i;
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

    /// Select the fill-reducing ordering for the NEXT analyze() call.
    /// Default: COLAMD (Phase-1). `LuOrdering::Rcm` restores the
    /// v1.3-v1.7 ordering for A/B comparison.
    void set_ordering(LuOrdering o) noexcept { ordering_ = o; }
    [[nodiscard]] LuOrdering ordering() const noexcept { return ordering_; }

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

        // 1. Path cache keyed by the CURRENT changed set — not a
        // monotone union (Phase-1 fix of audit finding
        // varying-set-monotone-union, CONFIRMED: after a successful
        // refactor the factor is fully consistent, so past columns
        // need no re-elimination; the old union-only-grows semantics
        // degraded every long run towards full-factorize cost). The
        // cache still serves the sweep fast path: repeated calls with
        // an identical changed set reuse the path without recompute.
        for (Index c : changed_cols) {
            if (c < 0 || c >= n_) {
                invalidate_path_cache_();
                return false;
            }
        }
        bool same_set = path_valid_ &&
            varying_set_.size() == changed_cols.size();
        if (same_set) {
            for (Index c : changed_cols) {
                if (!varying_set_.contains(c)) { same_set = false; break; }
            }
        }
        if (!same_set) {
            varying_set_.clear();
            varying_set_.insert(changed_cols.begin(), changed_cols.end());
            compute_path_();
            path_valid_ = true;
        }

        // 3. Re-eliminate path columns — sparse reach per column
        // (Phase-1: replaces the dense j<k scan, the dense pivot-max
        // scan and the O(n) pattern check with pattern-proportional
        // work; the factor now stores PIVOTAL row ids, and pivotal row
        // j < k is resolved by eliminated column j).
        const std::size_t nz = static_cast<std::size_t>(n_);
        std::vector<Scalar> x(nz, Scalar{0});
        std::vector<Index>  xi(nz, Index{0});
        std::vector<Index>  rstack(nz, Index{0});
        std::vector<Index>  pstack(nz, Index{0});
        std::vector<Index>  visit(nz, Index{-1});
        std::vector<bool>   in_pattern(nz, false);
        const Index*  Ap = new_M.outerIndexPtr();
        const Index*  Ai = new_M.innerIndexPtr();
        const Scalar* Ax = new_M.valuePtr();

        constexpr Real PIVOT_TOL    = Real{1e-14};
        constexpr Real PIVOT_THRESH = Real{1e-3};

        for (Index k : path_) {
            Index top = n_;
            // ---- scatter new_M[:, Pcol_[k]] by PIVOTAL row + reach ---
            const Index orig_col = Pcol_[static_cast<std::size_t>(k)];
            for (Index p = Ap[orig_col]; p < Ap[orig_col + 1]; ++p) {
                const Index r0 = Pinv_row_[static_cast<std::size_t>(Ai[p])];
                x[static_cast<std::size_t>(r0)] = Ax[p];
                if (visit[static_cast<std::size_t>(r0)] == k) continue;
                Index head = 0;
                rstack[0] = r0;
                while (head >= 0) {
                    const Index r = rstack[static_cast<std::size_t>(head)];
                    const bool resolved = (r < k);   // eliminated column
                    if (visit[static_cast<std::size_t>(r)] != k) {
                        visit[static_cast<std::size_t>(r)] = k;
                        pstack[static_cast<std::size_t>(head)] =
                            resolved
                                ? l_col_ptr_[static_cast<std::size_t>(r)]
                                : Index{0};
                    }
                    bool done = true;
                    if (resolved) {
                        const Index pend =
                            l_col_ptr_[static_cast<std::size_t>(r + 1)];
                        for (Index q = pstack[static_cast<std::size_t>(head)];
                             q < pend; ++q) {
                            const Index rr =
                                l_row_idx_[static_cast<std::size_t>(q)];
                            if (visit[static_cast<std::size_t>(rr)] != k) {
                                pstack[static_cast<std::size_t>(head)] = q + 1;
                                ++head;
                                rstack[static_cast<std::size_t>(head)] = rr;
                                done = false;
                                break;
                            }
                        }
                    }
                    if (done) {
                        xi[static_cast<std::size_t>(--top)] = r;
                        --head;
                    }
                }
            }

            // ---- eliminate in topological order ----------------------
            for (Index p = top; p < n_; ++p) {
                const Index r = xi[static_cast<std::size_t>(p)];
                if (r >= k) continue;                 // not eliminated
                const Scalar xr = x[static_cast<std::size_t>(r)];
                if (xr == Scalar{0}) continue;
                for (Index q = l_col_ptr_[static_cast<std::size_t>(r)];
                     q < l_col_ptr_[static_cast<std::size_t>(r + 1)]; ++q) {
                    x[static_cast<std::size_t>(
                        l_row_idx_[static_cast<std::size_t>(q)])] -=
                        l_values_[static_cast<std::size_t>(q)] * xr;
                }
            }

            // ---- pivot-fault check (pattern-proportional) ------------
            const Scalar pivot     = x[static_cast<std::size_t>(k)];
            const Real   pivot_abs = std::abs(pivot);
            if (pivot_abs < PIVOT_TOL) {
                for (Index p = top; p < n_; ++p)
                    x[static_cast<std::size_t>(
                        xi[static_cast<std::size_t>(p)])] = Scalar{0};
                invalidate_path_cache_();
                return false;
            }
            Real col_max = pivot_abs;
            for (Index p = top; p < n_; ++p) {
                const Index r = xi[static_cast<std::size_t>(p)];
                if (r > k) {
                    col_max = std::max(col_max,
                        std::abs(x[static_cast<std::size_t>(r)]));
                }
            }
            if (pivot_abs < PIVOT_THRESH * col_max) {
                for (Index p = top; p < n_; ++p)
                    x[static_cast<std::size_t>(
                        xi[static_cast<std::size_t>(p)])] = Scalar{0};
                invalidate_path_cache_();
                return false;
            }

            // ---- pattern check + in-place value update ---------------
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
            bool pattern_ok = true;
            for (Index p = top; p < n_; ++p) {
                const Index r = xi[static_cast<std::size_t>(p)];
                if (x[static_cast<std::size_t>(r)] != Scalar{0} &&
                    !in_pattern[static_cast<std::size_t>(r)]) {
                    pattern_ok = false;
                    break;
                }
            }
            if (pattern_ok) {
                for (Index q = u_col_ptr_[static_cast<std::size_t>(k)];
                     q < u_col_ptr_[static_cast<std::size_t>(k + 1)]; ++q) {
                    u_values_[static_cast<std::size_t>(q)] =
                        x[static_cast<std::size_t>(
                            u_row_idx_[static_cast<std::size_t>(q)])];
                }
                const Scalar inv_pivot = Scalar{1} / pivot;
                for (Index q = l_col_ptr_[static_cast<std::size_t>(k)];
                     q < l_col_ptr_[static_cast<std::size_t>(k + 1)]; ++q) {
                    l_values_[static_cast<std::size_t>(q)] =
                        x[static_cast<std::size_t>(
                            l_row_idx_[static_cast<std::size_t>(q)])] *
                        inv_pivot;
                }
            }
            // reset markers + workspace over touched entries only
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
            for (Index p = top; p < n_; ++p) {
                x[static_cast<std::size_t>(
                    xi[static_cast<std::size_t>(p)])] = Scalar{0};
            }
            if (!pattern_ok) {
                invalidate_path_cache_();
                return false;
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
        // Non-monotone semantics (Phase-1): the hypothetical path is
        // that of the CURRENT changed set alone — matching what
        // partial_refactor would actually walk. Identical-set calls
        // reuse the cached size.
        bool same_set = path_valid_ &&
            varying_set_.size() == changed_cols.size();
        if (same_set) {
            for (Index c : changed_cols) {
                if (c < 0 || c >= n_ || !varying_set_.contains(c)) {
                    same_set = false;
                    break;
                }
            }
        }
        if (same_set) {
            return static_cast<Index>(path_.size());
        }
        std::vector<bool> in_path(static_cast<std::size_t>(n_), false);
        Index count = 0;
        for (Index orig_c : changed_cols) {
            if (orig_c < 0 || orig_c >= n_) continue;
            Index k = Pinv_col_[static_cast<std::size_t>(orig_c)];
            while (k != Index{-1} &&
                   !in_path[static_cast<std::size_t>(k)]) {
                in_path[static_cast<std::size_t>(k)] = true;
                ++count;
                k = etree_parent_[static_cast<std::size_t>(k)];
            }
        }
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
    // Small helper: (k) -> size_t for u_col_ptr_ addressing at column
    // start (used before the diagonal push in factorize()).
    [[nodiscard]] std::size_t nzk_(Index k) const noexcept {
        return static_cast<std::size_t>(k);
    }

    /// Sort the (row, value) pairs of one CSC column slice ascending by
    /// row id. Used by factorize()'s final relabel pass and the U
    /// above-diagonal harvest. Insertion sort — column slices are short
    /// and mostly ordered.
    static void sort_col_pairs_(std::vector<Index>& rows,
                                 std::vector<Scalar>& vals,
                                 std::size_t lo, std::size_t hi) {
        for (std::size_t i = lo + 1; i < hi; ++i) {
            const Index  r = rows[i];
            const Scalar v = vals[i];
            std::size_t j = i;
            while (j > lo && rows[j - 1] > r) {
                rows[j] = rows[j - 1];
                vals[j] = vals[j - 1];
                --j;
            }
            rows[j] = r;
            vals[j] = v;
        }
    }

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
    LuOrdering ordering_   = LuOrdering::Colamd;
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
