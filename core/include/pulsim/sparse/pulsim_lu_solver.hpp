#pragma once

// =============================================================================
// Pulsim — Layer 0: PulsimSparseLuSolver (in-house sparse LU)
// =============================================================================
//
// `openspec/changes/replace-klu-with-pulsim-sparse-lu` Sections 2-5.
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
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <vector>

namespace pulsim::sparse {

class PulsimSparseLuSolver final : public DirectSolver {
public:
    PulsimSparseLuSolver() noexcept = default;
    ~PulsimSparseLuSolver() override = default;

    PulsimSparseLuSolver(const PulsimSparseLuSolver&) = delete;
    PulsimSparseLuSolver& operator=(const PulsimSparseLuSolver&) = delete;
    PulsimSparseLuSolver(PulsimSparseLuSolver&&) = delete;
    PulsimSparseLuSolver& operator=(PulsimSparseLuSolver&&) = delete;

    /// Symbolic factorization: computes the fill-reducing column
    /// permutation, the elimination tree, and the symbolic non-zero
    /// pattern of the upcoming L and U factors.
    ///
    /// Sections covered in this revision:
    ///   * Section 2 — RCM column ordering, etree, symbolic L+U
    ///     pattern. Returns true on valid input; false on
    ///     `M.rows() != M.cols()` or `M.rows() == 0`.
    [[nodiscard]] bool analyze(const Matrix& M) override {
        analyzed_   = false;
        factorized_ = false;
        l_col_ptr_.clear();
        l_row_idx_.clear();
        u_col_ptr_.clear();
        u_row_idx_.clear();

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

    /// Numeric factorization — Section 3, not yet implemented. Returns
    /// false so callers see "not ready" cleanly without throwing.
    [[nodiscard]] bool factorize([[maybe_unused]] const Matrix& M) override {
        if (!analyzed_) {
            throw std::logic_error(
                "PulsimSparseLuSolver::factorize called before analyze");
        }
        // Section 3 (Gilbert-Peierls + partial pivoting) lands in a
        // follow-up commit. Stub returns false; callers fall back to
        // SparseLuSolver in the meantime.
        return false;
    }

    /// Triangular solve — Section 4, not yet implemented.
    void solve([[maybe_unused]] const Vector& b,
                [[maybe_unused]] Vector& x) const override {
        throw std::logic_error(
            "PulsimSparseLuSolver::solve not yet implemented (Section 4 "
            "of openspec/changes/replace-klu-with-pulsim-sparse-lu/)");
    }

    [[nodiscard]] bool is_analyzed()   const noexcept override { return analyzed_; }
    [[nodiscard]] bool is_factorized() const noexcept override { return factorized_; }

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

    using AdjacencyList = std::vector<std::vector<Index>>;

    [[nodiscard]] AdjacencyList build_symmetric_adjacency_(const Matrix& M) const {
        const Index n = static_cast<Index>(M.rows());
        AdjacencyList adj(static_cast<std::size_t>(n));
        const int* Ap = M.outerIndexPtr();
        const int* Ai = M.innerIndexPtr();

        for (Index j = 0; j < n; ++j) {
            for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
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

    bool analyzed_   = false;
    bool factorized_ = false;
};

// -----------------------------------------------------------------------------
// Pulsim-aware factory implementation.
//
// Declared in `solver.hpp`; defined here at the bottom of
// `pulsim_lu_solver.hpp` so `PulsimSparseLuSolver` is a complete type at
// the point we call `std::make_unique<PulsimSparseLuSolver>()`. Same
// pattern V0 used for KLU; ODR-safe because only ONE definition of the
// 2-arg overload exists per build.
//
// Backend::Auto behaviour during the interim (Sections 2 done, 3-5
// pending): falls through to `SparseLuSolver` because
// `PulsimSparseLuSolver::factorize` is still a Section-3 stub. Once
// Section 3 lands the real numeric factorization, flip this to pick
// `PulsimSparseLuSolver` for any n ≥ 1.
// -----------------------------------------------------------------------------
inline std::unique_ptr<DirectSolver> make_default_solver(
    [[maybe_unused]] Size n, Backend hint) {
    switch (hint) {
        case Backend::Pulsim:
            return std::make_unique<PulsimSparseLuSolver>();
        case Backend::Eigen:
            return std::make_unique<SparseLuSolver>();
        case Backend::Auto:
        default:
            return std::make_unique<SparseLuSolver>();
    }
}

}  // namespace pulsim::sparse
