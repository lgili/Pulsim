#pragma once

// =============================================================================
// Pulsim v2 — Layer 0: sparse matrix wrapper
// =============================================================================
//
// `bootstrap-pulsim-v2-kernel` Phase 2.
//
// Sparse matrix layout: column-major, int32 indices.
//
//   * ColMajor matches every direct sparse solver's expected input
//     format (Eigen SparseLU, KLU, UMFPACK, MKL Pardiso). RowMajor
//     would force a transpose-and-copy at every solver call.
//
//   * ColMajor also matches the MNA stamping pattern: each device
//     stamps a few entries in a few columns (the device's nodes);
//     ColMajor means those entries live next to each other in memory.
//
//   * Index = int32 packs the index arrays at 4 B/entry. KLU and
//     UMFPACK expect int32 by default — no extra wrapper.
//
// Pure aliases + a few stamping utilities. No operator overloads
// beyond what Eigen already provides — wrapping would multiply
// maintenance for zero gain.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <Eigen/SparseCore>

namespace pulsim::sparse {

// -----------------------------------------------------------------------------
// Matrix — column-major, int32-index sparse matrix.
//
// This is THE matrix type for Layer 4's state-space cache and Layer 5's
// MNA. Choosing it correctly here means future layers compose naturally.
// -----------------------------------------------------------------------------
using Matrix = Eigen::SparseMatrix<Real, Eigen::ColMajor, Index>;

// -----------------------------------------------------------------------------
// Triplet — (row, col, value) entry for triplet-based assembly.
//
// Standard pattern:
//     std::vector<Triplet> triplets;
//     triplets.emplace_back(row, col, value);
//     ...
//     Matrix M(rows, cols);
//     M.setFromTriplets(triplets.begin(), triplets.end());
// -----------------------------------------------------------------------------
using Triplet = Eigen::Triplet<Real, Index>;

// -----------------------------------------------------------------------------
// stamp_dense — add a small dense block at (row, col).
//
// Used by Layer 4 to stamp state-space (A, B, C, D) sub-blocks into
// the assembled MNA matrix. Returns the count of entries added
// (block.rows() · block.cols()) so callers can sanity-check.
//
// The block's (i, j) entry lands at M(row + i, col + j). Entries that
// happen to be zero ARE still added to the triplet stream — that's the
// caller's responsibility to filter if it matters (typically it doesn't,
// because the next `setFromTriplets` deduplicates and the zero entries
// are pruned by the subsequent `makeCompressed`).
//
// Pre-condition: M.rows() ≥ row + block.rows() and M.cols() ≥ col +
//   block.cols(). Out-of-bounds is undefined behaviour (matches Eigen).
// -----------------------------------------------------------------------------
inline Size stamp_dense(Matrix& M, Index row, Index col,
                        const DenseMatrix& block) noexcept {
    for (Index j = 0; j < block.cols(); ++j) {
        for (Index i = 0; i < block.rows(); ++i) {
            M.coeffRef(row + i, col + j) += block(i, j);
        }
    }
    return static_cast<Size>(block.rows()) * static_cast<Size>(block.cols());
}

// -----------------------------------------------------------------------------
// reserve_capacity — reserve room for nnz_estimate non-zeros without
// changing structural sparsity.
//
// Wraps `Eigen::SparseMatrix::reserve` with a clearer name. Call before
// repeated `coeffRef` insertions to avoid quadratic re-allocation.
//
// PRECONDITION: Eigen's reserve() must be called on a freshly-
// constructed matrix (zero or sufficient capacity) BEFORE any
// `coeffRef` inserts. Calling after inserts on an uncompressed
// matrix aborts. This is an Eigen API constraint inherited as-is.
// -----------------------------------------------------------------------------
inline void reserve_capacity(Matrix& M, Size nnz_estimate) noexcept {
    M.reserve(static_cast<Eigen::Index>(nnz_estimate));
}

// -----------------------------------------------------------------------------
// compress_in_place — ensure the matrix is in compressed form.
//
// Direct sparse solvers (SparseLU, KLU, ...) expect compressed input.
// Call this once after assembly, before passing to a solver.
// -----------------------------------------------------------------------------
inline void compress_in_place(Matrix& M) noexcept {
    M.makeCompressed();
}

}  // namespace pulsim::sparse
