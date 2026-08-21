#pragma once

// =============================================================================
// Pulsim — Layer 0: direct sparse solver interface + reference impl
// =============================================================================
//
// `bootstrap-pulsim-v2-kernel` Phase 2.
// Templatized on Scalar in v1.4.0 (`add-pulsim-complex-sparse-lu`) so the
// same interface serves both real-valued PWL state-space MNA and
// complex-valued AC sweeps.
//
// The DirectSolverT<Scalar> interface separates the three phases of a
// direct sparse solve:
//
//     analyze  ── symbolic factorization (sparsity pattern only)
//        │       runs ONCE per topology change
//        ▼
//     factorize ── numeric factorization (uses cached pattern)
//        │        runs ONCE per matrix-value change
//        ▼
//     solve     ── triangular solve using the cached L+U
//                  runs every step
//
// This separation is the foundation of the Layer 4 PWL state-space
// cache: for a stable switch combination, the sparsity pattern AND
// the matrix values are constant across many simulation steps. So
//     `analyze + factorize` run ONCE
//     `solve` runs every step
// That's the 5-10× speedup vs v1's current cache.
//
// Backward-compat: `DirectSolver`, `SparseLuSolver`, and the
// non-templated `make_default_solver` continue to work for all Layer
// 1-9 consumers. They are now defined as typedefs to the
// `Scalar = Real` instantiation.
//
// Complex support: `DirectSolverT<std::complex<Real>>` is the new entry
// point for `core/include/pulsim/analysis/mna_sweep.hpp` (v1.4.0+).

#include "pulsim/sparse/matrix.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>

#include <complex>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace pulsim::sparse {

// -----------------------------------------------------------------------------
// DirectSolverT<Scalar> — abstract base for direct sparse solvers.
//
// Lifecycle contract:
//   1. `analyze(M)` — call once per topology change.
//      Returns false if M is structurally singular (caller stops here).
//   2. `factorize(M)` — call once per matrix-value change.
//      MUST be preceded by a successful `analyze`. Returns false if
//      M is numerically singular (zero pivot).
//   3. `solve(b, x)` — call as many times as needed.
//      MUST be preceded by a successful `factorize`.
//
// Calls out of order throw `std::logic_error` with a clear message
// naming the missing prerequisite step.
//
// `DirectSolver = DirectSolverT<Real>` is the backward-compat alias
// every existing Layer 1-9 consumer uses unchanged.
// -----------------------------------------------------------------------------
template <typename Scalar = Real>
class DirectSolverT {
public:
    using MatrixType = MatrixT<Scalar>;
    using VectorType = VectorT<Scalar>;

    DirectSolverT() = default;
    virtual ~DirectSolverT() = default;

    DirectSolverT(const DirectSolverT&) = delete;
    DirectSolverT& operator=(const DirectSolverT&) = delete;

    /// Symbolic factorization. Returns true on success, false if M is
    /// structurally singular (rank < rows by inspection of sparsity).
    /// Result depends only on M's sparsity pattern, not on its values.
    [[nodiscard]] virtual bool analyze(const MatrixType& M) = 0;

    /// Numeric factorization. Returns true on success, false if M is
    /// numerically singular (zero pivot encountered). MUST be preceded
    /// by a successful `analyze` call on a matrix with matching pattern.
    [[nodiscard]] virtual bool factorize(const MatrixType& M) = 0;

    /// Triangular solve `L · U · x = b` using the cached factor.
    /// MUST be preceded by a successful `factorize`. Throws
    /// std::logic_error if called out of order.
    virtual void solve(const VectorType& b, VectorType& x) const = 0;

    /// Diagnostic: has analyze been called and succeeded?
    [[nodiscard]] virtual bool is_analyzed() const noexcept = 0;
    /// Diagnostic: has factorize been called and succeeded?
    [[nodiscard]] virtual bool is_factorized() const noexcept = 0;

    // -------------------------------------------------------------------------
    // Rank-1 / partial-refactor extension (Layer 4 V8 —
    // openspec/changes/add-pwl-rank1-update). Subclasses that support an
    // O(path) re-elimination on a small set of perturbed columns override
    // these; the default implementations announce "not supported" and let
    // the caller fall back to a full `factorize`.
    // -------------------------------------------------------------------------

    /// Does this backend implement `partial_refactor`? Default: no.
    [[nodiscard]] virtual bool supports_partial_refactor() const noexcept {
        return false;
    }

    /// Re-eliminate only the columns of the LU factor that depend on the
    /// listed `changed_cols` of `new_M`. The caller MUST have already
    /// successfully `analyze`d a matrix with the same sparsity pattern as
    /// `new_M`, and SHOULD have `factorize`d the previous values at least
    /// once.
    ///
    /// `changed_cols` may have arbitrary length; the implementation
    /// computes the union of elimination-tree paths over all listed
    /// columns and re-eliminates each path column exactly once
    /// (deduplicated via an `in_path` bitmap). The single-column case
    /// is preserved as a sub-case of this generalised path-union.
    ///
    /// Returns true on success; false on any failure mode (numerical
    /// singularity in the updated columns, unsupported backend, etc.).
    /// On `false`, the cache invariants are preserved — the caller can
    /// safely fall back to a fresh `factorize(new_M)`.
    ///
    /// Default implementation: ignore the change set and return false
    /// (signals "not supported", forces the caller's fallback path).
    [[nodiscard]] virtual bool partial_refactor(
        [[maybe_unused]] const MatrixType& new_M,
        [[maybe_unused]] std::span<const Index> changed_cols) {
        return false;
    }

    /// Query: how many columns would `partial_refactor(new_M,
    /// changed_cols)` re-eliminate? Returns the union-path length
    /// **without** executing the refactor. Used by callers to decide
    /// between path-based update and full factorize via the
    /// `MAX_PATH_LENGTH_RATIO` heuristic (see
    /// `openspec/changes/add-generalised-path-refactor` Part A).
    ///
    /// Default implementation: return 0 (backends that don't support
    /// `partial_refactor` give zero work; the caller's ratio
    /// comparison falls into the "go path-based" branch but
    /// `partial_refactor` will return false anyway, triggering full
    /// `factorize()`).
    [[nodiscard]] virtual Index partial_refactor_count_path(
        [[maybe_unused]] std::span<const Index> changed_cols) const noexcept {
        return Index{0};
    }

    /// Estimated resident bytes of this solver's numeric factors
    /// (L/U values + indices + permutation/etree arrays). Consumed
    /// by the Layer 4 cache's byte-budget LRU (v2.0 Phase 1,
    /// audit finding `no-mode-cache-eviction`). An ESTIMATE for
    /// budget arithmetic — not an allocator audit. Default: 0
    /// (backends without introspection contribute only their
    /// caller-visible matrix storage to the budget).
    [[nodiscard]] virtual std::size_t factor_bytes() const noexcept {
        return 0;
    }

    /// Index of the matrix column whose elimination failed in the
    /// most recent `factorize`, in ORIGINAL (pre-ordering) index
    /// space — or `kInvalidIndex` when the backend cannot say.
    ///
    /// v2.0 Phase 1 (audit finding
    /// `singular-errors-dont-name-the-node`): for a square MNA
    /// system this index IS an unknown — a node voltage or a branch
    /// current — so a caller can turn "numerically singular for mask
    /// 0010111…1" into "node vout has no path to ground".
    ///
    /// Default: `kInvalidIndex`. The Eigen backend deliberately
    /// keeps the default: Eigen's SparseLU leaks its failing column
    /// only inside a free-text `lastErrorMessage()` that is compiled
    /// out under `EIGEN_NO_IO`, and parsing that would be worse than
    /// saying nothing. Every caller must therefore tolerate
    /// `kInvalidIndex` and degrade to the un-named message.
    [[nodiscard]] virtual Index singular_index() const noexcept {
        return kInvalidIndex;
    }
};

// Backward-compat alias: every Layer 1-9 consumer that uses
// `DirectSolver` continues to compile unchanged.
using DirectSolver = DirectSolverT<Real>;

// -----------------------------------------------------------------------------
// Backend hint for the factory (Layer 4 V8 — openspec/changes/replace-klu-with-pulsim-sparse-lu).
//
// `Auto`   — `PulsimSparseLuSolver<Scalar>` (v1.3.0+ for Real; v1.4.0+ for
//            std::complex<Real>).
// `Eigen`  — force `SparseLuSolverT<Scalar>`. Retained for parity testing
//            vs the in-house `PulsimSparseLuSolver`.
// `Pulsim` — force `PulsimSparseLuSolver<Scalar>` (explicit equivalent
//            of `Auto`).
// -----------------------------------------------------------------------------
enum class Backend { Auto, Eigen, Pulsim };

// -----------------------------------------------------------------------------
// SparseLuSolverT<Scalar> — reference implementation via Eigen::SparseLU.
//
// Kept intentionally as the benchmark baseline for the IEEE TPEL paper's
// microbench tables (Backend::Eigen). Production code paths use
// PulsimSparseLuSolver<Scalar> (v1.3.0+ for real; v1.4.0+ for complex).
//
// `SparseLuSolver = SparseLuSolverT<Real>` is the backward-compat alias.
// -----------------------------------------------------------------------------
template <typename Scalar = Real>
class SparseLuSolverT final : public DirectSolverT<Scalar> {
public:
    using MatrixType = typename DirectSolverT<Scalar>::MatrixType;
    using VectorType = typename DirectSolverT<Scalar>::VectorType;

    SparseLuSolverT() = default;

    [[nodiscard]] bool analyze(const MatrixType& M) override {
        // Eigen's analyzePattern records the column-ordering /
        // symbolic structure of M. It does not set m_info to Success
        // or Failure (those are only meaningful after factorize). The
        // analysis succeeds as long as the matrix dimensions are
        // sane; structural singularity surfaces as a factorize
        // failure later in the lifecycle. We mirror that contract.
        if (M.rows() != M.cols() || M.rows() == 0) {
            analyzed_ = false;
            factorized_ = false;
            return false;
        }
        impl_.analyzePattern(M);
        analyzed_ = true;
        factorized_ = false;
        return true;
    }

    [[nodiscard]] bool factorize(const MatrixType& M) override {
        if (!analyzed_) {
            throw std::logic_error(
                "SparseLuSolverT::factorize called before analyze "
                "(or analyze returned false). Call analyze(M) first "
                "and check its return value.");
        }
        impl_.factorize(M);
        factorized_ = (impl_.info() == Eigen::Success);
        return factorized_;
    }

    void solve(const VectorType& b, VectorType& x) const override {
        if (!factorized_) {
            throw std::logic_error(
                "SparseLuSolverT::solve called before factorize "
                "(or factorize returned false). Call factorize(M) first "
                "and check its return value.");
        }
        x = impl_.solve(b);
    }

    [[nodiscard]] bool is_analyzed()   const noexcept override { return analyzed_; }
    [[nodiscard]] bool is_factorized() const noexcept override { return factorized_; }

private:
    // COLAMDOrdering explicit on Index (int32) — Eigen's SparseLU
    // default ordering template parameter is COLAMDOrdering<int> which
    // can collide with Eigen::Index (int64 on most 64-bit platforms)
    // when the matrix's StorageIndex is int32. Making it explicit on
    // our int32 Index avoids the collision and the ensuing abort.
    Eigen::SparseLU<MatrixType, Eigen::COLAMDOrdering<Index>> impl_;
    bool analyzed_   = false;
    bool factorized_ = false;
};

// Backward-compat alias for all current Layer 1-9 consumers.
using SparseLuSolver = SparseLuSolverT<Real>;

// -----------------------------------------------------------------------------
// make_default_solver — factory.
//
// `make_default_solver()` (no args): always SparseLuSolver — safe baseline.
//
// `make_default_solver(n, hint)` honours the `Backend` enum above and
// returns the Scalar=Real path (backward compatible).
//
// `make_default_solver_t<Scalar>(n, hint)` is the new template entry
// point used by `mna_sweep.hpp` for the complex case (v1.4.0+).
//
// The Pulsim-aware factory impl lives at the bottom of
// `pulsim_lu_solver.hpp` (same include-at-bottom pattern V0 used for
// klu_solver.hpp): only one definition exists per build, ODR-safe,
// regardless of whether the user pulls in `solver.hpp` first or
// `pulsim_lu_solver.hpp` first.
// -----------------------------------------------------------------------------
template <typename Scalar = Real>
class PulsimSparseLuSolverT;  // forward decl — see pulsim_lu_solver.hpp

inline std::unique_ptr<DirectSolver> make_default_solver() {
    return std::make_unique<SparseLuSolver>();
}

[[nodiscard]] std::unique_ptr<DirectSolver> make_default_solver(
    Size n, Backend hint = Backend::Auto);

// Template factory for the complex / generic case (v1.4.0+). The
// non-template overload above is the backward-compat shim that
// dispatches to `make_default_solver_t<Real>(n, hint)`.
template <typename Scalar>
[[nodiscard]] std::unique_ptr<DirectSolverT<Scalar>>
make_default_solver_t(Size n, Backend hint = Backend::Auto);

}  // namespace pulsim::sparse

#include "pulsim/sparse/pulsim_lu_solver.hpp"
