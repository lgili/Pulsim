#pragma once

// =============================================================================
// Pulsim — Layer 0: direct sparse solver interface + reference impl
// =============================================================================
//
// `bootstrap-pulsim-v2-kernel` Phase 2.
//
// The DirectSolver interface separates the three phases of a direct
// sparse solve:
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
// That's the 5-10× speedup vs v1's current cache (which doesn't
// separate the two and redoes both on every miss).
//
// Layer 0 ships ONE concrete implementation: SparseLuSolver wrapping
// Eigen::SparseLU. Future layers can register KLU, UMFPACK, MKL
// Pardiso through the same interface — adding them does NOT touch
// any consumer that depends only on DirectSolver.

#include "pulsim/sparse/matrix.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>

#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace pulsim::sparse {

// -----------------------------------------------------------------------------
// DirectSolver — abstract base for direct sparse solvers.
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
// -----------------------------------------------------------------------------
class DirectSolver {
public:
    DirectSolver() = default;
    virtual ~DirectSolver() = default;

    DirectSolver(const DirectSolver&) = delete;
    DirectSolver& operator=(const DirectSolver&) = delete;

    /// Symbolic factorization. Returns true on success, false if M is
    /// structurally singular (rank < rows by inspection of sparsity).
    /// Result depends only on M's sparsity pattern, not on its values.
    [[nodiscard]] virtual bool analyze(const Matrix& M) = 0;

    /// Numeric factorization. Returns true on success, false if M is
    /// numerically singular (zero pivot encountered). MUST be preceded
    /// by a successful `analyze` call on a matrix with matching pattern.
    [[nodiscard]] virtual bool factorize(const Matrix& M) = 0;

    /// Triangular solve `L · U · x = b` using the cached factor.
    /// MUST be preceded by a successful `factorize`. Throws
    /// std::logic_error if called out of order.
    virtual void solve(const Vector& b, Vector& x) const = 0;

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
    /// Returns true on success; false on any failure mode (numerical
    /// singularity in the updated columns, unsupported backend, etc.).
    /// On `false`, the cache invariants are preserved — the caller can
    /// safely fall back to a fresh `factorize(new_M)`.
    ///
    /// Default implementation: ignore the change set and return false
    /// (signals "not supported", forces the caller's fallback path).
    [[nodiscard]] virtual bool partial_refactor(
        [[maybe_unused]] const Matrix& new_M,
        [[maybe_unused]] std::span<const Index> changed_cols) {
        return false;
    }
};

// -----------------------------------------------------------------------------
// Backend hint for the factory (Layer 4 V8).
//
// `Auto`  — pick KLU when `PULSIM_HAVE_KLU` is defined AND `n` is at or above
//           the partial-refactor crossover threshold (`PULSIM_KLU_AUTO_THRESHOLD`,
//           default 100). Otherwise pick Eigen::SparseLU.
// `Eigen` — force `SparseLuSolver` regardless of KLU availability.
// `KLU`   — force `KluSolver`. Throws `std::runtime_error` if `PULSIM_HAVE_KLU`
//           is not defined.
// -----------------------------------------------------------------------------
enum class Backend { Auto, Eigen, KLU };

#ifndef PULSIM_KLU_AUTO_THRESHOLD
/// Matrix dimension at which `Backend::Auto` switches from `SparseLuSolver`
/// to `KluSolver` (the regime where KLU's analyze-once + partial-refactor
/// amortisation outweighs Eigen's full-factorize cost; see Q6 of the
/// 2026-05-24 PWL library audit).
#define PULSIM_KLU_AUTO_THRESHOLD 100
#endif

// -----------------------------------------------------------------------------
// SparseLuSolver — reference implementation via Eigen::SparseLU.
//
// This is the default solver Layer 0 ships. Faster, multi-threaded
// backends (KLU, MKL Pardiso) can be added later through the same
// DirectSolver interface without touching consumers.
// -----------------------------------------------------------------------------
class SparseLuSolver final : public DirectSolver {
public:
    SparseLuSolver() = default;

    [[nodiscard]] bool analyze(const Matrix& M) override {
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

    [[nodiscard]] bool factorize(const Matrix& M) override {
        if (!analyzed_) {
            throw std::logic_error(
                "SparseLuSolver::factorize called before analyze "
                "(or analyze returned false). Call analyze(M) first "
                "and check its return value.");
        }
        impl_.factorize(M);
        factorized_ = (impl_.info() == Eigen::Success);
        return factorized_;
    }

    void solve(const Vector& b, Vector& x) const override {
        if (!factorized_) {
            throw std::logic_error(
                "SparseLuSolver::solve called before factorize "
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
    Eigen::SparseLU<Matrix, Eigen::COLAMDOrdering<Index>> impl_;
    bool analyzed_   = false;
    bool factorized_ = false;
};

// -----------------------------------------------------------------------------
// make_default_solver — factory.
//
// V0 default (n unknown): always SparseLuSolver — safe baseline that needs no
// optional dependency.
//
// V1 (Layer 4 V8 — openspec/changes/add-pwl-rank1-update) adds an overload
// that takes the matrix dimension `n` and an optional `Backend` hint:
//
//   * `Backend::Auto`  → KLU when `PULSIM_HAVE_KLU` is defined AND
//                        n >= `PULSIM_KLU_AUTO_THRESHOLD`. Else SparseLU.
//   * `Backend::Eigen` → always SparseLuSolver.
//   * `Backend::KLU`   → always KluSolver. Throws if KLU was not built in.
//
// `KluSolver`'s declaration lives in `klu_solver.hpp`; the factory needs
// only a forward declaration here. The actual ctor call is gated behind
// `#ifdef PULSIM_HAVE_KLU` to keep the `Backend::KLU` path callable without
// KLU at all (it throws at runtime with a clear message).
// -----------------------------------------------------------------------------
class KluSolver;  // forward decl — see pulsim/sparse/klu_solver.hpp

inline std::unique_ptr<DirectSolver> make_default_solver() {
    return std::make_unique<SparseLuSolver>();
}

[[nodiscard]] std::unique_ptr<DirectSolver> make_default_solver(
    Size n, Backend hint = Backend::Auto);

#ifndef PULSIM_HAVE_KLU
// KLU-less inline impl. When PULSIM_HAVE_KLU is defined the impl lives in
// klu_solver.hpp (pulled in below) and uses `KluSolver`; this fallback
// honours Backend::Eigen and Backend::Auto via SparseLuSolver and throws
// loudly on an explicit Backend::KLU request.
inline std::unique_ptr<DirectSolver> make_default_solver(
    [[maybe_unused]] Size n, Backend hint) {
    switch (hint) {
        case Backend::KLU:
            throw std::runtime_error(
                "make_default_solver(n, Backend::KLU): Pulsim was built "
                "without SuiteSparse KLU (PULSIM_HAVE_KLU not defined). "
                "Install SuiteSparse and reconfigure with "
                "-DPULSIM_ENABLE_KLU=ON (default).");
        case Backend::Eigen:
        case Backend::Auto:
        default:
            return std::make_unique<SparseLuSolver>();
    }
}
#endif  // !PULSIM_HAVE_KLU

}  // namespace pulsim::sparse

// -----------------------------------------------------------------------------
// Conditional KLU backend pull-in. Placed at the BOTTOM of the header so
// `DirectSolver` is fully declared before `klu_solver.hpp` inherits from it.
// `klu_solver.hpp` provides the KLU-aware `make_default_solver(n, hint)`
// impl when PULSIM_HAVE_KLU is set; the include-guards make the symbol
// ODR-safe regardless of which header the user pulls in first.
// -----------------------------------------------------------------------------
#ifdef PULSIM_HAVE_KLU
#include "pulsim/sparse/klu_solver.hpp"
#endif
