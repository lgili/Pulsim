#pragma once

// =============================================================================
// Pulsim — Layer 0 V8: KluSolver
// =============================================================================
//
// `openspec/changes/add-pwl-rank1-update` Phase 2.
//
// `DirectSolver` implementation wrapping SuiteSparse KLU (Davis & Natarajan,
// *ACM TOMS* 37(3), 2010, Algorithm 907). Purpose-built for circuit MNA
// matrices: BTF-ordered, COLAMD-permuted, asymmetric sparse LU with O(nnz)
// triangular solves.
//
// Compiled into the build only when `PULSIM_HAVE_KLU` is set by the root
// CMakeLists.txt's `find_package(KLU CONFIG)` block. When KLU is absent the
// rest of Pulsim builds and runs identically using `SparseLuSolver`.
//
// V8 partial-refactor MVP — `partial_refactor(...)` ignores the `changed_cols`
// hint and runs `klu_refactor` (full numeric refactor with the cached symbolic
// ordering). That already wins over `Eigen::SparseLU::factorize()` because it
// reuses the COLAMD ordering, but the asymptotic win documented in the
// proposal (O(path) vs O(nnz·log n)) comes only after the V8.1 follow-up that
// adds path-based re-elimination per Chen et al., IEEE TPEL 2024 §III.
// The MVP unblocks the rest of the proposal (cache fast-path, telemetry,
// benchmarks) without that follow-up.
//
// KLU's C API takes non-const `int*` / `double*` arrays for the matrix
// triplet. The functions DO NOT mutate those arrays; the non-const is a C
// convention. `const_cast`s in this header are intentional and safe.

#ifdef PULSIM_HAVE_KLU

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"

extern "C" {
#include <klu.h>
}

#include <span>
#include <stdexcept>
#include <utility>

namespace pulsim::sparse {

class KluSolver final : public DirectSolver {
public:
    KluSolver() noexcept {
        // Populate `common_` with KLU's defaults (BTF on, COLAMD ordering,
        // SCALE_SUM, partial-pivot tolerance 0.001, etc.). Returns 1 on
        // success; failure here is essentially impossible (no allocation,
        // just struct init), but we ignore the return value rather than
        // throw from a noexcept ctor.
        klu_defaults(&common_);
    }

    ~KluSolver() override {
        free_factors();
    }

    KluSolver(const KluSolver&) = delete;
    KluSolver& operator=(const KluSolver&) = delete;
    KluSolver(KluSolver&&) = delete;
    KluSolver& operator=(KluSolver&&) = delete;

    [[nodiscard]] bool analyze(const Matrix& M) override {
        free_factors();
        if (M.rows() != M.cols() || M.rows() == 0) {
            return false;
        }
        // Eigen's `SparseMatrix<Real, ColMajor, int32>` stores CSC natively:
        // outerIndexPtr() = column pointers (length n+1),
        // innerIndexPtr() = row indices (length nnz). KLU expects exactly
        // that triplet.
        const auto n = static_cast<int>(M.rows());
        symbolic_ = klu_analyze(
            n,
            const_cast<int*>(M.outerIndexPtr()),
            const_cast<int*>(M.innerIndexPtr()),
            &common_);
        analyzed_ = (symbolic_ != nullptr);
        factorized_ = false;
        return analyzed_;
    }

    [[nodiscard]] bool factorize(const Matrix& M) override {
        if (!analyzed_) {
            throw std::logic_error(
                "KluSolver::factorize called before analyze "
                "(or analyze returned false). Call analyze(M) first "
                "and check its return value.");
        }
        // klu_factor returns nullptr on numeric singularity (e.g. zero
        // pivot before partial pivoting can rescue). Free any stale
        // numeric factor before reusing the slot.
        if (numeric_) {
            klu_free_numeric(&numeric_, &common_);
            numeric_ = nullptr;
        }
        numeric_ = klu_factor(
            const_cast<int*>(M.outerIndexPtr()),
            const_cast<int*>(M.innerIndexPtr()),
            const_cast<double*>(M.valuePtr()),
            symbolic_, &common_);
        factorized_ = (numeric_ != nullptr);
        return factorized_;
    }

    void solve(const Vector& b, Vector& x) const override {
        if (!factorized_) {
            throw std::logic_error(
                "KluSolver::solve called before factorize "
                "(or factorize returned false). Call factorize(M) first "
                "and check its return value.");
        }
        // KLU's solve is in-place: it overwrites the RHS argument with the
        // solution. Copy b → x first so the caller's `b` survives.
        x = b;
        const auto n = static_cast<int>(symbolic_->n);
        klu_solve(symbolic_, numeric_,
                  n, /*nrhs=*/1,
                  x.data(),
                  const_cast<klu_common*>(&common_));
    }

    [[nodiscard]] bool is_analyzed()   const noexcept override { return analyzed_; }
    [[nodiscard]] bool is_factorized() const noexcept override { return factorized_; }

    // -------------------------------------------------------------------------
    // Rank-1 / partial-refactor support — V8 MVP
    // -------------------------------------------------------------------------

    [[nodiscard]] bool supports_partial_refactor() const noexcept override {
        return true;
    }

    /// V8 MVP: ignore `changed_cols`, run `klu_refactor` (full numeric
    /// refactor with cached symbolic ordering). Still ~2-3× faster than
    /// `Eigen::SparseLU::factorize()` because the COLAMD ordering is
    /// reused. V8.1 will replace this with path-based re-elimination per
    /// Chen et al. 2024 to push the speedup to O(path) per single-bit
    /// switch flip.
    [[nodiscard]] bool partial_refactor(
        const Matrix& new_M,
        [[maybe_unused]] std::span<const Index> changed_cols) override {
        if (!analyzed_) {
            return false;  // caller must analyze first
        }
        if (!numeric_) {
            // No prior factor to refactor over — fall through to a full
            // factor on the same symbolic.
            return factorize(new_M);
        }
        const int ok = klu_refactor(
            const_cast<int*>(new_M.outerIndexPtr()),
            const_cast<int*>(new_M.innerIndexPtr()),
            const_cast<double*>(new_M.valuePtr()),
            symbolic_, numeric_, &common_);
        // klu_refactor returns 1 on success, 0 on failure (singularity).
        // On failure, the existing numeric factor remains valid for the
        // PREVIOUS matrix — invariant the caller relies on for fallback.
        return ok != 0;
    }

private:
    void free_factors() noexcept {
        if (numeric_) {
            klu_free_numeric(&numeric_, &common_);
            numeric_ = nullptr;
        }
        if (symbolic_) {
            klu_free_symbolic(&symbolic_, &common_);
            symbolic_ = nullptr;
        }
        analyzed_   = false;
        factorized_ = false;
    }

    klu_common    common_{};
    klu_symbolic* symbolic_ = nullptr;
    klu_numeric*  numeric_  = nullptr;
    bool          analyzed_   = false;
    bool          factorized_ = false;
};

// -----------------------------------------------------------------------------
// KLU-aware factory implementation.
//
// Honours the `Backend` hint declared in `solver.hpp`. ODR-safe because
// solver.hpp only inlines its own fallback impl when PULSIM_HAVE_KLU is
// NOT defined — exactly one definition of this overload exists per build.
// -----------------------------------------------------------------------------
inline std::unique_ptr<DirectSolver> make_default_solver(Size n, Backend hint) {
    switch (hint) {
        case Backend::Eigen:
            return std::make_unique<SparseLuSolver>();
        case Backend::KLU:
            return std::make_unique<KluSolver>();
        case Backend::Auto:
        default:
            if (n >= static_cast<Size>(PULSIM_KLU_AUTO_THRESHOLD)) {
                return std::make_unique<KluSolver>();
            }
            return std::make_unique<SparseLuSolver>();
    }
}

}  // namespace pulsim::sparse

#endif  // PULSIM_HAVE_KLU
