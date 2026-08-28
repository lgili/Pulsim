#pragma once

// =============================================================================
// Pulsim — PED engine: stiffness detection (Gate 4 dispatch).
// =============================================================================
//
// Direct port of `prototype/dsed/stiffness_detector.py`. Selects
// the right integrator (DOPRI5 or BDF2) per mode based on the
// product |λ_max| · h:
//
//   |λ_max| = largest-magnitude eigenvalue of A_mask (cached per mode)
//   h       = current PI-controller recommended step
//   stiff   when |λ_max| · h > threshold (default 10.0)
//
// DOPRI5's stability region on the real axis is roughly h·|λ| < 3
// (boundary of the order-5 Butcher tableau), so a ratio of 10 means
// the explicit method would be forced to take ~10 sub-steps per
// accuracy-required step. Switching to BDF2 (A(α)-stable) removes
// the constraint at the cost of order-2 vs order-5 accuracy.
//
// Eigenvalue computation: ``Eigen::EigenSolver<DenseMatrix>::eigenvalues()``
// — O(n³). For our converters n is small (≤ 30 for MMC N=3), so
// this is negligible; for larger systems (Gate 5 Krylov-Φ work)
// a power-iteration approximation would be cheaper.

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <unordered_map>

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::dsed {

/// Which integrator the scheduler should use for the current mode.
enum class IntegratorChoice {
    DOPRI5,      // explicit, non-stiff
    BDF2,        // implicit, stiff
};

/// Pick the right integrator per mode based on |λ_max| · h.
class StiffnessDetector {
public:
    explicit StiffnessDetector(Real threshold = Real{10.0})
        : threshold_{threshold} {}

    /// Compute (and cache) the largest-magnitude eigenvalue of A.
    [[nodiscard]] Real lambda_max(int mode_id, const DenseMatrix& A) {
        auto it = cache_.find(mode_id);
        if (it != cache_.end()) {
            return it->second;
        }
        const Real lam_max = estimate_lambda_max_(A);
        cache_.emplace(mode_id, lam_max);
        return lam_max;
    }

    /// |λ|max estimate. Small systems keep the exact dense
    /// eigensolve; larger ones use POWER ITERATION (v2.0 Phase 3
    /// item 4 / audit E.4) — the dispatch decision only needs the
    /// magnitude to within a few percent against a threshold, and
    /// the full O(n³) Schur factorization was costing more than the
    /// steps it was routing. Two-step norm ratios converge on the
    /// dominant MODULUS even when the dominant eigenvalue is a
    /// complex pair (the single-step ratio oscillates; the
    /// two-step one is the pair's |λ|² up to phase, so its square
    /// root settles).
    [[nodiscard]] static Real estimate_lambda_max_(
        const DenseMatrix& A) {
        const auto n = A.rows();
        if (n == 0) {
            return Real{0};
        }
        if (n <= 64) {
            Eigen::EigenSolver<DenseMatrix> es(A);
            const auto eigs = es.eigenvalues();
            Real lam = Real{0};
            for (Eigen::Index i = 0; i < eigs.size(); ++i) {
                lam = std::max(lam, std::abs(eigs[i]));
            }
            return lam;
        }
        // Deterministic quasi-random start (no Date/random in the
        // kernel): a fixed mixing of index bits, then normalise.
        Vector v(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            v[i] = Real{1} + Real{0.37} *
                   static_cast<Real>((i * 2654435761u) & 0xFFFF) /
                   Real{65536};
        }
        v.normalize();
        Real est = Real{0};
        Vector w1, w2;
        for (int iter = 0; iter < 40; ++iter) {
            w1 = A * v;
            w2 = A * w1;
            const Real n2 = w2.norm();
            const Real n0 = v.norm();
            if (!(n2 > Real{0}) || !std::isfinite(n2)) {
                return est;      // nilpotent-ish or overflow: best so far
            }
            const Real next = std::sqrt(n2 / n0);
            const bool settled =
                iter > 4 && std::abs(next - est) <=
                                Real{0.02} * std::abs(next);
            est = next;
            v = w2 / n2;
            if (settled) {
                break;
            }
        }
        return est;
    }

    /// Return DOPRI5 or BDF2 for the given (mode, h).
    [[nodiscard]] IntegratorChoice select(int mode_id,
                                            const DenseMatrix& A,
                                            Real h) {
        const Real lam = lambda_max(mode_id, A);
        return (lam * h > threshold_)
                ? IntegratorChoice::BDF2
                : IntegratorChoice::DOPRI5;
    }

    /// Diagnostic — all relevant numbers for the choice.
    struct Diagnostic {
        int mode_id;
        Real lambda_max;
        Real h;
        Real lambda_h;          // = lambda_max * h
        Real threshold;
        bool stiff;
        IntegratorChoice choice;
    };

    [[nodiscard]] Diagnostic explain(int mode_id,
                                       const DenseMatrix& A,
                                       Real h) {
        const Real lam = lambda_max(mode_id, A);
        const Real ratio = lam * h;
        const bool stiff = ratio > threshold_;
        return Diagnostic{
            .mode_id = mode_id,
            .lambda_max = lam,
            .h = h,
            .lambda_h = ratio,
            .threshold = threshold_,
            .stiff = stiff,
            .choice = stiff ? IntegratorChoice::BDF2
                            : IntegratorChoice::DOPRI5,
        };
    }

    /// Drop all cached eigenvalues (call after parameter changes).
    void clear_cache() noexcept { cache_.clear(); }

    [[nodiscard]] Real threshold() const noexcept { return threshold_; }
    [[nodiscard]] std::size_t cache_size() const noexcept {
        return cache_.size();
    }

private:
    Real threshold_;
    std::unordered_map<int, Real> cache_;
};

}  // namespace pulsim::dsed
