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
        Eigen::EigenSolver<DenseMatrix> es(A);
        const auto eigs = es.eigenvalues();
        Real lam_max = Real{0};
        for (Eigen::Index i = 0; i < eigs.size(); ++i) {
            const Real m = std::abs(eigs(i));
            if (m > lam_max) lam_max = m;
        }
        cache_[mode_id] = lam_max;
        return lam_max;
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
