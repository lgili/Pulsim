#pragma once

// =============================================================================
// Pulsim — Path-Based Event-Driven (PED) engine: PI step-size controller.
// =============================================================================
//
// Söderlind, *Automatic Control and Adaptive Time-Stepping*,
// Numerical Algorithms 31 (2002) §4.
//
// Gate 2 of the `add-path-based-dsed-engine` OpenSpec proposal.
// Direct port of the Gate 1 Python prototype (`prototype/dsed/
// step_controller.py`); validated to zero rejections on buck CCM
// at $rtol=10^{-6}, atol=10^{-9}$.
//
// PI step update:
//   h_{n+1} = h_n · clamp(safety · (1/err_n)^kP · (err_{n-1}/err_n)^kI,
//                          rho_min, rho_max)
//
// Defaults are tuned for DOPRI5 (order 5) per Söderlind Table 4:
//   kP = 0.7, kI = 0.3. For BDF2 use kP=0.4, kI=0.3 (Table 5 H211b).

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::dsed {

/// PI step-size controller for adaptive ODE integration.
///
/// Owns mutable state across steps (`err_prev_`, `reject_streak_`).
/// Reset after a topology change to clear I-term wind-up.
class PIController {
public:
    /// Construct with custom or default gains. Defaults are
    /// Söderlind-2002 Table 4 values for order-5 methods (DOPRI5).
    explicit PIController(Real rtol = Real{1e-6},
                          Real atol = Real{1e-9},
                          Real kP = Real{0.7},
                          Real kI = Real{0.3},
                          Real rho_min = Real{0.1},
                          Real rho_max = Real{5.0},
                          Real safety = Real{0.9},
                          int max_rejects = 5) noexcept
        : rtol_{rtol}, atol_{atol},
          kP_{kP}, kI_{kI},
          rho_min_{rho_min}, rho_max_{rho_max},
          safety_{safety},
          max_rejects_{max_rejects} {}

    /// Weighted RMS error norm.
    ///
    /// $sc_i = atol + rtol \cdot \max(|x^{old}_i|, |x^{new}_i|)$;
    /// $\|err\| = \sqrt{(1/n) \sum_i (err_i / sc_i)^2}$.
    ///
    /// With this normalisation, acceptance is `||err|| <= 1`.
    [[nodiscard]] Real err_norm(const Vector& err,
                                  const Vector& x_old,
                                  const Vector& x_new) const noexcept {
        const auto n = err.size();
        Real s2 = Real{0};
        for (Eigen::Index i = 0; i < n; ++i) {
            const Real sc = atol_ + rtol_ * std::max(std::abs(x_old(i)),
                                                       std::abs(x_new(i)));
            const Real r = err(i) / sc;
            s2 += r * r;
        }
        return std::sqrt(s2 / static_cast<Real>(n));
    }

    /// Decide whether to accept the proposed step + compute the next h.
    ///
    /// On accept: returns ``(true, h_next)``; updates internal
    /// PI memory (`err_prev_`, `reject_streak_` reset).
    /// On reject: returns ``(false, h_smaller)``; caller should
    /// retry the SAME ``t`` with the smaller ``h``. After
    /// ``max_rejects`` consecutive rejections, throws
    /// ``std::runtime_error``.
    [[nodiscard]] std::pair<bool, Real> accept(const Vector& err,
                                                  const Vector& x_old,
                                                  const Vector& x_new,
                                                  Real h) {
        const Real e = err_norm(err, x_old, x_new);

        if (e <= Real{1}) {
            const Real e_safe = std::max(e, Real{1e-12});
            const Real ratio = std::pow(Real{1} / e_safe, kP_)
                              * std::pow(err_prev_ / e_safe, kI_);
            const Real clamped = std::clamp(ratio, rho_min_, rho_max_);
            const Real h_next = safety_ * clamped * h;
            // Wind-up protection: clamp err_prev to [0.1, 1.0]
            err_prev_ = std::clamp(e > Real{0} ? e : Real{0.1},
                                    Real{0.1}, Real{1.0});
            reject_streak_ = 0;
            return {true, h_next};
        }

        // Reject — shrink h, retry same t
        ++reject_streak_;
        if (reject_streak_ > max_rejects_) {
            throw std::runtime_error(
                std::string{"PIController: "} +
                std::to_string(reject_streak_) +
                " consecutive rejections (last err=" +
                std::to_string(e) + ", h=" + std::to_string(h) + ")");
        }
        // Pure-I shrink (no kI on rejection to avoid wind-up).
        // Formula: h_new = safety · (1/e)^kP · h  — Hairer & Wanner
        // Solving ODE I, §II.4 eq. (4.13). Bug fix vs earlier ``/safety``:
        // for e only slightly > 1 the divide-by-safety inflated the
        // ratio above 1, clamped to 1, and h NEVER shrank → infinite
        // reject loop until max_rejects fired. Surfaced in Gate 4.D
        // auto-dispatch transitions where mode changes can push err
        // just above 1.
        const Real ratio = safety_ * std::pow(Real{1} / e, kP_);
        const Real clamped = std::clamp(ratio, rho_min_, Real{1.0});
        return {false, clamped * h};
    }

    /// Reset PI memory. Call on topology change to clear I-term wind-up.
    void reset() noexcept {
        err_prev_ = Real{1};
        reject_streak_ = 0;
    }

    // --- Read-only accessors for diagnostics ---
    [[nodiscard]] Real err_prev() const noexcept { return err_prev_; }
    [[nodiscard]] int reject_streak() const noexcept { return reject_streak_; }

private:
    Real rtol_;
    Real atol_;
    Real kP_;
    Real kI_;
    Real rho_min_;
    Real rho_max_;
    Real safety_;
    int max_rejects_;
    Real err_prev_ = Real{1};
    int reject_streak_ = 0;
};

}  // namespace pulsim::dsed
