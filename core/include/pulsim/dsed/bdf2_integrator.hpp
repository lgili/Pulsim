#pragma once

// =============================================================================
// Pulsim — PED engine: BDF2 implicit integrator (Gate 4 stiff path).
// =============================================================================
//
// Direct port of `prototype/dsed/bdf2_integrator.py`. BDF2 is the
// A(α)-stable, order-2 implicit method used on stiff masks where
// DOPRI5's explicit step constraint (from |λ_max|) shrinks below
// what accuracy alone would require.
//
// Mathematical formulation (Hairer & Wanner *Solving ODE II* §V.1):
//
//   y_{n+1} - (4/3) y_n + (1/3) y_{n-1} = (2h/3) f(t_{n+1}, y_{n+1})
//
// For our LTI-per-mode case f(t, y) = A·y + b(t):
//
//   (I - (2h/3) A) y_{n+1} = (4/3) y_n - (1/3) y_{n-1} + (2h/3) b(t_{n+1})
//
// The system matrix J = (I - (2h/3) A) is independent of t and y.
// Factor it ONCE at mask change or h change; re-use the factor
// across many steps. (This is where the Pulsim path-based
// partial_refactor pays off at the C++ binding level — a future
// Gate 4.B/4.C wiring task once the Pulsim cache C++ surface is
// thread-safe-enough for the PED outer loop.)
//
// Bootstrap (first step, no history): Crank-Nicolson (trapezoidal
// rule, order-2 A-stable) — matches BDF2's order so the startup
// error doesn't contaminate global accuracy:
//
//   (I - (h/2) A) y_1 = (I + (h/2) A) y_0 + h · b(t_0 + h/2)
//
// Validated against an explicit DOPRI5 ground truth on the
// stiff RLC scenario (`prototype/dsed/run_stiff_rlc_validation.py`):
//
//   RMSE(v_C, BDF2 h=5µs vs DOPRI5 h=50ns) = 0.14 % V_in
//   Convergence ratio (h halving) = 3.72×  (theoretical 4× for order-2)
//   Speedup vs DOPRI5-stability-limit = 6.55× wall-clock

#include <Eigen/Dense>
#include <cstddef>
#include <functional>
#include <optional>

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"

namespace pulsim::dsed {

/// Persistent state between accepted BDF2 steps.
///
/// BDF2 is a 2-step method: it needs y_{n-1} and y_n to advance
/// to y_{n+1}. The cached LU factor of J = I - (2h/3) A is also
/// stored — only re-factored when h or mask changes.
struct BDF2State {
    std::optional<Vector> y_prev;                                    // y_{n-1}
    Real y_prev_t = Real{0};                                          // t_{n-1}
    Real h_cached = Real{0};                                          // h at which J was factored
    std::optional<Eigen::FullPivLU<DenseMatrix>> lu;                 // LU of J = I - (2h/3) A
    int mode_id = -1;                                                 // identifier for cached mask

    [[nodiscard]] bool has_history() const noexcept {
        return y_prev.has_value();
    }

    void invalidate() noexcept {
        y_prev.reset();
        lu.reset();
        h_cached = Real{0};
    }
};

/// Take one BDF2 step on the LTI system y' = A·y + b(t).
///
/// On the first call (no history): falls back to a single
/// Crank-Nicolson step to obtain y_1; the next call is then full
/// BDF2 with (y_{n-1} = y_0, y_n = y_1).
///
/// Returns {y_new, err} where err is the embedded order-2 error
/// estimate from y^{BDF2} − y^{BE-extrapolation}. The estimate is
/// returned as zeros on the bootstrap step.
template <class BFn>
[[nodiscard]] std::pair<Vector, Vector> bdf2_step(
    const DenseMatrix& A,
    BFn&& b_fn,
    Real t,
    const Vector& y,
    Real h,
    BDF2State& state) {

    // (Re-)factor J if h changed or first step
    if (!state.lu.has_value() || state.h_cached != h) {
        DenseMatrix J = DenseMatrix::Identity(A.rows(), A.cols())
                       - (Real{2} * h / Real{3}) * A;
        state.lu = Eigen::FullPivLU<DenseMatrix>(J);
        state.h_cached = h;
    }

    const Vector b_new = b_fn(t + h);

    if (!state.has_history()) {
        // Bootstrap: Crank-Nicolson (trapezoidal, order-2 A-stable)
        // (I - (h/2) A) y_1 = (I + (h/2) A) y_0 + h · b(t + h/2)
        const Vector b_old = b_fn(t);
        const Vector b_mid = Real{0.5} * (b_old + b_new);
        DenseMatrix J_CN = DenseMatrix::Identity(A.rows(), A.cols())
                          - Real{0.5} * h * A;
        DenseMatrix K_CN = DenseMatrix::Identity(A.rows(), A.cols())
                          + Real{0.5} * h * A;
        Eigen::FullPivLU<DenseMatrix> lu_CN(J_CN);
        Vector y_new = lu_CN.solve(K_CN * y + h * b_mid);
        // No embedded error from bootstrap → return zeros
        Vector err = Vector::Zero(y.size());
        state.y_prev = y;
        state.y_prev_t = t;
        return {std::move(y_new), std::move(err)};
    }

    // Full BDF2 step:
    // (I - (2h/3) A) y_new = (4/3) y - (1/3) y_prev + (2h/3) b(t+h)
    const Vector& y_prev = *state.y_prev;
    Vector rhs_bdf2 = (Real{4} / Real{3}) * y
                     - (Real{1} / Real{3}) * y_prev
                     + (Real{2} * h / Real{3}) * b_new;
    Vector y_new = state.lu->solve(rhs_bdf2);

    // Embedded error: (1/6) of the 3rd backward difference
    // y_new - 2·y + y_prev   ≈   h² · y'''(t_n)
    // → leading h² truncation coefficient is 1/6 of that.
    Vector err = (Real{1} / Real{6}) * (y_new - Real{2} * y + y_prev);

    state.y_prev = y;
    state.y_prev_t = t;

    return {std::move(y_new), std::move(err)};
}

/// PI step controller tuned for BDF2 (Söderlind 2002 Table 5, H211b).
///
/// Slower-reacting than DOPRI5's PIController (kP=0.7, kI=0.3) —
/// appropriate for an order-2 method where the per-step error has
/// more memory. Uses exponent 1/(p+1) = 1/3 (vs 1/6 for DOPRI5).
struct BDF2PIController {
    Real rtol = Real{1e-6};
    Real atol = Real{1e-9};
    Real kP = Real{0.4};
    Real kI = Real{0.3};
    Real rho_min = Real{0.1};
    Real rho_max = Real{3.0};      // tighter than DOPRI5's 5 (BDF2 more sensitive)
    Real safety = Real{0.85};
    int max_rejects = 5;

    Real err_prev = Real{1.0};
    int reject_streak = 0;

    [[nodiscard]] Real err_norm(
        const Vector& err,
        const Vector& y_old,
        const Vector& y_new) const {
        // sc_i = atol + rtol · max(|y_old_i|, |y_new_i|)
        // ||err|| = sqrt(mean((err_i / sc_i)²))
        const auto abs_old = y_old.array().abs();
        const auto abs_new = y_new.array().abs();
        const Eigen::ArrayXd max_abs = abs_old.max(abs_new);
        const Eigen::ArrayXd sc = atol + rtol * max_abs;
        const Eigen::ArrayXd scaled = err.array() / sc;
        return std::sqrt((scaled * scaled).mean());
    }

    [[nodiscard]] std::pair<bool, Real> accept(
        const Vector& err,
        const Vector& y_old,
        const Vector& y_new,
        Real h) {
        const Real e = err_norm(err, y_old, y_new);

        if (e <= Real{1.0}) {
            const Real e_clamped = std::max(e, Real{1e-12});
            // Order p = 2 for BDF2 → exponent = 1/(p+1) = 1/3
            const Real exp_p = kP / Real{3};
            const Real exp_i = kI / Real{3};
            Real ratio = std::pow(Real{1.0} / e_clamped, exp_p)
                        * std::pow(err_prev / e_clamped, exp_i);
            ratio = std::clamp(ratio, rho_min, rho_max);
            const Real h_next = safety * ratio * h;
            err_prev = std::clamp(e, Real{0.1}, Real{1.0});
            if (e <= Real{0}) err_prev = Real{0.1};
            reject_streak = 0;
            return {true, h_next};
        }

        ++reject_streak;
        if (reject_streak > max_rejects) {
            throw std::runtime_error(
                "BDF2 PI controller: max consecutive rejections exceeded");
        }
        const Real exp_p = kP / Real{3};
        // Pure-I shrink on rejection. Bug fix vs the earlier ``/ safety``:
        // for `e` only slightly > 1 the divide-by-safety inflated the ratio
        // above 1, clamped to 1.0, and h NEVER shrank → infinite reject loop
        // until max_rejects fired. `safety` (< 1) must MULTIPLY, matching the
        // accept path above and the identical fix in step_controller.hpp's
        // PIController (Söderlind 2002 / Hairer & Wanner, Solving ODE I §II.4).
        Real ratio = safety * std::pow(Real{1.0} / e, exp_p);
        ratio = std::clamp(ratio, rho_min, Real{1.0});
        return {false, ratio * h};
    }

    void reset() noexcept {
        err_prev = Real{1.0};
        reject_streak = 0;
    }
};

}  // namespace pulsim::dsed
