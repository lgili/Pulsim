// SPDX-License-Identifier: MIT
//
// Pulsim — Phase 2.4 / C++ port: Radau IIA(3) implicit RK.
//
// Mirrors `python/pulsim/integrators.py::RadauIIA3`.
//
// 2-stage, order-3, L-stable implicit Runge-Kutta. Each accepted step
// requires solving a 2n×2n implicit block system via Newton iteration:
//
//   [I − dt·a11·J,  -dt·a12·J ] [δK1]   [-r1]
//   [-dt·a21·J,    I − dt·a22·J] [δK2] = [-r2]
//
// where r_i = K_i − f(t + c_i·dt, x + dt·(a_i1·K1 + a_i2·K2)) and
// J = ∂f/∂x evaluated at (t, x).
//
// Error estimate via **step doubling** (one full step vs two half
// steps). More expensive than the embedded Hairer-Wanner estimate but
// robust and simple — matches the Python prototype.
//
// L-stability means the method is unconditionally stable for stiff
// linear test problems; for the Van der Pol oscillator at μ=10⁴ it
// completes in O(10) steps where explicit RK methods need O(10⁵).
//
// Reference: Hairer & Wanner, "Solving Ordinary Differential Equations
// II: Stiff and Differential-Algebraic Problems" §IV.5.

#pragma once

#include "pulsim/integrators/dormand_prince5.hpp"  // for AdaptiveSolution
#include "pulsim/numeric/types.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace pulsim::integrators {

/// Radau IIA(3) implicit RK solver.
///
/// Template parameter `F` is any callable
/// `Eigen::VectorXd f(Real t, const Eigen::VectorXd& x)`.
template <typename F>
class RadauIIA3 {
public:
    F f;
    // Optional analytical Jacobian (set to nullptr → use finite-diff).
    std::function<Eigen::MatrixXd(Real, const Eigen::VectorXd&)> jac{};
    Real rtol = Real{1e-5};
    Real atol = Real{1e-8};
    Real dt_min = Real{1e-12};
    Real dt_max = std::numeric_limits<Real>::infinity();
    Real safety = Real{0.9};
    Real growth_max = Real{5.0};
    Real shrink_min = Real{0.2};
    int newton_max_iter = 8;
    Real newton_tol = Real{1e-8};

    explicit RadauIIA3(F f_in) noexcept : f(std::move(f_in)) {}

    /// Integrate from t0 to t_end starting at x0.
    AdaptiveSolution solve(Real t0, Real t_end,
                                const Eigen::VectorXd& x0,
                                Real dt_init = -Real{1}) const {
        if (t_end <= t0) {
            throw std::invalid_argument(
                "RadauIIA3: t_end must be > t_start");
        }
        Eigen::VectorXd x = x0;

        Real dt;
        if (dt_init <= Real{0}) {
            Eigen::VectorXd f0 = f(t0, x);
            const Real f_mag = f0.norm() + Real{1e-30};
            dt = std::max(dt_min,
                            std::min(dt_max, Real{0.01} / f_mag));
        } else {
            dt = dt_init;
        }
        dt = std::min(dt, t_end - t0);

        AdaptiveSolution sol;
        sol.t.push_back(t0);
        sol.x.push_back(x);

        Real t = t0;
        const Index max_steps =
            10 * static_cast<Index>(std::max<Real>(
                Real{1}, (t_end - t0) / std::max(dt, Real{1e-30})));

        for (Index step = 0; step < max_steps && t < t_end; ++step) {
            if (t + dt > t_end) {
                dt = t_end - t;
            }

            // One full step.
            Eigen::VectorXd x_full;
            int it_full;
            std::tie(x_full, it_full) = solve_stage_block_(t, x, dt);
            sol.n_f_evals += 2 * std::max(1, std::abs(it_full)) + 1;
            if (it_full < 0) {
                sol.n_rejected += 1;
                dt = std::max(dt_min, shrink_min * dt);
                continue;
            }
            // Two half steps for step-doubling error estimate.
            Eigen::VectorXd x_half1;
            int it_h1;
            std::tie(x_half1, it_h1) = solve_stage_block_(t, x, dt * Real{0.5});
            sol.n_f_evals += 2 * std::max(1, std::abs(it_h1)) + 1;
            if (it_h1 < 0) {
                sol.n_rejected += 1;
                dt = std::max(dt_min, shrink_min * dt);
                continue;
            }
            Eigen::VectorXd x_half2;
            int it_h2;
            std::tie(x_half2, it_h2) = solve_stage_block_(
                t + dt * Real{0.5}, x_half1, dt * Real{0.5});
            sol.n_f_evals += 2 * std::max(1, std::abs(it_h2)) + 1;
            if (it_h2 < 0) {
                sol.n_rejected += 1;
                dt = std::max(dt_min, shrink_min * dt);
                continue;
            }

            // Error: ||x_full − x_half|| / scale; order-3 leading-error
            // → Richardson factor 1/(2^3 − 1) = 1/7.
            Real err_norm_sq = Real{0};
            const Index n = x.size();
            for (Index i = 0; i < n; ++i) {
                const Real scale = atol +
                    rtol * std::max(std::abs(x[i]),
                                     std::max(std::abs(x_full[i]),
                                              std::abs(x_half2[i])));
                const Real diff = (x_full[i] - x_half2[i]) / Real{7};
                err_norm_sq += (diff / scale) * (diff / scale);
            }
            const Real err_norm =
                std::sqrt(err_norm_sq / static_cast<Real>(n));

            if (err_norm <= Real{1}) {
                // Accept (use the more accurate half-step result).
                t = t + dt;
                x = x_half2;
                sol.n_accepted += 1;
                sol.t.push_back(t);
                sol.x.push_back(x);
                sol.dt.push_back(dt);

                Real factor = (err_norm > Real{0})
                    ? safety * std::pow(err_norm, Real{-1.0 / 3.0})
                    : growth_max;
                factor = std::min(growth_max, std::max(shrink_min, factor));
                dt = std::min(dt_max, std::max(dt_min, dt * factor));
            } else {
                sol.n_rejected += 1;
                const Real factor = std::max(
                    shrink_min,
                    safety * std::pow(err_norm, Real{-1.0 / 3.0}));
                dt = dt * factor;
                if (dt < dt_min) {
                    throw std::runtime_error(
                        "RadauIIA3: step size collapsed below dt_min");
                }
            }
        }
        if (t < t_end) {
            throw std::runtime_error(
                "RadauIIA3: max_steps exceeded");
        }
        return sol;
    }

private:
    // Butcher tableau (Radau IIA, s=2).
    static constexpr Real C0_() noexcept { return Real{1} / Real{3}; }
    static constexpr Real C1_() noexcept { return Real{1}; }
    static constexpr Real A00_() noexcept { return Real{5} / Real{12}; }
    static constexpr Real A01_() noexcept { return -Real{1} / Real{12}; }
    static constexpr Real A10_() noexcept { return Real{3} / Real{4}; }
    static constexpr Real A11_() noexcept { return Real{1} / Real{4}; }
    static constexpr Real B0_() noexcept { return Real{3} / Real{4}; }
    static constexpr Real B1_() noexcept { return Real{1} / Real{4}; }

    Eigen::MatrixXd finite_diff_jacobian_(
        Real t, const Eigen::VectorXd& x) const {
        const Index n = x.size();
        Eigen::MatrixXd J(n, n);
        const Real eps =
            std::sqrt(std::numeric_limits<Real>::epsilon());
        for (Index i = 0; i < n; ++i) {
            const Real h = eps * std::max(Real{1}, std::abs(x[i]));
            Eigen::VectorXd xp = x;
            xp[i] += h;
            Eigen::VectorXd xm = x;
            xm[i] -= h;
            Eigen::VectorXd fp = f(t, xp);
            Eigen::VectorXd fm = f(t, xm);
            J.col(i) = (fp - fm) / (Real{2} * h);
        }
        return J;
    }

    /// Solve the 2-stage Newton block for one step. Returns
    /// `(x_new, iter_count)`; iter_count < 0 means Newton diverged.
    std::pair<Eigen::VectorXd, int> solve_stage_block_(
        Real t, const Eigen::VectorXd& x, Real dt) const {
        const Index n = x.size();
        // Initial guess: K_0 = K_1 = f(t, x).
        Eigen::VectorXd f0 = f(t, x);
        Eigen::VectorXd K0 = f0;
        Eigen::VectorXd K1 = f0;
        // Compute J once at (t, x).
        Eigen::MatrixXd J = jac ? jac(t, x) : finite_diff_jacobian_(t, x);
        const Eigen::MatrixXd I_n = Eigen::MatrixXd::Identity(n, n);
        for (int it = 0; it < newton_max_iter; ++it) {
            const Eigen::VectorXd x1 =
                x + dt * (A00_() * K0 + A01_() * K1);
            const Eigen::VectorXd x2 =
                x + dt * (A10_() * K0 + A11_() * K1);
            const Eigen::VectorXd f1 = f(t + C0_() * dt, x1);
            const Eigen::VectorXd f2 = f(t + C1_() * dt, x2);
            const Eigen::VectorXd r1 = K0 - f1;
            const Eigen::VectorXd r2 = K1 - f2;
            const Real err = std::max(r1.norm(), r2.norm());
            const Real K_norm = std::max(K0.norm(), K1.norm());
            if (err < newton_tol * std::max(Real{1}, K_norm)) {
                return {x + dt * (B0_() * K0 + B1_() * K1), it + 1};
            }
            // Build the 2n × 2n block matrix.
            Eigen::MatrixXd M(2 * n, 2 * n);
            M.topLeftCorner(n, n) = I_n - dt * A00_() * J;
            M.topRightCorner(n, n) = -dt * A01_() * J;
            M.bottomLeftCorner(n, n) = -dt * A10_() * J;
            M.bottomRightCorner(n, n) = I_n - dt * A11_() * J;
            Eigen::VectorXd rhs(2 * n);
            rhs.head(n) = r1;
            rhs.tail(n) = r2;
            Eigen::PartialPivLU<Eigen::MatrixXd> lu(M);
            // Singularity check (PartialPivLU doesn't throw — examine
            // determinant via U-diagonal sign).
            if (!lu.matrixLU().diagonal().array().abs().minCoeff()) {
                return {x + dt * (B0_() * K0 + B1_() * K1), -1};
            }
            Eigen::VectorXd delta = lu.solve(rhs);
            K0 -= delta.head(n);
            K1 -= delta.tail(n);
        }
        return {x + dt * (B0_() * K0 + B1_() * K1), -1};
    }
};

template <typename F>
RadauIIA3(F) -> RadauIIA3<F>;

}  // namespace pulsim::integrators
