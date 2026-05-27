// SPDX-License-Identifier: MIT
//
// Pulsim — Phase 2.4 / C++ port: Dormand-Prince 5(4) embedded RK.
//
// Mirrors `python/pulsim/integrators.py::DormandPrince5`.
//
// 7-stage FSAL (First-Same-As-Last) explicit Runge-Kutta with embedded
// 4th-order error estimate. Per accepted step the controller computes:
//
//   err_norm = || (x_new − x_emb) / (atol + rtol·max(|x|, |x_new|)) ||_RMS
//   dt_new = dt · safety · err_norm^(−1/5), clamped to [dt_min, dt_max]
//            and to the growth/shrink factors.
//
// This C++ implementation is a standalone ODE solver. It is NOT (yet)
// integrated into `run_transient`'s MNA loop — that would require
// refactoring the kernel's fixed-step companion-model path. For pure
// ODE problems (van der Pol, stiff thermal RC, etc.) the API matches
// the Python solver exactly.
//
// Reference: Hairer, Nørsett & Wanner, "Solving Ordinary Differential
// Equations I: Nonstiff Problems" §II.4.

#pragma once

#include "pulsim/numeric/types.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pulsim::integrators {

/// Output of `DormandPrince5::solve(...)` — full accepted-step trajectory.
struct AdaptiveSolution {
    std::vector<Real> t;                // accepted step times
    std::vector<Eigen::VectorXd> x;     // states at each accepted step
    std::vector<Real> dt;               // accepted step sizes
    Index n_accepted = 0;
    Index n_rejected = 0;
    Index n_f_evals = 0;
};


/// Dormand-Prince 5(4) adaptive RK solver.
///
/// Template parameter `F` is any callable with signature
/// `Eigen::VectorXd f(Real t, const Eigen::VectorXd& x)`.
template <typename F>
class DormandPrince5 {
public:
    F f;
    Real rtol = Real{1e-5};
    Real atol = Real{1e-8};
    Real dt_min = Real{1e-12};
    Real dt_max = std::numeric_limits<Real>::infinity();
    Real safety = Real{0.9};
    Real growth_max = Real{5.0};
    Real shrink_min = Real{0.2};

    explicit DormandPrince5(F f_in) noexcept : f(std::move(f_in)) {}

    /// Integrate from t_span[0] to t_span[1] starting at x0.
    /// Throws std::runtime_error if the step controller demands a step
    /// smaller than `dt_min`.
    AdaptiveSolution solve(Real t0, Real t_end,
                                const Eigen::VectorXd& x0,
                                Real dt_init = -Real{1}) const {
        if (t_end <= t0) {
            throw std::invalid_argument(
                "DormandPrince5: t_end must be > t_start");
        }
        Eigen::VectorXd x = x0;
        const Index n = x.size();

        // Hairer & Wanner §II.4 initial-step heuristic when not given.
        Real dt;
        if (dt_init <= Real{0}) {
            Eigen::VectorXd f0 = f(t0, x);
            const Real d0 = (x.cwiseQuotient(
                (Eigen::VectorXd::Constant(n, atol)
                 + rtol * x.cwiseAbs()))).norm();
            const Real d1 = (f0.cwiseQuotient(
                (Eigen::VectorXd::Constant(n, atol)
                 + rtol * x.cwiseAbs()))).norm();
            if (d0 < Real{1e-5} || d1 < Real{1e-5}) {
                dt = Real{1e-6};
            } else {
                dt = Real{0.01} * d0 / d1;
            }
            dt = std::min(dt, t_end - t0);
        } else {
            dt = dt_init;
        }

        AdaptiveSolution sol;
        sol.t.push_back(t0);
        sol.x.push_back(x);

        Real t = t0;
        // FSAL stage cache: k[0..6].
        std::array<Eigen::VectorXd, 7> k{};
        for (auto& ki : k) {
            ki.setZero(n);
        }
        k[0] = f(t, x);
        sol.n_f_evals += 1;

        const Index max_steps =
            10 * static_cast<Index>(std::max<Real>(
                Real{1}, (t_end - t0) / std::max(dt, Real{1e-30})));

        for (Index step = 0; step < max_steps && t < t_end; ++step) {
            // Don't overshoot the end point.
            if (t + dt > t_end) {
                dt = t_end - t;
            }

            // Stages 2..7. Reusable temp.
            Eigen::VectorXd xi(n);
            for (int i = 1; i < 7; ++i) {
                xi = x;
                for (int j = 0; j < i; ++j) {
                    const Real aij = A_(i, j);
                    if (aij != Real{0}) {
                        xi.noalias() += (dt * aij) * k[j];
                    }
                }
                k[i] = f(t + C_(i) * dt, xi);
                sol.n_f_evals += 1;
            }

            // 5th-order solution.
            Eigen::VectorXd x_new = x;
            for (int j = 0; j < 7; ++j) {
                if (B5_(j) != Real{0}) {
                    x_new.noalias() += (dt * B5_(j)) * k[j];
                }
            }

            // Error vector = dt · Σ E_j · k_j (E = B5 − B4).
            Eigen::VectorXd err = Eigen::VectorXd::Zero(n);
            for (int j = 0; j < 7; ++j) {
                if (E_(j) != Real{0}) {
                    err.noalias() += (dt * E_(j)) * k[j];
                }
            }
            // RMS-normalised error: ||err / scale||_RMS.
            Real err_norm_sq = Real{0};
            for (Index i = 0; i < n; ++i) {
                const Real scale = atol +
                    rtol * std::max(std::abs(x[i]), std::abs(x_new[i]));
                const Real ei = err[i] / scale;
                err_norm_sq += ei * ei;
            }
            const Real err_norm =
                std::sqrt(err_norm_sq / static_cast<Real>(n));

            if (err_norm <= Real{1}) {
                // Accept.
                t = t + dt;
                x = x_new;
                k[0] = k[6];      // FSAL
                sol.n_accepted += 1;
                sol.t.push_back(t);
                sol.x.push_back(x);
                sol.dt.push_back(dt);

                Real factor = (err_norm > Real{0})
                    ? safety * std::pow(err_norm, Real{-0.2})
                    : growth_max;
                factor = std::min(growth_max, std::max(shrink_min, factor));
                dt = std::min(dt_max, std::max(dt_min, dt * factor));
            } else {
                // Reject — shrink and retry.
                sol.n_rejected += 1;
                const Real factor = std::max(
                    shrink_min, safety * std::pow(err_norm, Real{-0.2}));
                dt = dt * factor;
                if (dt < dt_min) {
                    throw std::runtime_error(
                        "DormandPrince5: step size collapsed below dt_min");
                }
            }
        }
        if (t < t_end) {
            throw std::runtime_error(
                "DormandPrince5: max_steps exceeded");
        }
        return sol;
    }

private:
    // Butcher tableau — Dormand-Prince 5(4).
    static constexpr Real C_(int i) noexcept {
        constexpr Real c[7] = {
            Real{0},       Real{1} / Real{5},  Real{3} / Real{10},
            Real{4} / Real{5}, Real{8} / Real{9}, Real{1}, Real{1}};
        return c[i];
    }
    static constexpr Real A_(int i, int j) noexcept {
        // i ∈ [0, 6], j < i. Returns 0 for j >= i.
        if (j >= i) return Real{0};
        switch (i) {
            case 1:
                return (j == 0) ? Real{1} / Real{5} : Real{0};
            case 2:
                if (j == 0) return Real{3} / Real{40};
                if (j == 1) return Real{9} / Real{40};
                return Real{0};
            case 3:
                if (j == 0) return Real{44} / Real{45};
                if (j == 1) return -Real{56} / Real{15};
                if (j == 2) return Real{32} / Real{9};
                return Real{0};
            case 4:
                if (j == 0) return Real{19372} / Real{6561};
                if (j == 1) return -Real{25360} / Real{2187};
                if (j == 2) return Real{64448} / Real{6561};
                if (j == 3) return -Real{212} / Real{729};
                return Real{0};
            case 5:
                if (j == 0) return Real{9017} / Real{3168};
                if (j == 1) return -Real{355} / Real{33};
                if (j == 2) return Real{46732} / Real{5247};
                if (j == 3) return Real{49} / Real{176};
                if (j == 4) return -Real{5103} / Real{18656};
                return Real{0};
            case 6:
                if (j == 0) return Real{35} / Real{384};
                if (j == 1) return Real{0};
                if (j == 2) return Real{500} / Real{1113};
                if (j == 3) return Real{125} / Real{192};
                if (j == 4) return -Real{2187} / Real{6784};
                if (j == 5) return Real{11} / Real{84};
                return Real{0};
            default:
                return Real{0};
        }
    }
    // B5 == A[6, :] (FSAL property: 5th-order solution = last row of A).
    static constexpr Real B5_(int j) noexcept { return A_(6, j); }
    // B4 — embedded 4th-order weights.
    static constexpr Real B4_(int j) noexcept {
        constexpr Real b4[7] = {
            Real{5179}    / Real{57600},
            Real{0},
            Real{7571}    / Real{16695},
            Real{393}     / Real{640},
            -Real{92097}  / Real{339200},
            Real{187}     / Real{2100},
            Real{1}       / Real{40}};
        return b4[j];
    }
    // E = B5 − B4 (error coefficients).
    static constexpr Real E_(int j) noexcept {
        return B5_(j) - B4_(j);
    }
};

// Deduction guide so users can write `DormandPrince5 dp{lambda}` without
// spelling out the template parameter.
template <typename F>
DormandPrince5(F) -> DormandPrince5<F>;

}  // namespace pulsim::integrators
