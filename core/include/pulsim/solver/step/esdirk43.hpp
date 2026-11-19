#pragma once

// add-high-order-stiff-integrator (design D1–D3, DECISIONS 1–6): the
// fourth-order ESDIRK as a stage table of the unified stepper.
//
// ESDIRK4(3)6L[2]SA, Kennedy & Carpenter (Appl. Numer. Math. 44, 2003),
// the implicit part of ARK4(3)6L[2]SA; the same rationals as SUNDIALS'
// ARKODE_ARK436L2SA_DIRK_6_3_4. Six stages, the first explicit, γ = 1/4
// on the diagonal of the other five; stiffly accurate, L-stable, stage
// order 2, embedded third-order weights b̂ (R̂(−∞) = −3/20, so the
// embedded difference is filtered before it is used: `error_increment`).
//
// The skeleton sees the five implicit stages (s = 0 … 4 ↔ tableau stages
// 2 … 6). Each is a backward-Euler solve of size γh on the trapezoidal
// factor of h/2 (G + C/(γh) = G + 2C/(h/2)): one factor per step. Stage s
// starts from the pseudo previous state
//
//   ψ_s = q(x0) + explicit_weight(s)·h·q̇0 + Σ_{j<s} weight(s, j)·(q(Y_j) − ψ_j)
//
// (the Sdirk22 form plus the explicit stage's term), with q̇0 = −f(x0, t0)
// the derivative at the step start: the trapezoidal history of the
// stateful devices at x0, evaluated, never solved for. Everything is in
// q (charge / flux) space, so no algebraic unknown ever needs a
// derivative and a step needs no history beyond (x0, q̇0).

#include <array>
#include <cstddef>

#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/step/integrators.hpp"

namespace pulsim::step {

struct Esdirk43 {
    static constexpr const char* name = "esdirk43";
    static constexpr int order = 4;
    /// Implicit stages (the explicit first stage costs no solve).
    static constexpr int n_stages = 5;
    /// The embedded difference is O(h^4) (b̂ is third order).
    static constexpr Real err_exponent = Real{-0.25};
    static constexpr Real gamma = Real{0.25};

    /// Tableau entry a_ij, 0-based over all six stages (i, j < 6). Each is
    /// num/den of two integers below 2^53: the division is correctly
    /// rounded, so these are the nearest doubles to the exact rationals.
    [[nodiscard]] static constexpr Real a(int i, int j) noexcept {
        if (j == i && i > 0) return gamma;
        if (j >= i) return Real{0};
        constexpr Real t[6][5] = {
            {0, 0, 0, 0, 0},
            {Real(1) / 4, 0, 0, 0, 0},
            {Real(8611) / 62500, Real(-1743) / 31250, 0, 0, 0},
            {Real(5012029) / 34652500, Real(-654441) / 2922500,
             Real(174375) / 388108, 0, 0},
            {Real(15267082809.0) / Real(155376265600.0), Real(-71443401) / 120774400,
             Real(730878875) / 902184768, Real(2285395) / 8070912, 0},
            {Real(82889) / 524892, 0, Real(15625) / 83664, Real(69875) / 102672,
             Real(-2260) / 8211}};
        return t[i][j];
    }
    /// Main weights: the last row (stiffly accurate).
    [[nodiscard]] static constexpr Real b(int j) noexcept { return a(5, j); }
    /// Embedded third-order weights.
    [[nodiscard]] static constexpr Real b_hat(int j) noexcept {
        constexpr Real t[6] = {Real(4586570599.0) / Real(29645900160.0), 0,
                               Real(178811875) / 945068544,
                               Real(814220225) / 1159782912,
                               Real(-3700637) / 11593932, Real(61727) / 225920};
        return t[j];
    }
    /// Stage abscissae, exact decimal fractions.
    [[nodiscard]] static constexpr Real c(int i) noexcept {
        constexpr Real t[6] = {0, Real(1) / 2, Real(83) / 250, Real(31) / 50,
                               Real(17) / 20, 1};
        return t[i];
    }

    /// Implicit stage s: backward Euler of γh on the factor of h/2; it ends
    /// at c_{s+1}·h (the last one on h itself).
    [[nodiscard]] static StageSpec stage(Real h, int s) noexcept {
        const Real gh = gamma * h;
        return {pwl::TrBdf2Stage::BackwardEuler, Real{2} * gh, gh,
                s == n_stages - 1 ? h : c(s + 1) * h};
    }
    /// Weight of the increment (q(Y_j) − ψ_j) = γ·h·q̇_j in ψ_s.
    [[nodiscard]] static constexpr Real weight(int s, int j) noexcept {
        return j < s ? a(s + 1, j + 1) / gamma : Real{0};
    }
    /// Weight of h·q̇0 (the explicit stage) in ψ_s.
    [[nodiscard]] static constexpr Real explicit_weight(int s) noexcept {
        return a(s + 1, 0);
    }
    /// The embedded difference h·Σ (b_i − b̂_i)·q̇_i, as weights of h·q̇0
    /// and of the increments. Unfiltered it is unbounded relative to the
    /// error on stiff modes; filter it with one back-solve on the stage
    /// matrix: (G + C/(γh))·ẽ = e/(γh), then take the norm of ẽ.
    [[nodiscard]] static constexpr Real err_explicit_weight() noexcept {
        return b(0) - b_hat(0);
    }
    [[nodiscard]] static constexpr Real err_weight(int j) noexcept {
        return (b(j + 1) - b_hat(j + 1)) / gamma;
    }

    /// Dense output (DECISIONS 3): v(t0 + θh) = v0 + Σ_s w_s(θ)·(V_s − v0)
    /// over stage VALUES (no stage derivative). The continuous weights
    /// b(θ) = Σ w_s(θ)·a_{s+1} meet all eight order-4 conditions for every
    /// θ (probes/esdirk43_dense.py, exact); w(0) = 0, w(1) = e_4.
    /// Order 4 (local 5) holds for what the method integrates: q (charge,
    /// flux) on non-stiff modes. Apply it to q, then re-solve x from q and
    /// the algebraic equations at t0 + θh (the consistent solve); for a
    /// linear q, combining x and projecting is the same. Applied raw to an
    /// algebraic unknown, to x of a nonlinear q, or to a mode much faster
    /// than h, it is local order 3 only (Σ w·c³ ≠ θ³: stage order 2): use
    /// it to bracket an event, then land the event by re-steps.
    [[nodiscard]] static constexpr Real dense_weight(int s, Real theta) noexcept {
        constexpr Real p[5][4] = {
            {Real(-103) / 27, Real(101477) / 3507, Real(-474928) / 10521,
             Real(631898) / 31563},
            {Real(1445875) / 47061, Real(-21408125) / 124749,
             Real(628142000) / 2619729, Real(-777175250) / 7859187},
            {Real(-723370) / 57753, Real(73126550) / 1071639,
             Real(-266912960) / 3214917, Real(263402720) / 9644751},
            {Real(-7232) / 483, Real(49286080) / 457079, Real(-34308608) / 195891,
             Real(112833664) / 1371237},
            {Real(66) / 5, Real(-15273) / 167, Real(119664) / 835,
             Real(-53486) / 835}};
        return theta * (p[s][0] + theta * (p[s][1] + theta * (p[s][2] + theta * p[s][3])));
    }

    // ---- combinations a body needs (V: any vector with + and scalar *) --

    /// ψ_s from q(x0), h·q̇0 and the increments d_j = q(Y_j) − ψ_j, j < s.
    template <class V, class D>
    [[nodiscard]] static V psi(int s, const V& q0, const V& hqd0, const D& d) {
        V r = q0 + explicit_weight(s) * hqd0;
        for (int j = 0; j < s; ++j) r = r + weight(s, j) * d[static_cast<std::size_t>(j)];
        return r;
    }
    /// The unfiltered embedded difference (q units) from h·q̇0 and all
    /// five increments.
    template <class V, class D>
    [[nodiscard]] static V error_increment(const V& hqd0, const D& d) {
        V r = err_explicit_weight() * hqd0;
        for (int j = 0; j < n_stages; ++j) r = r + err_weight(j) * d[static_cast<std::size_t>(j)];
        return r;
    }
    /// v(t0 + θh) from v0 and the five stage values (see dense_weight).
    template <class V, class Y>
    [[nodiscard]] static V dense(Real theta, const V& x0, const Y& y) {
        V r = x0;
        for (int s = 0; s < n_stages; ++s)
            r = r + dense_weight(s, theta) * (y[static_cast<std::size_t>(s)] - x0);
        return r;
    }
};

}  // namespace pulsim::step
