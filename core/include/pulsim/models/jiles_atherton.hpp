#pragma once

// =============================================================================
// Pulsim — Jiles-Atherton hysteresis: parameters, Langevin helpers, and
// the in-loop core evaluator (Phase 4 C.4, "JA inside the loop")
// =============================================================================
//
// THE MODEL. Magnetisation M splits into an irreversible and a
// reversible part around the anhysteretic curve
//
//     M_an = M_s · L(H_e / a),   H_e = H + α·M,   L(x) = coth x − 1/x
//
//     dM_irr/dH = (M_an − M) / (k·δ − α·(M_an − M)),   δ = sign(dH)
//     dM_rev/dH = c · dM_an/dH_e / (1 − c·α·dM_an/dH_e)
//     dM/dH     = (1 − c)·dM_irr/dH + dM_rev/dH,      B = μ₀ (H + M)
//
// with the standard Jiles modification that removes the model's
// unphysical negative susceptibility: when (M_an − M)·δ < 0 the
// irreversible term is zero. That guard is what makes dM/dH ≥ 0
// and i(H) strictly increasing, which is what the inversion below
// rests on.
//
// THE DEVICE. A winding of N turns on a core (area Ae, path le,
// total gap lg). Ampère with the flux continuous through the gap:
//
//     N·i = H·le + (B/μ₀)·lg = H·le + (H + M)·lg
//     λ   = N·Ae·B = N·Ae·μ₀·(H + M)
//
// Given the step-start state (H_n, M_n), the trial current of a
// Newton iterate fixes H through the monotone map i(H) — M(H) being
// the JA ODE integrated from H_n to H in the direction sign(H − H_n)
// — and then λ, and the exact
//
//     L = dλ/di = N·Ae·μ₀·(1 + dM/dH) / [ (le + lg·(1 + dM/dH)) / N ].
//
// This is the whole difference from the observer it replaces: the
// magnetisation is part of the SAME solve as the current, at the
// same time level, with its exact tangent in the Jacobian. The
// observer injected ψ·dM/dt from the PREVIOUS step into a dummy
// source — an explicit treatment of a stiff inductive term, unstable
// for q = L_M/(dt(R + 2L₀/dt)) above ~0.5 (NaN measured at q = 40),
// and with its sign inverted besides.
//
// WITHIN ONE STEP THE FIELD IS MONOTONE. The ODE is integrated from
// H_n to H in one direction, so a step cannot contain a reversal; a
// peak that falls inside a step is reached and left on the same
// branch, an O(h) error localised to the peak that the step
// controller's flux-error estimate sees. The committed state after
// the step is (H, M) at the converged point.

#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>
#include <string>

namespace pulsim::models {

/// Five-parameter Jiles-Atherton set. Defaults are NOT physical —
/// always set them from a material catalog or a fit.
struct JilesAthertonParams {
    Real Ms    = Real{4.0e5};   //!< saturation magnetisation [A/m]
    Real a     = Real{50.0};    //!< anhysteretic shape parameter [A/m]
    Real alpha = Real{5e-5};    //!< mean-field coupling (dimensionless)
    Real c     = Real{0.20};    //!< reversibility (0..1)
    Real k     = Real{30.0};    //!< pinning coefficient [A/m]
};

namespace ja_detail {

inline constexpr Real MU_0 = Real{4} * std::numbers::pi_v<Real> * Real{1e-7};

/// Langevin L(x) = coth x − 1/x, stable at 0 and at large |x|.
[[nodiscard]] inline Real langevin(Real x) noexcept {
    const Real ax = std::abs(x);
    if (ax < Real{1e-4}) return x * (Real{1.0 / 3.0} - x * x * Real{1.0 / 45.0});
    if (ax > Real{30.0}) return ((x > Real{0}) ? Real{1} : Real{-1}) - Real{1} / x;
    return std::cosh(x) / std::sinh(x) - Real{1} / x;
}
[[nodiscard]] inline Real langevin_deriv(Real x) noexcept {
    const Real ax = std::abs(x);
    if (ax < Real{1e-4}) return Real{1.0 / 3.0} - x * x * Real{1.0 / 15.0};
    if (ax > Real{30.0}) return Real{1} / (x * x);
    const Real s = std::sinh(x);
    return Real{1} / (x * x) - Real{1} / (s * s);
}

/// dM/dH at (H, M) moving in direction δ, with the negative-
/// susceptibility guard.
[[nodiscard]] inline Real dM_dH(const JilesAthertonParams& p, Real H, Real M,
                                Real delta) noexcept {
    const Real He = H + p.alpha * M;
    const Real x = (p.a > Real{0}) ? He / p.a : Real{0};
    const Real M_an = p.Ms * langevin(x);
    const Real dMan_dHe = (p.a > Real{0}) ? (p.Ms / p.a) * langevin_deriv(x) : Real{0};
    const Real diff = M_an - M;
    Real dM_irr = Real{0};
    // Jiles' guard: the irreversible component only moves M TOWARD
    // the anhysteretic curve.
    if (diff * delta > Real{0}) {
        const Real denom = p.k * delta - p.alpha * diff;
        const Real floor = std::max(Real{1e-9}, Real{1e-6} * p.k);
        if (std::abs(denom) >= floor) dM_irr = diff / denom;
        if (dM_irr < Real{0}) dM_irr = Real{0};
    }
    const Real rev_den = Real{1} - p.c * p.alpha * dMan_dHe;
    const Real dM_rev = (std::abs(rev_den) > Real{1e-12})
                            ? p.c * dMan_dHe / rev_den
                            : p.c * dMan_dHe;
    const Real out = (Real{1} - p.c) * dM_irr + dM_rev;
    return out > Real{0} ? out : Real{0};
}

}  // namespace ja_detail

/// The in-loop hysteretic core.
struct JilesAthertonCore {
    struct Params {
        Real N  = Real{1};       //!< turns
        Real Ae = Real{1e-4};    //!< effective area [m²]
        Real le = Real{0.1};     //!< mean path in the core [m]
        Real lg = Real{0};       //!< TOTAL gap [m]
        JilesAthertonParams ja;
        //! MINIMUM number of RK4 sub-steps per evaluation. The count
        //! actually used is fixed FOR A STEP (see State::n_sub), never
        //! per evaluation:
        //!  * a count that adapts to the evaluation's own |ΔH| makes
        //!    λ(H) jump wherever it increments — a finite-difference
        //!    L across one such jump came out NEGATIVE (−0.018 H
        //!    against +0.0008 H);
        //!  * a count fixed for all time is smooth but inaccurate on
        //!    a large step, and the RK4 error then varies with the
        //!    target enough to make i(H) non-monotone (seen at eight
        //!    sub-steps over 30a).
        //! So the count is chosen once per step from the previous
        //! step's field excursion — sub-steps of a/4 — and every
        //! evaluation inside the step uses it. Smooth in H, and as
        //! accurate as the step is small.
        int substeps_min = 8;
        //! Sub-step length target as a fraction of `a`.
        Real substep_frac = Real{0.25};
        //! Initial (remanent) magnetisation at zero current [A/m] —
        //! the state a previous shutdown left, which is what drives
        //! inrush. |M0| ≤ Ms.
        Real M0 = Real{0};
    };

    /// Step-start state. `H` and `M` at the last committed point,
    /// and the sub-step count in force for the coming step.
    struct State {
        Real H = Real{0};
        Real M = Real{0};
        int  n_sub = 8;
        //! The branch direction IN FORCE for the coming step — the
        //! direction of the leg that brought the state here, +1
        //! initially. It is held FIXED for every evaluation within a
        //! step rather than re-derived from sign(H − H_base) per
        //! trial: the JA loop has a corner at a reversal (the
        //! ascending and descending branches have different slopes at
        //! the same point), and a Newton iterate that lands on that
        //! corner bounces between the two tangents — measured on a
        //! phase-shifted full bridge, whose current plateaus put the
        //! solution exactly at i_n every step: ||dx|| stuck at 5e-6 V
        //! against a 1e-9 tolerance, line search and LM no help. With
        //! δ fixed the law is C¹ inside the solve. A step that
        //! reverses is integrated with the old direction — the
        //! irreversible part then moves M along the wrong branch for
        //! that ONE step, an O(h) error at the loop tip that the next
        //! step (whose δ is updated at commit) does not repeat.
        Real delta_hint = Real{1};
    };

    /// The sub-step count for a step expected to span |ΔH|.
    [[nodiscard]] static int substeps_for(const Params& c, Real dH_span) noexcept {
        const Real target = std::max(c.substep_frac * c.ja.a, Real{1e-12});
        const Real n = std::ceil(std::abs(dH_span) / target);
        return std::clamp(static_cast<int>(std::min(n, Real{4096})), std::max(1, c.substeps_min), 4096);
    }

    /// Result of one evaluation at a trial current.
    struct Eval {
        Real H = Real{0};       //!< core field at the trial current
        Real M = Real{0};       //!< magnetisation there
        Real lambda = Real{0};  //!< N·Ae·μ₀·(H + M)
        Real L = Real{0};       //!< dλ/di, exact for this evaluation
        Real dM_dH = Real{0};   //!< at the end point
    };

    static void validate(const Params& c, const std::string& what) {
        auto bad = [&](const char* name, Real v, const char* why) {
            throw std::invalid_argument(std::format("{}: {} = {} — {}", what, name, v, why));
        };
        if (!(c.N > 0) || c.N != std::floor(c.N)) bad("N", c.N, "turns must be a positive integer");
        if (!(c.Ae > 0) || c.Ae > Real{1e-2}) bad("Ae", c.Ae, "core area must be in (0, 100 cm²] — mm² typed as m²?");
        if (!(c.le > 0) || c.le > Real{10}) bad("le", c.le, "path length must be in (0, 10 m] — mm typed as m?");
        if (!(c.lg >= 0) || c.lg > c.le) bad("lg", c.lg, "gap must be in [0, le]");
        if (!(c.ja.Ms > 0)) bad("Ms", c.ja.Ms, "saturation magnetisation must be > 0");
        if (!(c.ja.a > 0)) bad("a", c.ja.a, "anhysteretic shape parameter must be > 0");
        if (!(c.ja.k >= 0)) bad("k", c.ja.k, "pinning coefficient must be >= 0");
        if (!(c.ja.c >= 0) || c.ja.c > 1) bad("c", c.ja.c, "reversibility must be in [0, 1]");
        if (!(c.ja.alpha >= 0)) bad("alpha", c.ja.alpha, "mean-field coupling must be >= 0");
        // The initial anhysteretic susceptibility Ms/(3a − α Ms) must be
        // positive, or the material has negative slope at the origin.
        if (c.ja.alpha * c.ja.Ms >= Real{3} * c.ja.a) {
            bad("alpha", c.ja.alpha,
                "alpha·Ms >= 3a: the anhysteretic slope at the origin is "
                "negative — this parameter set is not a soft magnet");
        }
        if (c.substeps_min < 1 || c.substeps_min > 1024) bad("substeps_min", static_cast<Real>(c.substeps_min), "must be in [1, 1024]");
        if (!(std::abs(c.M0) <= c.ja.Ms)) bad("M0", c.M0, "|M0| must not exceed Ms");
        if (!(c.substep_frac > 0) || c.substep_frac > 1) bad("substep_frac", c.substep_frac, "must be in (0, 1]");
    }

    /// Integrate M from the state's (H, M) to `H_end` with a fixed
    /// number of RK4 sub-steps on dM/dH, in the direction
    /// sign(H_end − H). Returns M and, through `dMdH_end`, the slope
    /// at the end point — which is the exact derivative of the
    /// continuous solution with respect to H_end.
    [[nodiscard]] static Real integrate_M(const Params& c, const State& s, Real H_end,
                                          Real* dMdH_end = nullptr) noexcept {
        const Real dH = H_end - s.H;
        // Direction held fixed for the step (see State::delta_hint).
        const Real delta = s.delta_hint >= Real{0} ? Real{1} : Real{-1};
        const int n_sub = std::max(1, s.n_sub);
        const Real h = dH / static_cast<Real>(n_sub);
        Real H = s.H, M = s.M;
        if (dH != Real{0}) {
            for (int k = 0; k < n_sub; ++k) {
                const Real k1 = ja_detail::dM_dH(c.ja, H, M, delta);
                const Real k2 = ja_detail::dM_dH(c.ja, H + Real{0.5} * h, M + Real{0.5} * h * k1, delta);
                const Real k3 = ja_detail::dM_dH(c.ja, H + Real{0.5} * h, M + Real{0.5} * h * k2, delta);
                const Real k4 = ja_detail::dM_dH(c.ja, H + h, M + h * k3, delta);
                M += h * (k1 + Real{2} * k2 + Real{2} * k3 + k4) / Real{6};
                H += h;
                M = std::clamp(M, -c.ja.Ms, c.ja.Ms);
            }
        }
        if (dMdH_end) *dMdH_end = ja_detail::dM_dH(c.ja, H_end, M, delta);
        return M;
    }

    [[nodiscard]] static Real current_of(const Params& c, Real H, Real M) noexcept {
        return (H * c.le + (H + M) * c.lg) / c.N;
    }
    [[nodiscard]] static Real flux_of(const Params& c, Real H, Real M) noexcept {
        return c.N * c.Ae * ja_detail::MU_0 * (H + M);
    }
    /// Small-signal L for a given dM/dH.
    [[nodiscard]] static Real inductance_of(const Params& c, Real dMdH) noexcept {
        return c.N * c.N * c.Ae * ja_detail::MU_0 * (Real{1} + dMdH)
               / (c.le + c.lg * (Real{1} + dMdH));
    }

    /// Evaluate at a trial current: invert i(H) from the step-start
    /// state by a safeguarded Newton (bisection fallback), then λ and
    /// the exact L. Monotone because dM/dH ≥ 0.
    [[nodiscard]] static Eval evaluate(const Params& c, const State& s, Real i) noexcept {
        // Bracket. i(H) is increasing; L_air = N²μ₀Ae/(le+lg) is the
        // smallest slope, so H moves at most N·(i − i_n)/(le + lg)...
        // conservatively use di/dH ≥ le/N (the air-core core term).
        const Real i_n = current_of(c, s.H, s.M);
        const Real di = i - i_n;
        Real lo = s.H, hi = s.H;
        const Real span0 = std::abs(di) * c.N / std::max(c.le, Real{1e-12}) + Real{1e-9};
        if (di >= Real{0}) hi = s.H + span0; else lo = s.H - span0;
        // Expand the bracket if needed (M can push i the other way only
        // through the gap term, which is monotone too, so this is
        // conservative).
        auto f = [&](Real H, Real* M_out, Real* dMdH_out) {
            Real dm = 0;
            const Real M = integrate_M(c, s, H, &dm);
            if (M_out) *M_out = M;
            if (dMdH_out) *dMdH_out = dm;
            return current_of(c, H, M) - i;
        };
        Real M_lo = 0, M_hi = 0, d_lo = 0, d_hi = 0;
        Real f_lo = f(lo, &M_lo, &d_lo), f_hi = f(hi, &M_hi, &d_hi);
        for (int k = 0; k < 60 && f_lo * f_hi > Real{0}; ++k) {
            if (di >= Real{0}) { hi = s.H + (hi - s.H) * Real{2}; f_hi = f(hi, &M_hi, &d_hi); }
            else               { lo = s.H - (s.H - lo) * Real{2}; f_lo = f(lo, &M_lo, &d_lo); }
        }
        // Newton with bisection safeguard.
        Real H = (di >= Real{0}) ? hi : lo;
        Real M = (di >= Real{0}) ? M_hi : M_lo;
        Real dMdH = (di >= Real{0}) ? d_hi : d_lo;
        Real fH = (di >= Real{0}) ? f_hi : f_lo;
        const Real tol = Real{1e-13} * (std::abs(i) + std::abs(i_n) + Real{1e-9});
        for (int it = 0; it < 60; ++it) {
            if (std::abs(fH) <= tol) break;
            const Real di_dH = (c.le + c.lg * (Real{1} + dMdH)) / c.N;
            Real H_new = H - fH / di_dH;
            if (!(H_new > lo && H_new < hi)) H_new = Real{0.5} * (lo + hi);
            const Real f_new = f(H_new, &M, &dMdH);
            if (f_new * f_lo < Real{0}) { hi = H_new; f_hi = f_new; }
            else                        { lo = H_new; f_lo = f_new; }
            H = H_new; fH = f_new;
            if (hi - lo <= Real{1e-15} * (Real{1} + std::abs(H))) break;
        }
        Eval e;
        e.H = H; e.M = M; e.dM_dH = dMdH;
        e.lambda = flux_of(c, H, M);
        e.L = inductance_of(c, dMdH);
        return e;
    }
};

}  // namespace pulsim::models
