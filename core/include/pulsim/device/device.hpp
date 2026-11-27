#pragma once

// =============================================================================
// Pulsim — the Newton device contract (add-unified-device-interface D2 / D3)
// =============================================================================
//
// Step 4: the interface. Since the dual path (DECISIONS #23) the engines
// stamp every ported kind through it when the run asks for
// `device::DevicePath::Interface` (laws in device/laws.hpp, sparse adapters
// in pwl/interface_stamps.hpp); the hand stamps stay the default until the
// release flips it.
//
// A device contributes f(x, p, t) + d/dt q(x, p) = 0 to its rows through ONE
// law, `eval`, written once as a template over the scalar:
//
//   * `eval<Real>`                     — the residual;
//   * `eval` over `ad::ADRealN<R>` x   — the Jacobian ∂f/∂x, ∂q/∂x (Newton);
//   * `eval` over AD parameters        — ∂f/∂p, ∂q/∂p (add-parameter-
//     sensitivities), the parameters seeded by `seed_params`.
//
// `Params<S>` is a template over the scalar for that reason: a struct of S
// only, standard layout, `n_params` of them, with a `map(f)` that rebuilds it
// over another scalar. `Params<Real>` is the plain parameter struct, so the
// Real path is the one an engine runs.
//
// The integrator, not the device, owns the time formula (`history_term`,
// `stamp_fq`): rows stay in conductance form (a·∂q/∂x, a = 2/h, c1/h), the
// step comes from the `StepCtx` argument of every solve and from nowhere
// else.

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Forced inlining of the per-instance stamps (task 15.7; see
// pwl/interface_stamps.hpp): left out of line they cost more than the law.
#if defined(__GNUC__) || defined(__clang__)
#define PULSIM_DEVICE_INLINE __attribute__((always_inline))
#else
#define PULSIM_DEVICE_INLINE
#endif

namespace pulsim::device {

/// Time formula of one solve; chosen by the integrator, never by a device.
enum class Formula : std::uint8_t { Dc, BackwardEuler, Trapezoidal, Bdf2Stage2 };

/// The step of one solve, passed by value. There is no other source of h.
struct StepCtx {
    Real t = Real{0};
    Formula formula = Formula::Trapezoidal;
    Real h = Real{0};     ///< the time this solve spans (0 for Dc)
};

/// TR-BDF2 second-stage weights, γ = 2 − √2 (same numbers as
/// pwl::trbdf2_coeffs()): d/dt q ≈ (c1·q + c2·q_γ + c3·q_n)/h.
struct Bdf2Weights { Real c1, c2, c3; };
[[nodiscard]] inline Bdf2Weights bdf2_weights() noexcept {
    const Real root2 = std::sqrt(Real{2});
    const Real gamma = Real{2} - root2;
    const Real rho = (Real{1} - gamma) / gamma;
    return {Real{2} + root2, -(Real{1} + rho) / (Real{1} - gamma),
            Real{1} / root2};
}

/// d/dt q ≈ a·q(x) + r: the coefficient a of the formula.
[[nodiscard]] inline Real formula_a(const StepCtx& c) noexcept {
    switch (c.formula) {
    case Formula::Dc:            return Real{0};
    case Formula::BackwardEuler: return Real{1} / c.h;
    case Formula::Trapezoidal:   return Real{2} / c.h;
    case Formula::Bdf2Stage2:    return bdf2_weights().c1 / c.h;
    }
    return Real{0};
}

/// The history part r of one q row, from the committed (q_n, q̇_n) and, in
/// the BDF2 stage, the stage scratch q_γ.
[[nodiscard]] inline Real history_term(const StepCtx& c, Real q_n,
                                       Real qdot_n, Real q_gamma) noexcept {
    switch (c.formula) {
    case Formula::Dc:            return Real{0};
    case Formula::BackwardEuler: return -q_n / c.h;
    case Formula::Trapezoidal:   return -(Real{2} / c.h) * q_n - qdot_n;
    case Formula::Bdf2Stage2: {
        const auto w = bdf2_weights();
        return (w.c2 * q_gamma + w.c3 * q_n) / c.h;
    }
    }
    return Real{0};
}

/// a·q + r of one PIN q row taken as an increment, in the hand stamps'
/// operation order (refresh_nonlinear_capacitors): a·(q − q_n) [− q̇_n], and
/// (c1·q + c2·q_γ + c3·q_n)/h in the BDF2 stage. a·q + r rounds relative to
/// a·q: an eps·Q/h residual floor that grows as h shrinks (F-Z-7: case A's
/// esdirk43 Newton at h_min, eps·Q/(γh) ~ 1e-6 A at 400 V and h ~ 1e-16 s),
/// where the increment form rounds relative to the step's own a·(q − q_n).
[[nodiscard]] inline Real formula_rate(const StepCtx& c, Real a, Real q, Real q_n,
                                       Real qdot_n, Real q_gamma) noexcept {
    switch (c.formula) {
    case Formula::Dc:            return Real{0};
    case Formula::BackwardEuler: return a * (q - q_n);
    case Formula::Trapezoidal:   return a * (q - q_n) - qdot_n;
    case Formula::Bdf2Stage2: {
        const auto w = bdf2_weights();
        return (w.c1 * q + w.c2 * q_gamma + w.c3 * q_n) / c.h;
    }
    }
    return Real{0};
}

/// `history_term` for every row at once, the formula switch taken once (the
/// same operations per row, so the same values).
template <std::size_t R>
[[nodiscard]] inline std::array<Real, R> history_terms(
    const StepCtx& c, const std::array<Real, R>& q_n,
    const std::array<Real, R>& qdot_n, const std::array<Real, R>& q_gamma) noexcept {
    std::array<Real, R> r{};
    switch (c.formula) {
    case Formula::Dc: break;
    case Formula::BackwardEuler:
        for (std::size_t k = 0; k < R; ++k) r[k] = -q_n[k] / c.h;
        break;
    case Formula::Trapezoidal: {
        const Real t = Real{2} / c.h;
        for (std::size_t k = 0; k < R; ++k) r[k] = -t * q_n[k] - qdot_n[k];
        break;
    }
    case Formula::Bdf2Stage2: {
        const auto w = bdf2_weights();
        for (std::size_t k = 0; k < R; ++k) r[k] = (w.c2 * q_gamma[k] + w.c3 * q_n[k]) / c.h;
        break;
    }
    }
    return r;
}

/// One evaluation: f and q per row (pins first, then internal rows).
template <class S, std::size_t R>
struct Out {
    std::array<S, R> f{};
    std::array<S, R> q{};
};

/// A law's LOCAL rows (`n_local`, default 0): device-owned unknowns the
/// engines never see (the eliminated stored charge of the Lauritzen diode
/// and of the IGBT tail), numbered after the internal rows. `stamp_fq_local`
/// condenses them out; the committed value lives in the instance's history.
template <class D>
inline constexpr std::size_t local_v = [] {
    if constexpr (requires { D::n_local; }) return static_cast<std::size_t>(D::n_local);
    else return std::size_t{0};
}();

/// Rows the law evaluates: pins, internal rows, local rows.
template <class D>
inline constexpr std::size_t rows_v = D::n_pins + D::n_internal + local_v<D>;

/// Rows the engine stamps: pins and internal rows (the global unknowns).
template <class D>
inline constexpr std::size_t global_rows_v = D::n_pins + D::n_internal;

/// Per-instance data a law reads but never differentiates: the law
/// variant, a datasheet table, the Jiles-Atherton step-start state (D2's
/// `State`, read-only during the Newton iterations). A law without one
/// omits `Aux` and writes `eval(p, x, o)`; with one, `eval(aux, p, x, o)`.
struct NoAux {};
template <class D> struct aux_of { using type = NoAux; };
template <class D>
    requires requires { typename D::Aux; }
struct aux_of<D> { using type = typename D::Aux; };
template <class D> using aux_t = typename aux_of<D>::type;

template <class D, class P, class X, class O>
concept EvalWith =
    requires(const aux_t<D>& a, const P& p, const X& x, O& o) {
        D::eval(a, p, x, o);
    } ||
    requires(const P& p, const X& x, O& o) { D::eval(p, x, o); };

/// Call a law with or without its `Aux`.
template <class D, class P, class X, class O>
inline void eval(const aux_t<D>& a, const P& p, const X& x, O& o) {
    if constexpr (requires { D::eval(a, p, x, o); }) {
        D::eval(a, p, x, o);
    } else {
        D::eval(p, x, o);
    }
}

/// The Newton device contract. `eval` must compile for the three scalar
/// combinations an engine and the sensitivity solve use.
template <class D>
concept Device =
    requires {
        { D::n_pins } -> std::convertible_to<std::size_t>;
        { D::n_internal } -> std::convertible_to<std::size_t>;
        { D::n_params } -> std::convertible_to<std::size_t>;
        typename D::template Params<Real>;
    } &&
    std::is_standard_layout_v<typename D::template Params<Real>> &&
    std::is_trivially_copyable_v<typename D::template Params<Real>> &&
    (D::n_params == 0  // an empty struct still has size 1
     || sizeof(typename D::template Params<Real>)
            == D::n_params * sizeof(Real)) &&
    // residual
    EvalWith<D, typename D::template Params<Real>,
             std::array<Real, rows_v<D>>, Out<Real, rows_v<D>>> &&
    // Jacobian in x
    EvalWith<D, typename D::template Params<Real>,
             std::array<ad::ADRealN<rows_v<D>>, rows_v<D>>,
             Out<ad::ADRealN<rows_v<D>>, rows_v<D>>> &&
    // derivative in the parameters
    EvalWith<D, typename D::template Params<ad::ADRealN<D::n_params>>,
             std::array<Real, rows_v<D>>,
             Out<ad::ADRealN<D::n_params>, rows_v<D>>>;

/// Parameters over the AD scalar with slot `Offset + k` seeded for the
/// k-th parameter: the sensitivity solve's input. `k` counts `map`'s calls
/// of `f`, so `map` must call it in declaration order (braced init does;
/// a function call's argument order is unspecified).
template <class D, std::size_t N, std::size_t Offset = 0>
[[nodiscard]] typename D::template Params<ad::ADRealN<N>> seed_params(
    const typename D::template Params<Real>& p) {
    static_assert(Offset + D::n_params <= N, "not enough AD slots");
    std::size_t k = 0;
    return p.map([&k](Real v) {
        std::array<Real, N> d{};
        d[Offset + k++] = Real{1};
        return ad::ADRealN<N>{v, d};
    });
}

/// A law may declare it has no `q` (`kHasQ = false`: algebraic) or no `f`
/// (`kHasF = false`: charge only); the stamp then skips that half (the
/// result is the same: the skipped terms are exact zeros).
template <class D>
inline constexpr bool has_q_v = [] {
    if constexpr (requires { D::kHasQ; }) return static_cast<bool>(D::kHasQ);
    else return true;
}();
template <class D>
inline constexpr bool has_f_v = [] {
    if constexpr (requires { D::kHasF; }) return static_cast<bool>(D::kHasF);
    else return true;
}();
/// A law whose `q` lives on internal rows only declares `kPinQ = false`
/// (the saturable inductor, the machines): its commit then takes no time
/// formula at all (q̇_n of a pin row is 0, of an internal row −f).
template <class D>
inline constexpr bool pin_q_v = [] {
    if constexpr (requires { D::kPinQ; }) return static_cast<bool>(D::kPinQ);
    else return true;
}();

namespace detail_fq {
/// The rest of `stamp_fq` over value / tangent accessors.
/// `WithJ = false`: J was written by the law (`eval_jac`), only F here.
template <class D, bool WithJ, class FV, class QV, class FD, class QD>
PULSIM_DEVICE_INLINE inline Real assemble(
    const StepCtx& c, const std::array<Real, rows_v<D>>& q_n,
    const std::array<Real, rows_v<D>>& qdot_n, const std::array<Real, rows_v<D>>& q_gamma,
    std::array<Real, rows_v<D>>& F, std::array<std::array<Real, rows_v<D>>, rows_v<D>>& J,
    const std::array<Real, rows_v<D>>* dq, const std::array<Real, rows_v<D>>* dq_gamma,
    FV fv, QV qv, FD fd, QD qd) {
    constexpr std::size_t R = rows_v<D>;
    constexpr bool hq = has_q_v<D>, hf = has_f_v<D>;
    if constexpr (!hq) {
        for (std::size_t r = 0; r < R; ++r) {
            F[r] = fv(r);
            if constexpr (WithJ)
                for (std::size_t k = 0; k < R; ++k) J[r][k] = fd(r, k);
        }
        Real worst = Real{0};
        for (std::size_t r = 0; r < R; ++r) worst = std::max(worst, std::abs(F[r]));
        return worst;
    }
    const Real a = formula_a(c);
    if constexpr (!hf) {   // charge only (the Coss): pin rows, increment form (dq unused)
        Real worst = Real{0};
        const bool bdf2 = c.formula == Formula::Bdf2Stage2;
        const Real c1 = bdf2_weights().c1;
        for (std::size_t r = 0; r < R; ++r) {
            F[r] = formula_rate(c, a, qv(r), q_n[r], qdot_n[r], q_gamma[r]);
            if constexpr (WithJ)   // the hand stamp's c1·C/h in the BDF2 stage
                for (std::size_t k = 0; k < R; ++k)
                    J[r][k] = bdf2 ? c1 * qd(r, k) / c.h : a * qd(r, k);
            worst = std::max(worst, std::abs(F[r]));
        }
        return worst;
    }
    const auto hist = dq == nullptr ? history_terms<R>(c, q_n, qdot_n, q_gamma)
                                    : std::array<Real, R>{};
    Real worst = Real{0};
    for (std::size_t r = 0; r < R; ++r) {
        if (dq == nullptr) {
            F[r] = fv(r) + a * qv(r) + hist[r];
        } else {
            Real hist = Real{0};
            if (c.formula == Formula::Trapezoidal) {
                hist = -qdot_n[r];
            } else if (c.formula == Formula::Bdf2Stage2) {
                hist = bdf2_weights().c2 * (*dq_gamma)[r] / c.h;
            }
            F[r] = fv(r) + a * (*dq)[r] + hist;
        }
        if constexpr (WithJ) {
            for (std::size_t k = 0; k < R; ++k) J[r][k] = fd(r, k) + a * qd(r, k);
        }
        worst = std::max(worst, std::abs(F[r]));
    }
    return worst;
}
}  // namespace detail_fq

/// One device's Newton contribution under `c`, dense: F[r] = f + a·q + r
/// and J[r][k] = ∂F[r]/∂x[k], from one AD evaluation. `q_n`, `qdot_n`,
/// `q_gamma` are the instance's committed charges, their derivatives and
/// the BDF2 stage scratch, per row. Returns max |F|.
///
/// Increment form (`dq` given): the engine wants a·(q − q_n) taken as one
/// closed-form increment `dq` (round-off relative to the increment, not to
/// a·q; TR-BDF2 track 4), and `dq_gamma` = q(x_γ) − q_n in the BDF2 stage.
/// Since c1 + c2 + c3 = 0 the history is then −q̇_n (trapezoidal), 0
/// (backward Euler) or c2·dq_γ/h (BDF2). J is the same either way.
template <Device D>
PULSIM_DEVICE_INLINE inline Real stamp_fq(const aux_t<D>& aux,
              const typename D::template Params<Real>& p,
              const std::array<Real, rows_v<D>>& x, const StepCtx& c,
              const std::array<Real, rows_v<D>>& q_n,
              const std::array<Real, rows_v<D>>& qdot_n,
              const std::array<Real, rows_v<D>>& q_gamma,
              std::array<Real, rows_v<D>>& F,
              std::array<std::array<Real, rows_v<D>>, rows_v<D>>& J,
              const std::array<Real, rows_v<D>>* dq = nullptr,
              const std::array<Real, rows_v<D>>* dq_gamma = nullptr) {
    constexpr std::size_t R = rows_v<D>;
    using Mat = std::array<std::array<Real, R>, R>;
    // A law may write its Jacobian by hand (`eval_jac(aux, p, x, a, o, J)`:
    // the Real values in `o` and the nonzero entries of J = ∂f/∂x + a·∂q/∂x
    // as fd + a·qd of the AD tangents; the induction machine, whose AD width
    // 10 cost half its stamp, DECISIONS #48).
    if constexpr (requires(Out<Real, R>& o, Mat& m) { D::eval_jac(aux, p, x, Real{}, o, m); }) {
        static_assert(has_q_v<D> && has_f_v<D>);
        Out<Real, R> o;
        J = Mat{};
        D::eval_jac(aux, p, x, formula_a(c), o, J);
        const auto none = [](std::size_t, std::size_t) { return Real{0}; };
        return detail_fq::assemble<D, false>(
            c, q_n, qdot_n, q_gamma, F, J, dq, dq_gamma, [&](std::size_t r) { return o.f[r]; },
            [&](std::size_t r) { return o.q[r]; }, none, none);
    } else {
        std::array<ad::ADRealN<R>, R> xs{};
        for (std::size_t k = 0; k < R; ++k) {
            std::array<Real, R> d{};
            d[k] = Real{1};
            xs[k] = ad::ADRealN<R>{x[k], d};
        }
        Out<ad::ADRealN<R>, R> o;
        device::eval<D>(aux, p, xs, o);
        return detail_fq::assemble<D, true>(
            c, q_n, qdot_n, q_gamma, F, J, dq, dq_gamma,
            [&](std::size_t r) { return o.f[r].value(); },
            [&](std::size_t r) { return o.q[r].value(); },
            [&](std::size_t r, std::size_t k) { return o.f[r].deriv(k); },
            [&](std::size_t r, std::size_t k) { return o.q[r].deriv(k); });
    }
}


template <Device D>
Real stamp_fq(const typename D::template Params<Real>& p,
              const std::array<Real, rows_v<D>>& x, const StepCtx& c,
              const std::array<Real, rows_v<D>>& q_n,
              const std::array<Real, rows_v<D>>& qdot_n,
              const std::array<Real, rows_v<D>>& q_gamma,
              std::array<Real, rows_v<D>>& F,
              std::array<std::array<Real, rows_v<D>>, rows_v<D>>& J) {
    return stamp_fq<D>(aux_t<D>{}, p, x, c, q_n, qdot_n, q_gamma, F, J);
}

/// `stamp_fq` with the local rows condensed out (static condensation). The
/// local rows and the global rows must be AFFINE in the local unknowns
/// (`D::kLocalAffine`), so ONE Newton step from any start solves them to
/// rounding: s = s₀ − J_LL⁻¹·F_L. Returns F and J over the global rows at
/// the solved s, J = J_GG − J_GL·J_LL⁻¹·J_LG (the Schur complement), and
/// writes the solved local values into `x[global..]` (the commit evaluates
/// there). The start s₀ is the caller's `x[global..]` (the committed q_n,
/// so the step is the increment of this solve). Shortcut: one local row
/// (the two kinds that have one); a block solve is the upgrade for more.
template <Device D>
PULSIM_DEVICE_INLINE inline Real stamp_fq_local(const aux_t<D>& aux, const typename D::template Params<Real>& p,
                    std::array<Real, rows_v<D>>& x, const StepCtx& c,
                    const std::array<Real, rows_v<D>>& q_n,
                    const std::array<Real, rows_v<D>>& qdot_n,
                    const std::array<Real, rows_v<D>>& q_gamma,
                    std::array<Real, global_rows_v<D>>& F,
                    std::array<std::array<Real, global_rows_v<D>>, global_rows_v<D>>& J) {
    static_assert(local_v<D> == 1, "stamp_fq_local: one local row (see the shortcut)");
    static_assert(D::kLocalAffine, "stamp_fq_local: the local rows must be affine "
                                   "in the local unknowns (one step solves them)");
    constexpr std::size_t R = rows_v<D>, G = global_rows_v<D>, L = G;
    std::array<Real, R> Ff{};
    std::array<std::array<Real, R>, R> Jf{};
    (void)stamp_fq<D>(aux, p, x, c, q_n, qdot_n, q_gamma, Ff, Jf);
    const Real ds = -Ff[L] / Jf[L][L];
    x[L] += ds;
    Real worst = Real{0};
    for (std::size_t r = 0; r < G; ++r) {
        const Real w = Jf[r][L] / Jf[L][L];
        F[r] = Ff[r] + Jf[r][L] * ds;
        for (std::size_t k = 0; k < G; ++k) J[r][k] = Jf[r][k] - w * Jf[L][k];
        worst = std::max(worst, std::abs(F[r]));
    }
    return worst;
}

/// Commit after an accepted step: q_n ← q(x) and q̇_n ← the derivative the
/// step produced. On a PIN row that is the formula's rate in increment
/// form (formula_rate; (q(x) − q_n)/h on backward Euler, as the hand commit) (the row
/// is a shared KCL, so f alone is not the derivative; physical, so a
/// variable h stays legal). On an INTERNAL row the device owns the whole
/// equation f + q̇ = 0, so q̇_n ← −f(x) exactly, not the formula value,
/// which differs from it by the Newton residual (a saturable inductor's
/// committed voltage is the solved v, as in every companion model).
/// Returns the Real evaluation at `x` (the capture of a stage point reads
/// it) — for a law with `eval_internal`, only its internal rows.
template <Device D>
PULSIM_DEVICE_INLINE inline Out<Real, rows_v<D>> commit_fq(const aux_t<D>& aux,
                               const typename D::template Params<Real>& p,
                               const std::array<Real, rows_v<D>>& x,
                               const StepCtx& c,
                               std::array<Real, rows_v<D>>& q_n,
                               std::array<Real, rows_v<D>>& qdot_n,
                               const std::array<Real, rows_v<D>>& q_gamma) {
    Out<Real, rows_v<D>> o;
    // A law with internal-only q may evaluate just those rows for the
    // commit (`eval_internal`; the pin rows of the returned Out are then
    // not meaningful — no caller reads them).
    if constexpr (!pin_q_v<D> && requires { D::eval_internal(aux, p, x, o); }) {
        D::eval_internal(aux, p, x, o);
    } else {
        device::eval<D>(aux, p, x, o);
    }
    if constexpr (!pin_q_v<D>) {
        for (std::size_t r = 0; r < rows_v<D>; ++r) {
            assert(r >= D::n_pins || o.q[r] == Real{0});
            qdot_n[r] = r >= D::n_pins ? -o.f[r] : Real{0};
        }
        q_n = o.q;
        return o;
    }
    const Real a = formula_a(c);
    for (std::size_t r = 0; r < rows_v<D>; ++r) {
        const Real qd = r >= D::n_pins
            ? -o.f[r]
            : c.formula == Formula::BackwardEuler   // the hand commit divides
                ? (o.q[r] - q_n[r]) / c.h
                : formula_rate(c, a, o.q[r], q_n[r], qdot_n[r], q_gamma[r]);
        q_n[r] = o.q[r];
        qdot_n[r] = qd;
    }
    return o;
}

template <Device D>
void commit_fq(const typename D::template Params<Real>& p,
               const std::array<Real, rows_v<D>>& x, const StepCtx& c,
               std::array<Real, rows_v<D>>& q_n,
               std::array<Real, rows_v<D>>& qdot_n,
               const std::array<Real, rows_v<D>>& q_gamma) {
    (void)commit_fq<D>(aux_t<D>{}, p, x, c, q_n, qdot_n, q_gamma);
}

}  // namespace pulsim::device

#undef PULSIM_DEVICE_INLINE
