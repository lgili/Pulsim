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
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

template <class D>
inline constexpr std::size_t rows_v = D::n_pins + D::n_internal;

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
Real stamp_fq(const aux_t<D>& aux,
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
    std::array<ad::ADRealN<R>, R> xs{};
    for (std::size_t k = 0; k < R; ++k) {
        std::array<Real, R> d{};
        d[k] = Real{1};
        xs[k] = ad::ADRealN<R>{x[k], d};
    }
    Out<ad::ADRealN<R>, R> o;
    device::eval<D>(aux, p, xs, o);
    constexpr bool hq = has_q_v<D>, hf = has_f_v<D>;
    if constexpr (!hq) {
        for (std::size_t r = 0; r < R; ++r) {
            F[r] = o.f[r].value();
            for (std::size_t k = 0; k < R; ++k) J[r][k] = o.f[r].deriv(k);
        }
        Real worst = Real{0};
        for (std::size_t r = 0; r < R; ++r) worst = std::max(worst, std::abs(F[r]));
        return worst;
    }
    const Real a = formula_a(c);
    if constexpr (!hf) {
        const auto hist = history_terms<R>(c, q_n, qdot_n, q_gamma);
        Real worst = Real{0};
        for (std::size_t r = 0; r < R; ++r) {
            F[r] = a * o.q[r].value() + hist[r];
            for (std::size_t k = 0; k < R; ++k) J[r][k] = a * o.q[r].deriv(k);
            worst = std::max(worst, std::abs(F[r]));
        }
        return worst;
    }
    const auto hist = dq == nullptr ? history_terms<R>(c, q_n, qdot_n, q_gamma)
                                    : std::array<Real, R>{};
    Real worst = Real{0};
    for (std::size_t r = 0; r < R; ++r) {
        if (dq == nullptr) {
            F[r] = o.f[r].value() + a * o.q[r].value() + hist[r];
        } else {
            Real hist = Real{0};
            if (c.formula == Formula::Trapezoidal) {
                hist = -qdot_n[r];
            } else if (c.formula == Formula::Bdf2Stage2) {
                hist = bdf2_weights().c2 * (*dq_gamma)[r] / c.h;
            }
            F[r] = o.f[r].value() + a * (*dq)[r] + hist;
        }
        for (std::size_t k = 0; k < R; ++k) {
            J[r][k] = o.f[r].deriv(k) + a * o.q[r].deriv(k);
        }
        worst = std::max(worst, std::abs(F[r]));
    }
    return worst;
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

/// Commit after an accepted step: q_n ← q(x) and q̇_n ← the derivative the
/// step produced. On a PIN row that is the formula's a·q(x) + r (the row
/// is a shared KCL, so f alone is not the derivative; physical, so a
/// variable h stays legal). On an INTERNAL row the device owns the whole
/// equation f + q̇ = 0, so q̇_n ← −f(x) exactly, not the formula value,
/// which differs from it by the Newton residual (a saturable inductor's
/// committed voltage is the solved v, as in every companion model).
/// Returns the Real evaluation at `x` (the capture of a stage point reads
/// it).
template <Device D>
Out<Real, rows_v<D>> commit_fq(const aux_t<D>& aux,
                               const typename D::template Params<Real>& p,
                               const std::array<Real, rows_v<D>>& x,
                               const StepCtx& c,
                               std::array<Real, rows_v<D>>& q_n,
                               std::array<Real, rows_v<D>>& qdot_n,
                               const std::array<Real, rows_v<D>>& q_gamma) {
    Out<Real, rows_v<D>> o;
    device::eval<D>(aux, p, x, o);
    const Real a = formula_a(c);
    for (std::size_t r = 0; r < rows_v<D>; ++r) {
        const Real qd = r >= D::n_pins
            ? -o.f[r]
            : a * o.q[r] + history_term(c, q_n[r], qdot_n[r], q_gamma[r]);
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
