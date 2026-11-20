#pragma once

// =============================================================================
// Pulsim — the shipped Newton kinds as laws on the `Device` contract
// =============================================================================
//
// add-unified-device-interface steps 5–7 under the dual path (DECISIONS
// #23). Each law is written ONCE as a template over the scalar and models
// `device::Device`: Real (the residual), AD unknowns (the Newton Jacobian)
// and AD parameters (∂f/∂p, ∂q/∂p for add-parameter-sensitivities).
//
// The static kinds (smooth / Shockley diode, MOSFET and IGBT level 1) ARE
// their model's `current<S, P>` body: the law passes its own `Params<S>`
// (same member names), so there is one body for both paths (the MOSFET's
// C¹, passive form at V_DS = 0 lives in the model, #44). The charge and
// flux laws below repeat the model's expression in the same operation
// order, so their Real values equal the model's to rounding; the hand
// stamp's Jacobian (C(v), L(i)) and the law's AD derivative agree to a few
// ulp. Exception: the junction Coss takes value and tangent from ONE pow
// (the model's two calls cost the 3 % speed gate), a few ulp of the model.
// A law univariate in one quantity (λ(i), Q(v)) runs its AD at one tangent
// (`univariate`), not at the row count. Table and Jiles–Atherton laws go through `ad::chain`: exact ∂/∂x,
// NO parameter tangent (a table has no parameters; the hysteretic core's
// parameters are not differentiated), so a seeded ∂/∂p there reads 0. Who
// has one: `kParamTangent` (static laws, always), `param_tangent(aux)`
// (the stateful laws, per instance). A sensitivity consumer must check it
// and refuse rather than report the zero.
//
// Rows: pins first, then internal rows. `kPattern[r][k]` is the structural
// Jacobian entry set the engines stamp (a hand stamp writes exactly these;
// writing a structural zero would insert into the sparse pattern).

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/device/device.hpp"
#include "pulsim/models/charge_table.hpp"
#include "pulsim/models/flux_table.hpp"
#include "pulsim/models/ideal_diode.hpp"
#include "pulsim/models/igbt_level1.hpp"
#include "pulsim/models/jiles_atherton.hpp"
#include "pulsim/models/lauritzen_diode.hpp"
#include "pulsim/models/magnetic_loss.hpp"
#include "pulsim/models/mosfet_level1.hpp"
#include "pulsim/models/nonlinear_capacitor.hpp"
#include "pulsim/models/saturable_inductor.hpp"
#include "pulsim/models/shockley_diode.hpp"
#include "pulsim/numeric/types.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <string_view>
#include <stdexcept>
#include <type_traits>

namespace pulsim::device::laws {

// ---- scalar plumbing --------------------------------------------------------
template <class A, class B> struct promote { using type = A; };
template <Size N> struct promote<Real, ad::ADRealN<N>> { using type = ad::ADRealN<N>; };
template <Size N> struct promote<ad::ADRealN<N>, Real> { using type = ad::ADRealN<N>; };
/// The scalar of one evaluation: AD if the parameters or the unknowns are.
template <class A, class B> using promote_t = typename promote<A, B>::type;

[[nodiscard]] inline Real value_of(Real v) noexcept { return v; }
template <Size N>
[[nodiscard]] inline Real value_of(const ad::ADRealN<N>& v) noexcept { return v.value(); }

template <class T> inline constexpr bool is_ad_v = false;
template <Size N> inline constexpr bool is_ad_v<ad::ADRealN<N>> = true;

/// g(u) for a law univariate in u (λ(i), Q(v)), parameters Real: g runs at
/// ONE tangent and `ad::chain` spreads it over u's N slots (u's slots are
/// the seeds ±1 or 0 at every call site, so the result equals g at width N
/// to the bit; for a general AD u it is equal to rounding — at the cost
/// of one tangent instead of N — the 3 % speed gate, task 15.7). With AD
/// parameters (sensitivities) g runs at the full width.
template <class P, class T, class G>
[[nodiscard]] inline T univariate(const T& u, G&& g) {
    if constexpr (std::is_same_v<P, Real> && is_ad_v<T>) {
        const auto r = g(ad::ADRealN<1>{u.value(), {Real{1}}});
        return ad::chain(u, r.value(), r.deriv(0));
    } else {
        return g(u);
    }
}

template <std::size_t R>
using Pattern = std::array<std::array<bool, R>, R>;

/// Every entry of the pin rows 0 and 1 (a two-terminal current, or a
/// three-terminal one whose control terminal carries no current).
template <std::size_t R>
[[nodiscard]] constexpr Pattern<R> current_pattern() {
    Pattern<R> p{};
    for (std::size_t k = 0; k < R; ++k) p[0][k] = p[1][k] = true;
    return p;
}

// ---- static kinds: i(v) from terminal 0 to terminal 1 ----------------------
/// f = {i, −i, 0…}, no charge. `Model::current<T>(v, p)` is the one body.
template <class Model, std::size_t NT, template <class> class ParamsT>
struct CurrentLaw {
    static constexpr std::size_t n_pins = NT;
    static constexpr std::size_t n_internal = 0;
    static constexpr bool kHasQ = false;
    static constexpr bool kParamTangent = true;
    static constexpr Pattern<NT> kPattern = current_pattern<NT>();
    template <class P, class X, class O>
    static void eval(const ParamsT<P>& p, const std::array<X, NT>& x, O& o) {
        using T = promote_t<P, X>;
        std::array<T, NT> v{};
        for (std::size_t k = 0; k < NT; ++k) v[k] = T(x[k]);
        const T i = Model::template current<T>(v.data(), p);
        o.f[0] = i;
        o.f[1] = -i;
    }
};

template <class S> struct ShockleyParams {
    S I_S, n, V_T, G_min, i_lim, BV;
    template <class F> auto map(F&& f) const {
        return ShockleyParams<decltype(f(I_S))>{f(I_S), f(n), f(V_T),
                                                f(G_min), f(i_lim), f(BV)};
    }
};
struct ShockleyDiode : CurrentLaw<models::ShockleyDiode, 2, ShockleyParams> {
    /// Parameter names in `Params` declaration order (sensitivity refusals).
    static constexpr std::array<std::string_view, 6> kParamNames{"I_S", "n", "V_T", "G_min", "i_lim", "BV"};
    static constexpr std::size_t n_params = 6;
    template <class S> using Params = ShockleyParams<S>;
    [[nodiscard]] static Params<Real> from(const models::ShockleyDiode::Params& m) {
        return {m.I_S, m.n, m.V_T, m.G_min, m.i_lim, m.BV};
    }
};

template <class S> struct ExcessLossParams {
    S c, eps;
    template <class F> auto map(F&& f) const {
        return ExcessLossParams<decltype(f(c))>{f(c), f(eps)};
    }
};
/// The excess-loss element's dual (add-permeance-magnetic-domain 4.2):
/// i = c·v/(v²+ε²)^¼ between {a, b}; the model's body.
struct ExcessLoss : CurrentLaw<models::ExcessLoss, 2, ExcessLossParams> {
    /// Parameter names in `Params` declaration order (sensitivity refusals).
    static constexpr std::array<std::string_view, 2> kParamNames{"c", "eps"};
    static constexpr std::size_t n_params = 2;
    template <class S> using Params = ExcessLossParams<S>;
    [[nodiscard]] static Params<Real> from(const models::ExcessLoss::Params& m) {
        return {m.c, m.eps};
    }
};

template <class S> struct IdealDiodeParams {
    S V_F0, R_d, G_off, kappa;
    template <class F> auto map(F&& f) const {
        return IdealDiodeParams<decltype(f(V_F0))>{f(V_F0), f(R_d), f(G_off),
                                                  f(kappa)};
    }
};
struct IdealDiode : CurrentLaw<models::IdealDiode, 2, IdealDiodeParams> {
    /// Parameter names in `Params` declaration order (sensitivity refusals).
    static constexpr std::array<std::string_view, 4> kParamNames{"V_F0", "R_d", "G_off", "kappa"};
    static constexpr std::size_t n_params = 4;
    template <class S> using Params = IdealDiodeParams<S>;
    [[nodiscard]] static Params<Real> from(const models::IdealDiode::Params& m) {
        return {m.V_F0, m.R_d, m.G_off, m.kappa};
    }
};

template <class S> struct MosfetL1Params {
    S K, V_T, lambda, kappa;
    template <class F> auto map(F&& f) const {
        return MosfetL1Params<decltype(f(K))>{f(K), f(V_T), f(lambda), f(kappa)};
    }
};
/// Pins {drain, source, gate}. The model body: C¹ and zero current at
/// V_DS = 0 (forward_c1_, DECISIONS #44).
struct MosfetLevel1 : CurrentLaw<models::MosfetLevel1, 3, MosfetL1Params> {
    /// Parameter names in `Params` declaration order (sensitivity refusals).
    static constexpr std::array<std::string_view, 4> kParamNames{"K", "V_T", "lambda", "kappa"};
    static constexpr std::size_t n_params = 4;
    template <class S> using Params = MosfetL1Params<S>;
    [[nodiscard]] static Params<Real> from(const models::MosfetLevel1::Params& m) {
        return {m.K, m.V_T, m.lambda, m.kappa};
    }
};

template <class S> struct IgbtL1Params {
    S V_CE_sat, R_CE_sat, V_T, kappa, v_knee;
    template <class F> auto map(F&& f) const {
        return IgbtL1Params<decltype(f(V_CE_sat))>{f(V_CE_sat), f(R_CE_sat),
                                                  f(V_T), f(kappa), f(v_knee)};
    }
};
/// Pins {collector, emitter, gate}; the steady-state law (the turn-off
/// tail is the stateful IgbtTail kind).
struct IgbtLevel1 : CurrentLaw<models::IgbtLevel1, 3, IgbtL1Params> {
    /// Parameter names in `Params` declaration order (sensitivity refusals).
    static constexpr std::array<std::string_view, 5> kParamNames{"V_CE_sat", "R_CE_sat", "V_T", "kappa", "v_knee"};
    static constexpr std::size_t n_params = 5;
    template <class S> using Params = IgbtL1Params<S>;
    [[nodiscard]] static Params<Real> from(const models::IgbtLevel1::Params& m) {
        return {m.V_CE_sat, m.R_CE_sat, m.V_T, m.kappa, m.v_knee};
    }
};

// ---- charge kind: the nonlinear capacitor (Coss, junction) -----------------
/// q = {Q(v), −Q(v)}, v = x0 − x1; no f. Parameters of the junction law
/// (C0, V0, m) and of the VDMOS C_gd law (CGDMAX, CGDMIN, A); a table law
/// has none.
template <class S> struct NonlinearCapacitorParams {
    S C0, V0, m, cgd_max, cgd_min, a;
    template <class F> auto map(F&& f) const {
        return NonlinearCapacitorParams<decltype(f(C0))>{
            f(C0), f(V0), f(m), f(cgd_max), f(cgd_min), f(a)};
    }
};
struct NonlinearCapacitor {
    /// Parameter names in `Params` declaration order (sensitivity refusals).
    static constexpr std::array<std::string_view, 6> kParamNames{"C0", "V0", "m", "cgd_max", "cgd_min", "a"};
    static constexpr std::size_t n_pins = 2, n_internal = 0, n_params = 6;
    static constexpr bool kHasF = false;
    static constexpr Pattern<2> kPattern = current_pattern<2>();
    template <class S> using Params = NonlinearCapacitorParams<S>;
    struct Aux {
        models::ChargeLawKind law = models::ChargeLawKind::Junction;
        Real v_floor = Real{-0.9};
        const models::ChargeTable* table = nullptr;
    };
    [[nodiscard]] static Params<Real> from(const models::NonlinearCapacitor::Params& m) {
        return {m.C0, m.V0, m.m, m.cgd_max, m.cgd_min, m.a};
    }
    [[nodiscard]] static Aux aux_from(const models::NonlinearCapacitor::Params& m) {
        return {m.law, m.v_floor, m.table.get()};
    }
    [[nodiscard]] static bool param_tangent(const Aux& a) noexcept {
        return a.law != models::ChargeLawKind::Table;
    }

    /// Q(v): the model's `charge` in the same operation order (VDMOS),
    /// the table's value, or the junction form below (a few ulp of it).
    template <class T, class P>
    [[nodiscard]] static T charge(const Aux& a, const Params<P>& p, const T& v) {
        using std::abs; using std::atan; using std::exp; using std::log1p;
        using std::pow; using std::tanh;
        if (a.law == models::ChargeLawKind::Table) {
            const Real vv = value_of(v);
            return ad::chain(v, a.table->charge(vv), a.table->capacitance(vv));
        }
        if (a.law == models::ChargeLawKind::Tanh) {
            // the saturable permeance (add-permeance-magnetic-domain): the
            // model's charge, same operation order; with Real parameters the
            // tangent is the model's sech² (AD tanh's 1 − t² cancels in
            // saturation: (C_max/C_min)·eps of J, exactly C_min past 19·V0)
            if constexpr (std::is_same_v<P, Real> && is_ad_v<T>) {
                const Real vv = value_of(v);
                const Real q = p.cgd_min * vv + (p.cgd_max - p.cgd_min) * p.V0 * std::tanh(vv / p.V0);
                return ad::chain(v, q, p.cgd_min + (p.cgd_max - p.cgd_min)
                                           * models::NonlinearCapacitor::sech2_(vv / p.V0));
            } else {
                return p.cgd_min * v + (p.cgd_max - p.cgd_min) * p.V0 * tanh(v / p.V0);
            }
        }
        if (a.law == models::ChargeLawKind::VdmosCgd) {
            const auto s = (p.cgd_max - p.cgd_min)
                           / (Real{1} + std::numbers::pi_v<Real> / Real{2});
            const T x = p.a * v;
            if (x > Real{0}) {
                const T ax = abs(x);
                const T lc = ax + log1p(exp(Real{-2} * ax))
                             - std::numbers::ln2_v<Real>;
                return (p.cgd_max - s) * v + (s / p.a) * lc;
            }
            const T cap = x < Real{-1} ? T(p.cgd_min - s * atan(Real{1} / x))
                                       : T(p.cgd_max - s + s * atan(x));
            return v * cap - s * log1p(x * x) / (Real{2} * p.a);
        }
        // Junction: Q = C0·V0/(1−m)·((1+u)·(1+u)^(−m) − 1), u = v/V0 clamped
        // at the floor; dQ/dv = C0·(1+u)^(−m) — ONE pow for value and
        // tangent (the model's charge + capacitance take two: the 3 % speed
        // gate, task 15.7). Below the floor Q is flat and the tangent stays
        // C(floor), as the hand stamp's (DECISIONS #26: a zero row would be
        // a singular Newton matrix). AD parameters take the same operations
        // in AD arithmetic (same value to the bit; x is Real there).
        if constexpr (std::is_same_v<P, Real>) {
            Real u = value_of(v) / p.V0;
            if (u < a.v_floor) u = a.v_floor;
            const Real w = std::pow(Real{1} + u, -p.m);
            const Real q = p.C0 * p.V0 / (Real{1} - p.m) * ((Real{1} + u) * w - Real{1});
            return ad::chain(v, q, p.C0 * w);
        } else {
            T u = v / p.V0;
            if (u < a.v_floor) u = T(a.v_floor);
            const T w = pow(Real{1} + u, -p.m);
            return p.C0 * p.V0 / (Real{1} - p.m) * ((Real{1} + u) * w - Real{1});
        }
    }

    template <class P, class X, class O>
    static void eval(const Aux& a, const Params<P>& p, const std::array<X, 2>& x, O& o) {
        using T = promote_t<P, X>;
        const T q = univariate<P>(T(x[0]) - T(x[1]), [&](const auto& v) {
            return charge<std::remove_cvref_t<decltype(v)>>(a, p, v);
        });
        o.q[0] = q;
        o.q[1] = -q;
    }
};

// ---- flux kind: the saturable inductor (atan, table, Jiles–Atherton) -------
/// Pins {from, to}, internal row = the branch current i. f = {i, −i, v},
/// q = {0, 0, −λ(i)}: the branch row is v − dλ/dt = 0 in conductance form
/// (J[2][2] = −a·L(i), never scaled by h).
template <class S> struct SaturableInductorParams {
    S L_0, I_sat, L_residual;
    template <class F> auto map(F&& f) const {
        return SaturableInductorParams<decltype(f(L_0))>{f(L_0), f(I_sat),
                                                        f(L_residual)};
    }
};
struct SaturableInductor {
    /// Parameter names in `Params` declaration order (sensitivity refusals).
    static constexpr std::array<std::string_view, 3> kParamNames{"L_0", "I_sat", "L_residual"};
    static constexpr std::size_t n_pins = 2, n_internal = 1, n_params = 3;
    static constexpr bool kPinQ = false;   // the flux is the internal row's
    static constexpr Pattern<3> kPattern{{{false, false, true},
                                          {false, false, true},
                                          {true, true, true}}};
    template <class S> using Params = SaturableInductorParams<S>;
    struct Aux {
        const models::FluxTable* table = nullptr;
        const models::JilesAthertonCore::Params* ja = nullptr;
        /// The step-start (H, M) the hysteretic law integrates from.
        const models::JilesAthertonCore::State* base = nullptr;
    };
    [[nodiscard]] static Params<Real> from(const models::SaturableInductor::Params& m) {
        return {m.L_0, m.I_sat, m.L_residual};
    }
    [[nodiscard]] static bool param_tangent(const Aux& a) noexcept {
        return a.table == nullptr && a.ja == nullptr;
    }

    /// λ(i), the model's `flux` in the same operation order.
    template <class T, class P>
    [[nodiscard]] static T flux(const Aux& a, const Params<P>& p, const T& i) {
        using std::atan;
        if (a.table != nullptr) {
            const Real iv = value_of(i);
            return ad::chain(i, a.table->flux(iv), a.table->inductance(iv));
        }
        if (a.ja != nullptr) {
            const auto ev = models::JilesAthertonCore::evaluate(*a.ja, *a.base,
                                                                value_of(i));
            return ad::chain(i, ev.lambda, ev.L);
        }
        return p.L_residual * i + (p.L_0 - p.L_residual) * p.I_sat * atan(i / p.I_sat);
    }

    template <class P, class X, class O>
    static void eval(const Aux& a, const Params<P>& p, const std::array<X, 3>& x, O& o) {
        using T = promote_t<P, X>;
        const T i = T(x[2]);
        o.f[0] = i;
        o.f[1] = -i;
        o.f[2] = T(x[0]) - T(x[1]);
        o.q[2] = -univariate<P>(i, [&](const auto& u) {
            return flux<std::remove_cvref_t<decltype(u)>>(a, p, u);
        });
    }

    /// `eval` in Real with J = ∂f/∂x + a·∂q/∂x written out (`stamp_fq`'s
    /// hand-Jacobian branch, DECISIONS #50): λ and dλ/di from the same
    /// one-tangent flux `univariate` runs, the five structural entries as
    /// fd + a·qd of the AD tangents (exact zeros dropped). J arrives zeroed.
    static void eval_jac(const Aux& a, const Params<Real>& p, const std::array<Real, 3>& x,
                         Real fa, Out<Real, 3>& o, std::array<std::array<Real, 3>, 3>& J) {
        const Real i = x[2];
        o.f[0] = i;
        o.f[1] = -i;
        o.f[2] = x[0] - x[1];
        const auto r = flux<ad::ADRealN<1>>(a, p, ad::ADRealN<1>{i, {Real{1}}});
        o.q[2] = -r.value();
        J[0][2] = Real{1};
        J[1][2] = Real{-1};
        J[2][0] = Real{1};
        J[2][1] = Real{-1};
        J[2][2] = fa * -r.deriv(0);
    }
};

// ---- stored-charge kinds: one LOCAL row (condensed by stamp_fq_local) -----
/// Lauritzen–Mattsson diode. Pins {anode, cathode}; local row = the stored
/// charge q_M (the engines never see it; its committed value and rate live
/// in the history). f = {i, −i, A·q_M − q_E/T_M}, q = {0, 0, q_M}, with
/// i = (q_E(v) − q_M)/T_M + G_min·v and A = 1/T_M + 1/τ: the hand stamp's
/// q_M = K0 + K1·q_E is this local row solved for one step.
template <class S> struct LauritzenParams {
    S I_S, n, V_T, tau, T_M, G_min, i_lim;
    template <class F> auto map(F&& f) const {
        return LauritzenParams<decltype(f(I_S))>{f(I_S), f(n), f(V_T), f(tau),
                                                f(T_M), f(G_min), f(i_lim)};
    }
};
struct LauritzenDiode {
    /// Parameter names in `Params` declaration order (sensitivity refusals).
    static constexpr std::array<std::string_view, 7> kParamNames{"I_S", "n", "V_T", "tau", "T_M", "G_min", "i_lim"};
    static constexpr std::size_t n_pins = 2, n_internal = 0, n_local = 1,
                                 n_params = 7;
    static constexpr bool kParamTangent = true;
    static constexpr bool kLocalAffine = true;
    static constexpr Pattern<2> kPattern = current_pattern<2>();
    template <class S> using Params = LauritzenParams<S>;
    [[nodiscard]] static Params<Real> from(const models::LauritzenDiode::Params& m) {
        return {m.I_S, m.n, m.V_T, m.tau, m.T_M, m.G_min, m.i_lim};
    }
    /// q_E(v) = τ·i_shockley(v): the model's `junction_charge` body.
    template <class T, class P>
    [[nodiscard]] static T junction_charge(const Params<P>& p, const T& v) {
        const ShockleyParams<P> sp{p.I_S, p.n, p.V_T, P(Real{0}), p.i_lim, P(Real{0})};
        const T vv[2] = {v, T(Real{0})};
        return T(p.tau) * models::ShockleyDiode::current<T>(vv, sp);
    }
    template <class P, class X, class O>
    static void eval(const Params<P>& p, const std::array<X, 3>& x, O& o) {
        using T = promote_t<P, X>;
        const T v = T(x[0]) - T(x[1]);
        const T qm = T(x[2]);
        const T qe = univariate<P>(v, [&](const auto& u) {
            return junction_charge<std::remove_cvref_t<decltype(u)>>(p, u);
        });
        // 1/T_M once: a division per AD slot is most of this law's cost.
        const auto inv_tm = Real{1} / p.T_M;
        const T i = (qe - qm) * inv_tm + v * p.G_min;
        o.f[0] = i;
        o.f[1] = -i;
        o.f[2] = (inv_tm + Real{1} / p.tau) * qm - qe * inv_tm;
        o.q[2] = qm;
    }
};

/// IGBT turn-off tail. Pins {collector, emitter, gate}; local row = the
/// stored charge Q, dQ/dt = k_tail·i_ss(v) − Q/τ_tail. The branch adds
/// i_tail = Q/τ_tail − k_tail·i_ss(v) = −dQ/dt to the static law's i_ss
/// (the IgbtLevel1 kind), so f = {i_tail, −i_tail, 0, i_tail}, q = {0, 0,
/// 0, Q}: the hand stamp's Q = K0 + K1·i_ss is the local row solved.
template <class S> struct IgbtTailParams {
    S V_CE_sat, R_CE_sat, V_T, kappa, v_knee, tau_tail, k_tail;
    template <class F> auto map(F&& f) const {
        return IgbtTailParams<decltype(f(V_CE_sat))>{
            f(V_CE_sat), f(R_CE_sat), f(V_T), f(kappa), f(v_knee), f(tau_tail),
            f(k_tail)};
    }
};
struct IgbtTail {
    /// Parameter names in `Params` declaration order (sensitivity refusals).
    static constexpr std::array<std::string_view, 7> kParamNames{"V_CE_sat", "R_CE_sat", "V_T", "kappa", "v_knee", "tau_tail", "k_tail"};
    static constexpr std::size_t n_pins = 3, n_internal = 0, n_local = 1,
                                 n_params = 7;
    static constexpr bool kParamTangent = true;
    static constexpr bool kLocalAffine = true;
    static constexpr Pattern<3> kPattern = current_pattern<3>();
    template <class S> using Params = IgbtTailParams<S>;
    [[nodiscard]] static Params<Real> from(const models::IgbtLevel1::Params& m) {
        return {m.V_CE_sat, m.R_CE_sat, m.V_T, m.kappa, m.v_knee, m.tau_tail, m.k_tail};
    }
    template <class P, class X, class O>
    static void eval(const Params<P>& p, const std::array<X, 4>& x, O& o) {
        using T = promote_t<P, X>;
        const std::array<T, 3> v{T(x[0]), T(x[1]), T(x[2])};
        const T i_ss = models::IgbtLevel1::steady_state_current<T>(v.data(), p);
        const T i = T(x[3]) / p.tau_tail - p.k_tail * i_ss;
        o.f[0] = i;
        o.f[1] = -i;
        o.f[3] = i;
        o.q[3] = T(x[3]);
    }
};

static_assert(Device<ShockleyDiode>);
static_assert(Device<ExcessLoss>);
static_assert(Device<IdealDiode>);
static_assert(Device<MosfetLevel1>);
static_assert(Device<IgbtLevel1>);
static_assert(Device<NonlinearCapacitor>);
static_assert(Device<SaturableInductor>);
static_assert(Device<LauritzenDiode>);
static_assert(Device<IgbtTail>);

/// One name per parameter; `Params<AD>` packs like `Params<Real>` (no
/// padding: `seed_params` and the sensitivity solve index it by slot).
template <class D>
inline constexpr bool kLawLayout =
    D::kParamNames.size() == D::n_params
    && sizeof(typename D::template Params<ad::ADRealN<D::n_params>>)
           == D::n_params * sizeof(ad::ADRealN<D::n_params>)
    && std::is_standard_layout_v<typename D::template Params<ad::ADRealN<D::n_params>>>;
static_assert(kLawLayout<ShockleyDiode> && kLawLayout<ExcessLoss> && kLawLayout<IdealDiode>
              && kLawLayout<MosfetLevel1> && kLawLayout<IgbtLevel1>
              && kLawLayout<NonlinearCapacitor> && kLawLayout<SaturableInductor>
              && kLawLayout<LauritzenDiode> && kLawLayout<IgbtTail>);

/// Does this instance's law carry ∂/∂p? Static and stored-charge laws
/// always (`kParamTangent`); a table or Jiles–Atherton law never
/// (`param_tangent(aux)`: its value and x-tangent come through `ad::chain`,
/// whose parameter slots read 0).
template <class D>
[[nodiscard]] bool has_param_tangent(const aux_t<D>& aux) noexcept {
    if constexpr (requires { D::param_tangent(aux); }) return D::param_tangent(aux);
    else return static_cast<bool>(D::kParamTangent);
}

/// What a sensitivity consumer does with an instance whose law has no
/// parameter tangent: refuse, naming the device and the parameter, rather
/// than report the 0 its AD slots read. Call it per requested parameter.
template <class D>
void require_param_tangent(const aux_t<D>& aux, std::string_view device,
                           std::size_t param) {
    if (param >= D::n_params) {
        throw std::invalid_argument("sensitivity: " + std::string(device)
                                    + " has no parameter #" + std::to_string(param));
    }
    if (!has_param_tangent<D>(aux)) {
        throw std::invalid_argument(
            "sensitivity of " + std::string(device) + "." + std::string(D::kParamNames[param])
            + ": this instance's law (a table or a hysteretic core) has no derivative in "
              "its parameters; refused rather than reported as 0");
    }
}

}  // namespace pulsim::device::laws
