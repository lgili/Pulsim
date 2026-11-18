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
// (same member names), so there is one body for both paths. The charge and
// flux laws below repeat the model's expression in the same operation
// order, so their Real values equal the model's to rounding; the hand
// stamp's Jacobian (C(v), L(i)) and the law's AD derivative agree to a few
// ulp. Table and Jiles–Atherton laws go through `ad::chain`: exact ∂/∂x,
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
#include "pulsim/models/mosfet_level1.hpp"
#include "pulsim/models/nonlinear_capacitor.hpp"
#include "pulsim/models/saturable_inductor.hpp"
#include "pulsim/models/shockley_diode.hpp"
#include "pulsim/numeric/types.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

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
    static constexpr std::size_t n_params = 6;
    template <class S> using Params = ShockleyParams<S>;
    [[nodiscard]] static Params<Real> from(const models::ShockleyDiode::Params& m) {
        return {m.I_S, m.n, m.V_T, m.G_min, m.i_lim, m.BV};
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
/// Pins {drain, source, gate}.
struct MosfetLevel1 : CurrentLaw<models::MosfetLevel1, 3, MosfetL1Params> {
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

    /// Q(v), the model's `charge` in the same operation order.
    template <class T, class P>
    [[nodiscard]] static T charge(const Aux& a, const Params<P>& p, const T& v) {
        using std::abs; using std::atan; using std::exp; using std::log1p;
        using std::pow; using std::tanh;
        if (a.law == models::ChargeLawKind::Table) {
            const Real vv = value_of(v);
            return ad::chain(v, a.table->charge(vv), a.table->capacitance(vv));
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
        T u = v / p.V0;
        if (u < a.v_floor) u = T(a.v_floor);
        const auto e = Real{1} - p.m;
        return p.C0 * p.V0 / e * (pow(Real{1} + u, e) - Real{1});
    }

    template <class P, class X, class O>
    static void eval(const Aux& a, const Params<P>& p, const std::array<X, 2>& x, O& o) {
        using T = promote_t<P, X>;
        const T q = charge<T>(a, p, T(x[0]) - T(x[1]));
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
    static constexpr std::size_t n_pins = 2, n_internal = 1, n_params = 3;
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
        o.q[2] = -flux<T>(a, p, i);
    }
};

static_assert(Device<ShockleyDiode>);
static_assert(Device<IdealDiode>);
static_assert(Device<MosfetLevel1>);
static_assert(Device<IgbtLevel1>);
static_assert(Device<NonlinearCapacitor>);
static_assert(Device<SaturableInductor>);

}  // namespace pulsim::device::laws
