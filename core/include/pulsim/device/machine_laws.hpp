#pragma once

// =============================================================================
// Pulsim — the MNA-native machines as laws on the `Device` contract
// =============================================================================
//
// add-unified-device-interface 15.6 (DECISIONS #34, #42). Two laws:
//
//   PmsmMna            rows {a, b, c, n, ω, θ | i_a, i_b, i_c}: the constant-L
//                      (PM + saliency, PM harmonics) and the flux-map machine
//   InductionMachine   rows {a, b, c, n, ω, λ_rα, λ_rβ | i_a, i_b, i_c}
//
// The conserved quantity of each stator row is its FLUX LINKAGE: q = −λ_k(i, θ)
// (PMSM) or −λ_k(i, λ_r) (IM) on the current rows, so the integrator
// differentiates λ — never L·i — and the row stays in conductance form,
// J = −R·δ − a·∂λ/∂i (a = 2/h, 1/h, c1/h; never scaled by h). The engines pass
// the step's increment λ(x) − λ_n in closed form (`stamp_fq`'s `dq`, the
// history's `delta_lambda`), as the hand stamp does. The angular momentum J·ω
// and the angle θ (PMSM) / rotor fluxes λ_r (IM) are the charges of the
// builder's linear capacitors on those nodes, which the engines integrate as
// every capacitor; the law only injects their rates: T_em − T_load into ω,
// C_θ·ω into θ, C_r·dλ_r/dt into the rotor-flux nodes.
//
// λ, T and their partials come from the model's Real functions (the same ones
// the hand stamp calls, in the same operation order) through `chain_n`: exact
// tangents in i, θ, λ_r and ω. The law's parameters are R_s and T_load, the two
// that enter it explicitly (exact ∂/∂p); the flux law's parameters live in
// `Aux` and are not differentiated.

#include "pulsim/device/laws.hpp"
#include "pulsim/models/induction_machine_mna.hpp"
#include "pulsim/models/pmsm_mna.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace pulsim::device::laws {

/// A value the model computed in Real with its partials `g` in the inputs
/// `u`: the result carries Σ_m g[m]·∂u[m]. At every engine call site the u
/// are seeds (one slot 1, the rest 0), so the tangents are the model's
/// partials to the bit.
template <std::size_t M>
[[nodiscard]] inline Real chain_n(const std::array<Real, M>& /*u*/, Real value,
                                  const std::array<Real, M>& /*g*/) noexcept {
    return value;
}
template <Size N, std::size_t M>
[[nodiscard]] inline ad::ADRealN<N> chain_n(const std::array<ad::ADRealN<N>, M>& u,
                                            Real value,
                                            const std::array<Real, M>& g) noexcept {
    std::array<Real, N> d{};
    for (std::size_t m = 0; m < M; ++m)
        for (Size k = 0; k < N; ++k) d[k] += g[m] * u[m].deriv(k);
    return ad::ADRealN<N>{value, d};
}

template <class S> struct MachineParams {
    S R_s, T_load;
    template <class F> auto map(F&& f) const {
        return MachineParams<decltype(f(R_s))>{f(R_s), f(T_load)};
    }
};

namespace machine_detail {
/// KCL of the three phase currents (rows 0–2 leave the phase node, row 3
/// enters the neutral) and the stator rows' f = (v_k − v_n) − R_s·i_k.
template <class T, class P, class X, class O>
void stator_f(const MachineParams<P>& p, const X& x, std::size_t i0, O& o) {
    const T vn = T(x[3]);
    for (std::size_t k = 0; k < 3; ++k) {
        const T i = T(x[i0 + k]);
        o.f[k] = i;
        o.f[3] = k == 0 ? -i : o.f[3] - i;
        o.f[i0 + k] = (T(x[k]) - vn) - p.R_s * i;
    }
}
}  // namespace machine_detail

// ---- PMSM (constant L(θ) with PM harmonics, or a flux map) ----------------
struct PmsmMna {
    static constexpr std::array<std::string_view, 2> kParamNames{"R_s", "T_load"};
    static constexpr std::size_t n_pins = 6, n_internal = 3, n_params = 2;
    static constexpr bool kPinQ = false;   // q = −λ on the stator rows only
    static constexpr bool kParamTangent = true;
    /// Rows a, b, c (0–2), n (3), ω (4), θ (5), i_a..i_c (6–8): the hand
    /// stamp's entries exactly.
    static constexpr Pattern<9> kPattern = [] {
        Pattern<9> p{};
        for (std::size_t k = 0; k < 3; ++k) {
            p[k][6 + k] = true;              // KCL at the phase
            p[3][6 + k] = true;              // KCL at the neutral
            p[4][6 + k] = true;              // ∂T/∂i
            p[6 + k][k] = p[6 + k][3] = true;
            p[6 + k][5] = true;              // ∂λ/∂θ
            for (std::size_t j = 0; j < 3; ++j) p[6 + k][6 + j] = true;
        }
        p[4][5] = true;                      // ∂T/∂θ
        p[5][4] = true;                      // C_θ·ω
        return p;
    }();
    template <class S> using Params = MachineParams<S>;
    struct Aux {
        const models::PmsmMna::Params* m = nullptr;
        Real c_theta = Real{1};   ///< the θ node's capacitance to ground
        /// Optional: inductance(m, θ_e) at THIS call's θ, when the caller
        /// already has it (the stamp's increment needs L too); a cache of
        /// the model, not data of the law. Null: evaluated here.
        const models::PmsmMna::Inductance* ind = nullptr;
    };
    [[nodiscard]] static Params<Real> from(const models::PmsmMna::Params& m) {
        return {m.R_s, m.T_load};
    }

    using Vec3 = models::PmsmMna::Vec3;
    using Mat3 = models::PmsmMna::Mat3;
    /// The model at (i, θ_m): λ, ∂λ/∂i, ∂λ/∂θ_m, T and its partials.
    struct State {
        Vec3 lam{};
        Mat3 dlam_di{};
        Vec3 dlam_dthm{};
        Real T = 0;
        Vec3 dT_di{};
        Real dT_dthm = 0;
    };
    [[nodiscard]] static State evaluate(const models::PmsmMna::Params& m,
                                        Real theta_m, const Vec3& i,
                                        const models::PmsmMna::Inductance* ind_at = nullptr) {
        using M = models::PmsmMna;
        const Real pp = m.pole_pairs, th = pp * theta_m;
        State s;
        if (m.flux_map) {
            const auto st = M::flux_map_state(m, th, i);
            s.lam = st.lam;
            s.dlam_di = st.dlam_di;
            for (std::size_t k = 0; k < 3; ++k) s.dlam_dthm[k] = pp * st.dlam_dth[k];
            s.T = st.T;
            s.dT_di = st.dT_di;
            s.dT_dthm = st.dT_dtheta_m;
            return s;
        }
        // `PmsmMnaHistory::lambda_of` and the hand stamp, same order.
        const auto ind = ind_at != nullptr ? *ind_at : M::inductance(m, th);
        const auto pm = M::lambda_pm_all(m, th);
        for (std::size_t k = 0; k < 3; ++k) {
            Real lam = pm.d0[k], dth = pm.d1[k];
            for (std::size_t j = 0; j < 3; ++j) lam += ind.L[k][j] * i[j];
            for (std::size_t j = 0; j < 3; ++j) dth += ind.dL[k][j] * i[j];
            s.lam[k] = lam;
            s.dlam_dthm[k] = pp * dth;
        }
        s.dlam_di = ind.L;
        const auto tq = M::torque(m, ind, pm.d1, pm.d2, i);
        s.T = tq.T;
        s.dT_di = tq.dT_di;
        s.dT_dthm = tq.dT_dtheta_m;
        return s;
    }

    template <class P, class X, class O>
    static void eval(const Aux& a, const Params<P>& p, const std::array<X, 9>& x, O& o) {
        using T = promote_t<P, X>;
        const auto s = evaluate(*a.m, value_of(x[5]),
                                {value_of(x[6]), value_of(x[7]), value_of(x[8])}, a.ind);
        machine_detail::stator_f<T>(p, x, 6, o);
        const std::array<T, 4> u{T(x[6]), T(x[7]), T(x[8]), T(x[5])};
        for (std::size_t k = 0; k < 3; ++k) {
            o.q[6 + k] = -chain_n(u, s.lam[k], {s.dlam_di[k][0], s.dlam_di[k][1],
                                                s.dlam_di[k][2], s.dlam_dthm[k]});
        }
        o.f[4] = -(chain_n(u, s.T, {s.dT_di[0], s.dT_di[1], s.dT_di[2], s.dT_dthm})
                   - p.T_load);
        o.f[5] = -(a.c_theta * T(x[4]));
    }
};

// ---- induction machine (squirrel cage, rotor flux on two nodes) -----------
struct InductionMachine {
    static constexpr std::array<std::string_view, 2> kParamNames{"R_s", "T_load"};
    static constexpr std::size_t n_pins = 7, n_internal = 3, n_params = 2;
    static constexpr bool kPinQ = false;   // q = −(M·λ_r + L·i) on the stator rows only
    static constexpr bool kParamTangent = true;
    /// Rows a, b, c (0–2), n (3), ω (4), λ_rα, λ_rβ (5, 6), i_a..i_c (7–9).
    static constexpr Pattern<10> kPattern = [] {
        Pattern<10> p{};
        for (std::size_t k = 0; k < 3; ++k) {
            p[k][7 + k] = p[3][7 + k] = p[4][7 + k] = true;
            p[7 + k][k] = p[7 + k][3] = p[7 + k][5] = p[7 + k][6] = true;
            for (std::size_t j = 0; j < 3; ++j) p[7 + k][7 + j] = true;
        }
        p[4][5] = p[4][6] = true;
        for (std::size_t n = 5; n < 7; ++n) {
            p[n][4] = p[n][5] = p[n][6] = true;
            for (std::size_t j = 7; j < 10; ++j) p[n][j] = true;
        }
        return p;
    }();
    template <class S> using Params = MachineParams<S>;
    struct Aux {
        const models::InductionMachineMna::Params* m = nullptr;
        const models::InductionMachineMna::StatorMatrices* sm = nullptr;
        std::array<Real, 2> c_rotor{Real{1}, Real{1}};
    };
    [[nodiscard]] static Params<Real> from(const models::InductionMachineMna::Params& m) {
        return {m.R_s, m.T_load};
    }

    /// The nonlinear terms' values and partials in u = (i_a, i_b, i_c, λ_rα,
    /// λ_rβ, ω) (rows 7–9 and 5, 6 and 4 of x), in Real.
    struct Terms {
        std::array<Real, 3> lam;                   // stator flux λ_k = M·λ_r + L·i
        std::array<std::array<Real, 6>, 3> dlam;
        std::array<Real, 2> g;                     // rotor-flux rate
        std::array<std::array<Real, 6>, 2> dg;
        Real Te;                                   // torque
        std::array<Real, 6> dT;
    };
    static constexpr std::array<std::size_t, 6> kU{7, 8, 9, 5, 6, 4};
    template <class X>
    [[nodiscard]] static Terms terms(const Aux& a, const std::array<X, 10>& x) {
        using IM = models::InductionMachineMna;
        const auto& m = *a.m;
        const auto& sm = *a.sm;
        const auto dv = IM::derived(m);
        const auto C = IM::clarke();
        const IM::Vec3 iv{value_of(x[7]), value_of(x[8]), value_of(x[9])};
        const IM::Vec2 lr{value_of(x[5]), value_of(x[6])};
        const Real omega = value_of(x[4]);
        Terms t;
        // stator flux: λ_k = M·λ_r + L·i (`PmsmMnaHistory::lambda_of` order)
        for (std::size_t k = 0; k < 3; ++k) {
            Real lam = sm.M[k][0] * lr[0] + sm.M[k][1] * lr[1];
            for (std::size_t j = 0; j < 3; ++j) lam += sm.L[k][j] * iv[j];
            t.lam[k] = lam;
            t.dlam[k] = {sm.L[k][0], sm.L[k][1], sm.L[k][2], sm.M[k][0], sm.M[k][1], Real{0}};
        }
        // rotor flux: C_r·dλ_r/dt = C_r·g(i, λ_r, ω) injected into its node
        t.g = IM::rotor_rhs(m, iv, lr, omega);
        const Real pp = m.pole_pairs, we = pp * omega;
        const Real dg_dl[2][2] = {{-dv.inv_tau_r, -we}, {we, -dv.inv_tau_r}};
        const Real dg_dw[2] = {-pp * lr[1], pp * lr[0]};
        for (std::size_t n = 0; n < 2; ++n)
            t.dg[n] = {dv.Lm_over_tau_r * C[n][0], dv.Lm_over_tau_r * C[n][1],
                       dv.Lm_over_tau_r * C[n][2], dg_dl[n][0], dg_dl[n][1], dg_dw[n]};
        // torque into the speed node
        const Real kt = Real{1.5} * pp * dv.k_r;
        const auto is = IM::i_ab(iv);
        t.Te = kt * (lr[0] * is[1] - lr[1] * is[0]);
        t.dT = {};
        for (std::size_t j = 0; j < 3; ++j) t.dT[j] = kt * (lr[0] * C[1][j] - lr[1] * C[0][j]);
        t.dT[3] = kt * is[1];
        t.dT[4] = -(kt * is[0]);
        return t;
    }

    template <class P, class X, class O>
    static void eval(const Aux& a, const Params<P>& p, const std::array<X, 10>& x, O& o) {
        using T = promote_t<P, X>;
        const Terms t = terms(a, x);
        machine_detail::stator_f<T>(p, x, 7, o);
        const std::array<T, 6> u{T(x[7]), T(x[8]), T(x[9]), T(x[5]), T(x[6]), T(x[4])};
        for (std::size_t k = 0; k < 3; ++k) o.q[7 + k] = -chain_n(u, t.lam[k], t.dlam[k]);
        for (std::size_t n = 0; n < 2; ++n) o.f[5 + n] = -(a.c_rotor[n] * chain_n(u, t.g[n], t.dg[n]));
        o.f[4] = -(chain_n(u, t.Te, t.dT) - p.T_load);
    }

    /// The internal rows only, as `eval` computes them (q = −λ, f = v − v_n −
    /// R_s·i): all `commit_fq` keeps of a law whose q is internal.
    static void eval_internal(const Aux& a, const Params<Real>& p,
                              const std::array<Real, 10>& x, Out<Real, 10>& o) {
        const auto& sm = *a.sm;
        machine_detail::stator_f<Real>(p, x, 7, o);
        for (std::size_t k = 0; k < 3; ++k) {
            Real lam = sm.M[k][0] * x[5] + sm.M[k][1] * x[6];
            for (std::size_t j = 0; j < 3; ++j) lam += sm.L[k][j] * x[7 + j];
            o.q[7 + k] = -lam;
        }
    }

    /// `eval` in Real with J = ∂f/∂x + a·∂q/∂x written out: the nonzero
    /// entries of the AD tangents of `eval` on unit seeds, each as fd + a·qd
    /// with fd or qd an exact zero dropped (x + 0 = x), so the same values.
    /// `stamp_fq` uses it instead of AD width 10; J arrives zeroed.
    static void eval_jac(const Aux& a, const Params<Real>& p, const std::array<Real, 10>& x,
                         Real fa, Out<Real, 10>& o, std::array<std::array<Real, 10>, 10>& J) {
        const Terms t = terms(a, x);
        machine_detail::stator_f<Real>(p, x, 7, o);
        for (std::size_t k = 0; k < 3; ++k) {
            J[k][7 + k] = Real{1};
            J[3][7 + k] = Real{-1};
            J[7 + k][k] = Real{1};
            J[7 + k][3] = Real{-1};
            o.q[7 + k] = -t.lam[k];
            for (std::size_t m = 0; m < 5; ++m) J[7 + k][kU[m]] = fa * -t.dlam[k][m];
            J[7 + k][7 + k] = -p.R_s + fa * -t.dlam[k][k];
        }
        for (std::size_t n = 0; n < 2; ++n) {
            o.f[5 + n] = -(a.c_rotor[n] * t.g[n]);
            for (std::size_t m = 0; m < 6; ++m) J[5 + n][kU[m]] = -(a.c_rotor[n] * t.dg[n][m]);
        }
        o.f[4] = -(t.Te - p.T_load);
        for (std::size_t m = 0; m < 5; ++m) J[4][kU[m]] = -t.dT[m];
    }
};

static_assert(Device<PmsmMna>);
static_assert(Device<InductionMachine>);
static_assert(kLawLayout<PmsmMna> && kLawLayout<InductionMachine>);

}  // namespace pulsim::device::laws
