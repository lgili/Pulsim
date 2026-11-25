#pragma once

// =============================================================================
// Pulsim — Layer 2 V13: MOSFET Shichman-Hodges Level 1
// =============================================================================
//
// 3-terminal MOSFET model with the classic SH1 quadratic
// current law:
//
//   Cutoff     (V_OV <= 0):           I_D = 0
//   Triode     (V_OV > 0, V_DS < V_OV): I_D = K·(2·V_OV·V_DS − V_DS²)·(1+λ·V_DS)
//   Saturation (V_OV > 0, V_DS >= V_OV): I_D = K·V_OV²·(1+λ·V_DS)
//
// Where:
//   V_OV    = V_GS − V_T          (overdrive voltage)
//   V_GS    = V_gate − V_source
//   V_DS    = V_drain − V_source
//   K       = (μ·C_ox/2)·(W/L)    (transconductance parameter, A/V²)
//   V_T     = threshold voltage   (V)
//   λ       = channel-length modulation  (1/V), typically 0.01–0.05
//
// Unlike a 2-terminal device, the SH1 MOSFET depends on TWO
// branch voltages (V_GS and V_DS). The architecture: declare
// `num_terminals = 3` (drain, source, gate in that order) so
// the existing `evaluate_current_and_jacobian` returns a 3-
// element gradient (∂I/∂V_drain, ∂I/∂V_source, ∂I/∂V_gate).
// The Newton refresh function stamps the 6 Jacobian entries
// per device on the [drain, source] rows.
//
// Smoothing: SH1 piecewise transitions (cutoff↔triode↔sat) are
// C⁰ but only C¹ at the triode/saturation boundary — NOT at the
// cutoff/triode boundary (∂I/∂V_OV has a jump at V_OV=0). We
// blend the regions via sigmoid functions (analogous to v2's
// IdealDiode smooth-blend) so the whole I(V_GS,V_DS) surface is
// C¹-smooth and Newton converges robustly.
//
// `kappa` (1/V) controls sigmoid sharpness. Typical: 10–50.
// Too sharp → Newton struggles at the transition; too soft →
// sub-threshold leakage bleeds into the saturation regime.
//
// THIRD QUADRANT (v2.0, audit C.1). The raw triode polynomial
//   K·(2·V_OV·V_DS − V_DS²)
// is a DOWNWARD parabola in V_DS. The β sigmoid handles the
// upper turning point by blending to i_sat; the LOWER side used
// to be left alone, and the model evaluated the forward law at
// negative V_DS.
//
// That was not merely inaccurate, it was NON-MONOTONE. Measured
// on K = 50, V_T = 3, V_GS = 10: i rose to ~21 kA near −30 V,
// crossed ZERO at −50 V and dived after. So an inductor
// freewheeling through a gated-on device — ordinary synchronous
// rectification, the commonest thing a MOSFET does — had several
// operating points, and Newton settled on a far one: v(sw) =
// −63 V where the channel's own 1.43 mΩ gives −14 mV, reported
// as 544 W of loss with no warning.
//
// This header used to prescribe an anti-parallel body diode as
// the fix. IT DOES NOT WORK: with the body diode that same
// circuit landed at −63 V, without it at −50 V. (Add the body
// diode anyway — real devices have one, and it carries the
// current when the gate is OFF.)
//
// The actual fix is in the LAW. A MOSFET has no built-in drain
// and source: for V_DS < 0 the terminals swap roles, so
//   i(V_DS < 0) = −i_forward(V_OV − V_DS, −V_DS)
// where the overdrive is measured from the terminal now acting
// as the source. That is monotone everywhere, gives V_DS =
// −I·R_on in the third quadrant (synchronous rectification), and
// also reproduces false turn-on: with V_G = 0 and the drain
// pulled to −5 V the gate is +5 V above the new source and the
// channel forms, which the forward-only law could not see.

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/logistic.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>
#include <bit>
#include <cstdint>

namespace pulsim::models {

struct MosfetLevel1 {
    struct Params {
        Real K      = Real{1e-3};   // [A/V²] transconductance
        Real V_T    = Real{2.0};    // [V] threshold
        Real lambda = Real{0.02};   // [1/V] channel-length mod
        Real kappa  = Real{15.0};   // [1/V] sigmoid sharpness
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    static constexpr Size num_terminals = 3;  // drain, source, gate
    static constexpr bool is_linear = false;

    /// The one-directional law with the terminal roles as given.
    /// `current` below calls it once per quadrant.
    template <numeric::FloatingPoint S, class P = Params>
    [[nodiscard]] static S forward_current_(
        const S& V_OV, const S& V_DS, const P& p) noexcept {
        return forward_with_<S>(V_OV, V_DS, numeric::logistic(S{p.kappa * V_OV}),
                                numeric::logistic(S{p.kappa * (V_OV - V_DS)}), p);
    }

    /// `forward_current_` with α = σ(κ·V_OV) and β = σ(κ·(V_OV − V_DS))
    /// given (forward_c1_ shares their exponentials; same operations).
    template <numeric::FloatingPoint S, class P>
    [[nodiscard]] static S forward_with_(const S& V_OV, const S& V_DS, const S& alpha_on,
                                         const S& beta_triode, const P& p) noexcept {
        const S clm = S{1} + p.lambda * V_DS;
        const S i_triode = p.K *
            (Real{2} * V_OV * V_DS - V_DS * V_DS) * clm;
        const S i_sat    = p.K * V_OV * V_OV * clm;
        const S i_on = beta_triode * i_triode +
                       (S{1} - beta_triode) * i_sat;
        return alpha_on * i_on;
    }

    /// Current from drain → source. Terminal ordering:
    ///   v[0] = V(drain), v[1] = V(source), v[2] = V(gate).
    ///
    /// Templated on FloatingPoint S — instantiates for `Real`
    /// (forward) and `ADRealN<3>` (Newton Jacobian).
    // `P`: the parameter struct, or a law's `Params<S>` with the same
    // member names (device/laws.hpp), so ∂i/∂p comes from this body too.
    template <numeric::FloatingPoint S, class P = Params>
    [[nodiscard]] static S current(
        const S* v, const P& p) noexcept {
        using std::exp;
        // Branch voltages.
        const S V_GS = v[2] - v[1];
        const S V_DS = v[0] - v[1];
        const S V_OV = V_GS - p.V_T;

        // THIRD-QUADRANT SYMMETRY (audit C.1, "3o quadrante
        // simetrizado — retificação síncrona!"). A MOSFET has no
        // built-in drain/source: with V_DS < 0 the terminals
        // simply swap roles, so the current is the mirror of the
        // forward law evaluated from the other end, where the
        // overdrive is V_G − V_D = V_OV − V_DS.
        //
        // Evaluating the FORWARD polynomial at negative V_DS
        // instead is not merely inaccurate, it is non-monotone:
        // measured on K = 50, V_T = 3, V_GS = 10, i(V_DS) rose to
        // ~21 kA near −30 V, crossed ZERO at −50 V and dived
        // after. An inductor freewheeling through a gated-on
        // device — ordinary synchronous rectification — therefore
        // had several solutions and Newton settled on one of the
        // far ones: v(sw) = −63 V where the channel's own 1.43 mΩ
        // gives −14 mV, reported as 544 W of loss with no warning.
        // The anti-parallel body diode the header used to
        // prescribe as the fix does not prevent it (that case
        // landed at −63 V; without the diode, −50 V).
        //
        // C¹ AND PASSIVE AT V_DS = 0 (unified-device-interface #44). The β
        // blend leaks forward(V_OV, 0) = α(1 − β)·K·V_OV² = L(V_OV) ≠ 0 at
        // V_DS = 0 (peak 0.439·K/κ² at V_OV = 2.4/κ; 2.8 mA for K = 1.43,
        // κ = 15), so the plain mirror jumped by 2L there and both engines
        // aborted at the zero crossing of a current through a partly-on
        // channel (add-power-mosfet-datasheet-device probes/vds_zero_jump.py).
        // Each quadrant runs forward_c1_ = forward − L·e^{−κ|V_DS|}: zero at
        // V_DS = 0 (no current at zero voltage), equal one-sided derivatives
        // there (C¹), and the model's own bits wherever the subtracted term
        // is below half an ulp (fully on, or |V_DS| ≫ 1/κ).
        if (V_DS < S{0}) {
            return -forward_c1_<S>(V_OV - V_DS, -V_DS, p);
        }
        return forward_c1_<S>(V_OV, V_DS, p);
    }

    /// h(a, d) = forward(a, d) − forward(a, 0)·e^{−κd} for d ≥ 0 (a = V_OV,
    /// d = V_DS). h(·, 0) = 0; ∂h/∂d(a, 0) = ∂f/∂d(a, 0) + κ·f(a, 0) on both
    /// sides of the mirror, ∂h/∂V_GS(a, 0) = 0, so the mirrored law is C¹.
    /// For κd ≤ 1 it is written without the subtraction (expm1, no
    /// cancellation near d = 0):
    ///   α·K·{λd·[a² − β(u)u²] + a²(1−α)·E + α·d·(2a − d) + u²·α·(1−β(u))·E},
    ///   u = a − d, E = 1 − e^{−κd}.
    template <numeric::FloatingPoint S, class P = Params>
    [[nodiscard]] static S forward_c1_(const S& a, const S& d, const P& p) noexcept {
        using std::exp; using std::expm1;
        // #47: the same operations as the separate evaluations of 4aa3b15e
        // (forward(a, d), forward(a, 0), four `numeric::logistic` calls), so
        // the same bits; only shared and skipped work differs. σ(z), σ(−z)
        // come from one exponential; the exits are decided on Real values
        // (an AD value is the Real computation's value), and the AD f and
        // forward(a, 0) are formed only on the exits that return them.
        const S u = a - d;
        const LogisticPair<S> pa(S(p.kappa * a)), pu(S(p.kappa * u));
        const S alpha = pa.pos(), beta_u = pu.pos();
        const Real av = value_(a), dv = value_(d), alv = value_(alpha);
        const ValueParams pv{value_(p.K), value_(p.lambda)};
        const Real fv = forward_with_<Real>(av, dv, alv, value_(beta_u), pv);
        const S kd = p.kappa * d;
        if (leak_below_half_ulp_(av, value_(kd), fv, value_(p.kappa), pv.K)) {
            return forward_with_<S>(a, d, alpha, beta_u, p);
        }
        // forward(a, 0): β = σ(κ·(a − 0)) = α. κd ≤ 1 and |L|/4 above half
        // an ulp of f: e^{−κd} ≥ e^{−1} makes the computed leak exceed it
        // too, so the half-ulp exit cannot be taken and κd > 1 is not.
        const Real Lv = std::abs(forward_with_<Real>(av, Real{0}, alv, alv, pv));
        const bool active = !(kd > Real{1}) && Lv >= Real{0x1p-1000}
                            && Real{0.25} * Lv > Real{0x1p-54} * fv;
        if (!active) {
            const S f = forward_with_<S>(a, d, alpha, beta_u, p);
            const S leak = forward_with_<S>(a, S{0}, alpha, alpha, p) * exp(S(-kd));
            // below half an ulp of f: the model's bits, untouched
            if (leak <= Real{0x1p-54} * f && -leak <= Real{0x1p-54} * f) return f;
            if (kd > Real{1}) return f - leak;
        }
        const S one_minus_alpha = pa.neg(), one_minus_beta_u = pu.neg();
        const S E = -expm1(S(-kd));
        const S bracket = p.lambda * d * (a * a - beta_u * u * u)
                          + a * a * one_minus_alpha * E
                          + alpha * d * (Real{2} * a - d)
                          + u * u * alpha * one_minus_beta_u * E;
        return alpha * p.K * bracket;
    }

    /// Cheap sufficient test for the "leak ≤ 2⁻⁵⁴·f" exit of forward_c1_,
    /// without forward(a, 0) or exp (the default path's cost, #47). Bound
    /// on the computed leak: forward(a, 0) = α(1 − β)·K·a² with β = α at
    /// d = 0, and fl(α(1 − β)) ≤ 8·e^{−κ|a|} (for κa ≥ 0, 1 − fl(β) is exact
    /// and ≤ t + 2⁻⁵² ≤ 5t when nonzero, t = fl(e^{−κa}) > 2⁻⁵⁴; for κa < 0
    /// fl(α) ≤ e^{κa}(1 + 2u)), so leak ≤ 16·|K|·a²·e^{−s}, s = κ|a| + κd,
    /// and e^{−s} ≤ 2^{−n} with n = ⌊s·log₂e⌋ − 1. The test asks
    /// 64·|K|·a²·2^{−n} ≤ 2⁻⁵⁴·f: a factor 4 above the bound (rounding,
    /// a subnormal product) with f normal and ≥ 2⁻⁹⁶⁰, so whenever it
    /// passes the full code would also have returned f — bit-identical.
    [[nodiscard]] static bool leak_below_half_ulp_(Real av, Real kdv, Real fv, Real kappa,
                                                   Real K) noexcept {
        if (!(fv >= Real{0x1p-960} && fv <= Real{0x1.fffffffffffffp+1023})) return false;
        const Real sl = (std::abs(kappa * av) + kdv) * Real{1.4426950408889634};
        if (!(sl >= Real{2})) return false;
        const int n = static_cast<int>(sl < Real{1001} ? sl : Real{1001}) - 1;
        // 64·2^{−n}, n ∈ [1, 1000]: a normal double built from its exponent
        const Real scale = std::bit_cast<Real>(std::uint64_t(1023 + 6 - n) << 52);
        return std::abs(K) * av * av * scale <= Real{0x1p-54} * fv;
    }

    struct ValueParams { Real K, lambda; };

    template <class X>
    [[nodiscard]] static Real value_(const X& x) noexcept {
        if constexpr (requires { x.value(); }) return x.value();
        else return static_cast<Real>(x);
    }

    /// σ(z) and σ(−z) with `numeric::logistic`'s operations for each, from
    /// one exponential (both read exp(−|z|)), each formed on demand; z = ±0
    /// or NaN: the two calls.
    template <class S>
    struct LogisticPair {
        S z, e, den;
        int side;   // +1: z > 0, −1: z < 0, 0: neither
        explicit LogisticPair(const S& zz) noexcept : z(zz), side(0) {
            using std::exp;
            if (z > Real{0}) { e = exp(-z); den = S{1} + e; side = 1; }
            else if (z < Real{0}) { e = exp(z); den = S{1} + e; side = -1; }
        }
        [[nodiscard]] S pos() const noexcept {
            return side > 0 ? S{1} / den : side < 0 ? e / den : numeric::logistic(z);
        }
        [[nodiscard]] S neg() const noexcept {
            return side > 0 ? e / den : side < 0 ? S{1} / den : numeric::logistic(S(-z));
        }
    };

};

}  // namespace pulsim::models
