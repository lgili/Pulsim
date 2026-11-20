#pragma once

// =============================================================================
// Pulsim — charge-based nonlinear capacitor (a MOSFET's Coss)
// =============================================================================
//
// v2.0 Phase 4, audit C.1 ("Semicondutores nível-2 na solução
// elétrica", crítico). A grep for coss|qrr|tail|miller over the
// kernel returned nothing: every device was static I-V, so the
// resonant transition simply does not exist in the waveforms and
// ZVS/ZCS of an LLC, DAB or PSFB was unsimulable.
//
// WHY A LINEAR CAPACITOR IS NOT A SUBSTITUTE. What decides ZVS is
// the CHARGE that has to be moved, Q(V) = ∫C(v)dv, not the
// small-signal C at the operating point that a datasheet table
// quotes. For a planar junction with
//
//     C(v) = C0 / (1 + v/V0)^m ,     m ≈ 0.5
//
// at C0 = 2 nF, V0 = 25 V, V = 400 V the datasheet value is
// 485 pF while the charge-equivalent is 781 pF — 1.61x. Sized
// from the datasheet number, a 32 ns dead time reads as a clean
// ZVS transition; charge-accurately the node is still at 209 V
// when the switch turns on, which is hard switching and 17 µJ per
// edge (3.4 W at 100 kHz) that the simulation reported as zero.
//
// THE MODEL. Charge is the state, so the trapezoidal companion is
// written on Q rather than on v:
//
//     i_{n+1} = (2/h)·(Q(v_{n+1}) − Q(v_n)) − i_n
//
// which Newton stamps as a voltage-dependent conductance
// 2C(v)/h plus a current source — C(v) = dQ/dv falls out of the
// same closed form, so no differentiation is needed at runtime.
// Writing it on Q instead of on C·v is the whole point: it
// conserves charge exactly whatever C(v) does, which is what
// makes the dead-time answer trustworthy.

#include "pulsim/models/charge_table.hpp"
#include "pulsim/numeric/types.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace pulsim::models {

/// Which charge law a `NonlinearCapacitor` carries (add-power-mosfet-
/// datasheet-device). All three are stamped through the same charge-form
/// companion, so charge is conserved whatever C(v) does.
enum class ChargeLawKind : int {
    Junction = 0,  //!< C0 / (1 + v/V0)^m — the original law, unchanged
    Table    = 1,  //!< a `ChargeTable` (datasheet curve)
    VdmosCgd = 2,  //!< SPICE VDMOS gate-drain law (CGDMAX, CGDMIN, A)
    Tanh     = 3,  //!< odd tanh saturation (a saturable permeance)
};

/// Junction-law nonlinear capacitor: C(v) = C0 / (1 + v/V0)^m.
///
/// `m = 0` is an ordinary linear capacitor, and the formulas below
/// stay exact there — useful as a self-check and as a way to ask
/// for "same device, linear Coss" in an A/B.
///
/// Two further laws are selected by `Params::law` (the junction law is
/// the default and is evaluated by exactly the code it always was):
///
///   * `Table` — a `ChargeTable`, shared and immutable.
///   * `VdmosCgd` — ngspice's VDMOS C_gd, as a function of u = v (the
///     branch voltage, gate minus drain when the capacitor is placed
///     gate → drain). With s = (CGDMAX − CGDMIN)/(1 + π/2),
///     y = CGDMAX − s:
///         u > 0:  C = y + s·tanh(a·u),  Q = y·u + (s/a)·ln cosh(a·u)
///         u ≤ 0:  C = y + s·atan(a·u),
///                 Q = y·u + s·[u·atan(a·u) − ln(1 + a²u²)/(2a)]
///     C → CGDMAX for u → +∞ and → CGDMIN for u → −∞; C is C¹ at 0, so
///     Q is C². Matched against ngspice 45.2 at ten bias points to
///     1e-9 (probe `openspec/changes/add-power-mosfet-datasheet-device/
///     probes/cgd.cir`). ln cosh is evaluated as
///     |x| + log1p(exp(−2|x|)) − ln 2, which holds 1e-16 relative where
///     the naive log(cosh) overflows or cancels; for u ≤ 0 Q is evaluated
///     as u·C(u) − s·ln(1+a²u²)/(2a) and C below a·u = −1 from CGDMIN
///     (atan(x) = −π/2 − atan(1/x)), so Q stays monotone at 1 pV spacing
///     at −400 V where the expanded form cancels ten-fold.
///
/// add-permeance-magnetic-domain adds `Tanh`, the odd saturation law of
/// a saturable permeance (`add_saturable_permeance`), on the VDMOS fields
/// (`cgd_max` = C_max, `cgd_min` = C_min) and `V0`:
///         Q(v) = C_min·v + (C_max − C_min)·V0·tanh(v/V0)
///         C(v) = C_min + (C_max − C_min)·sech²(v/V0)
/// C^∞ and odd, C(0) = C_max, C → C_min from above, so Q is strictly
/// increasing. For a core segment of area A and length l with the
/// `GappedCore` material law B = μ₀(H + M_s·tanh(H/H₀)), v = F (the MMF),
/// Q = Φ: C_min = μ₀A/l, C_max = μ_r·μ₀A/l, V0 = l·H₀.
struct NonlinearCapacitor {
    struct Params {
        Real C0 = Real{1e-12};   //!< zero-bias capacitance [F]
        Real V0 = Real{1};       //!< junction potential [V]
        Real m  = Real{0.5};     //!< grading coefficient
        /// Reverse-bias floor. Real devices do not have infinite
        /// capacitance at v → −V0; clamping the argument keeps the
        /// model finite if a transient rings below zero.
        Real v_floor = Real{-0.9};
        /// Law selector; the fields below are read only by their law.
        ChargeLawKind law = ChargeLawKind::Junction;
        std::shared_ptr<const ChargeTable> table;  //!< `Table`
        Real cgd_max = Real{0};  //!< `VdmosCgd`: CGDMAX [F]
        Real cgd_min = Real{0};  //!< `VdmosCgd`: CGDMIN [F]
        Real a       = Real{0};  //!< `VdmosCgd`: A [1/V]
    };

    /// Junction-law params (the default law).
    [[nodiscard]] static Params junction(Real C0, Real V0, Real m = Real{0.5},
                                         Real v_floor = Real{-0.9}) {
        return Params{.C0 = C0, .V0 = V0, .m = m, .v_floor = v_floor};
    }
    /// Tanh-law params: Q = c_min·v + (c_max − c_min)·v0·tanh(v/v0).
    [[nodiscard]] static Params tanh_law(Real c_max, Real c_min, Real v0) {
        Params p;
        p.law = ChargeLawKind::Tanh;
        p.cgd_max = c_max;
        p.cgd_min = c_min;
        p.V0 = v0;
        return p;
    }
    /// Table-law params.
    [[nodiscard]] static Params from_table(std::shared_ptr<const ChargeTable> t) {
        Params p;
        p.law = ChargeLawKind::Table;
        p.table = std::move(t);
        return p;
    }
    /// VDMOS gate-drain-law params.
    [[nodiscard]] static Params vdmos_cgd(Real cgd_max, Real cgd_min, Real a) {
        Params p;
        p.law = ChargeLawKind::VdmosCgd;
        p.cgd_max = cgd_max;
        p.cgd_min = cgd_min;
        p.a = a;
        return p;
    }
    [[nodiscard]] static const char* law_name(const Params& p) noexcept {
        switch (p.law) {
        case ChargeLawKind::Table:    return "table";
        case ChargeLawKind::VdmosCgd: return "vdmos_cgd";
        case ChargeLawKind::Tanh:     return "tanh";
        case ChargeLawKind::Junction: break;
        }
        return "junction";
    }

    /// ln cosh(x) without overflow or cancellation.
    [[nodiscard]] static Real log_cosh_(Real x) noexcept {
        const Real ax = std::abs(x);
        return ax + std::log1p(std::exp(Real{-2} * ax))
               - std::numbers::ln2_v<Real>;
    }
    [[nodiscard]] static Real vdmos_s_(const Params& p) noexcept {
        return (p.cgd_max - p.cgd_min)
               / (Real{1} + std::numbers::pi_v<Real> / Real{2});
    }
    [[nodiscard]] static Real vdmos_capacitance_(const Params& p, Real u) noexcept {
        const Real s = vdmos_s_(p);
        const Real x = p.a * u;
        if (x > Real{0}) return p.cgd_max - s + s * std::tanh(x);
        // atan(x) = −π/2 − atan(1/x) for x < 0: deep in the off state
        // y + s·atan(x) cancels to CGDMIN, so write it from CGDMIN.
        if (x < Real{-1}) return p.cgd_min - s * std::atan(Real{1} / x);
        return p.cgd_max - s + s * std::atan(x);
    }
    [[nodiscard]] static Real vdmos_charge_(const Params& p, Real u) noexcept {
        const Real s = vdmos_s_(p);
        const Real x = p.a * u;
        if (x > Real{0}) return (p.cgd_max - s) * u + (s / p.a) * log_cosh_(x);
        // y·u + s·[u·atan(x) − ln(1+x²)/(2a)] = u·C(u) − s·ln(1+x²)/(2a),
        // without the O(10×) cancellation of the expanded form at −400 V.
        return u * vdmos_capacitance_(p, u) - s * std::log1p(x * x) / (Real{2} * p.a);
    }

    /// sech²(x) as 4e^{−2|x|}/(1 + e^{−2|x|})²: no overflow of cosh and
    /// no 1 − tanh² cancellation in the tail.
    [[nodiscard]] static Real sech2_(Real x) noexcept {
        const Real e = std::exp(Real{-2} * std::abs(x));
        const Real d = Real{1} + e;
        return Real{4} * e / (d * d);
    }

    static void validate(const Params& p) {
        if (p.law == ChargeLawKind::Tanh) {
            if (!(p.cgd_min > Real{0}) || !std::isfinite(p.cgd_min)) {
                throw std::invalid_argument(std::format(
                    "NonlinearCapacitor: the tanh law needs C_min > 0 (got {})",
                    p.cgd_min));
            }
            if (!(p.cgd_max > p.cgd_min) || !std::isfinite(p.cgd_max)) {
                throw std::invalid_argument(std::format(
                    "NonlinearCapacitor: the tanh law needs C_max > C_min "
                    "(got C_max = {}, C_min = {})", p.cgd_max, p.cgd_min));
            }
            if (!(p.V0 > Real{0}) || !std::isfinite(p.V0)) {
                throw std::invalid_argument(std::format(
                    "NonlinearCapacitor: the tanh law needs V0 > 0 (got {})", p.V0));
            }
            return;
        }
        if (p.law == ChargeLawKind::Table) {
            if (!p.table || p.table->size() < 3) {
                throw std::invalid_argument(
                    "NonlinearCapacitor: the table law needs a ChargeTable "
                    "(none was given)");
            }
            return;
        }
        if (p.law == ChargeLawKind::VdmosCgd) {
            // CGDMIN = 0 is ngspice's default and still gives C > 0 at
            // every finite bias (C → CGDMIN only as v_gd → −∞).
            if (!(p.cgd_min >= Real{0}) || !std::isfinite(p.cgd_min)) {
                throw std::invalid_argument(std::format(
                    "NonlinearCapacitor: vdmos_cgd needs CGDMIN >= 0 (got {})",
                    p.cgd_min));
            }
            if (!(p.cgd_max > p.cgd_min) || !std::isfinite(p.cgd_max)) {
                throw std::invalid_argument(std::format(
                    "NonlinearCapacitor: vdmos_cgd needs CGDMAX > CGDMIN "
                    "(got CGDMAX = {}, CGDMIN = {})", p.cgd_max, p.cgd_min));
            }
            if (!(p.a > Real{0}) || !std::isfinite(p.a)) {
                throw std::invalid_argument(std::format(
                    "NonlinearCapacitor: vdmos_cgd needs A > 0 (got {})", p.a));
            }
            return;
        }
        if (!(p.C0 > Real{0})) {
            throw std::invalid_argument(
                "NonlinearCapacitor: C0 must be positive");
        }
        if (!(p.V0 > Real{0})) {
            throw std::invalid_argument(
                "NonlinearCapacitor: V0 must be positive");
        }
        if (p.m < Real{0} || p.m >= Real{1}) {
            throw std::invalid_argument(
                "NonlinearCapacitor: m must be in [0, 1) — the "
                "charge integral diverges at m = 1 and the model "
                "is not physical above it");
        }
        if (!(p.v_floor > Real{-1})) {
            throw std::invalid_argument(
                "NonlinearCapacitor: v_floor must exceed -1 (it is "
                "the clamp on v/V0, and -1 is the pole)");
        }
    }

    /// u = clamp(v/V0), the dimensionless bias the laws use.
    [[nodiscard]] static Real u_of(const Params& p, Real v)
        noexcept {
        const Real u = v / p.V0;
        return u < p.v_floor ? p.v_floor : u;
    }

    /// Small-signal capacitance dQ/dv at `v` [F].
    [[nodiscard]] static Real capacitance(const Params& p, Real v)
        noexcept {
        if (p.law == ChargeLawKind::Table) return p.table->capacitance(v);
        if (p.law == ChargeLawKind::VdmosCgd) return vdmos_capacitance_(p, v);
        if (p.law == ChargeLawKind::Tanh) {
            return p.cgd_min + (p.cgd_max - p.cgd_min) * sech2_(v / p.V0);
        }
        return p.C0 / std::pow(Real{1} + u_of(p, v), p.m);
    }

    /// Stored charge Q(v) = ∫₀ᵛ C dv' [C], in closed form:
    ///   Q = C0·V0/(1−m)·[(1+v/V0)^(1−m) − 1]
    /// The m → 0 case reduces to C0·v, exactly.
    [[nodiscard]] static Real charge(const Params& p, Real v)
        noexcept {
        if (p.law == ChargeLawKind::Table) return p.table->charge(v);
        if (p.law == ChargeLawKind::VdmosCgd) return vdmos_charge_(p, v);
        if (p.law == ChargeLawKind::Tanh) {
            return p.cgd_min * v + (p.cgd_max - p.cgd_min) * p.V0 * std::tanh(v / p.V0);
        }
        const Real u = u_of(p, v);
        const Real e = Real{1} - p.m;
        return p.C0 * p.V0 / e
               * (std::pow(Real{1} + u, e) - Real{1});
    }

    /// The voltage at charge q (Q is non-decreasing in v): bracketed
    /// Newton from v_guess. NaN when no voltage reaches q (below a
    /// clamped floor, where Q is flat).
    [[nodiscard]] static Real voltage_of_charge(const Params& p, Real q,
                                                Real v_guess) noexcept {
        const Real step0 = Real{1e-3} * std::max(std::abs(v_guess), std::abs(p.V0));
        Real lo = v_guess, hi = v_guess, step = step0;
        for (int k = 0; charge(p, lo) > q; ++k, step *= 2) {
            if (k == 200) return std::numeric_limits<Real>::quiet_NaN();
            lo -= step;
        }
        step = step0;
        for (int k = 0; charge(p, hi) < q; ++k, step *= 2) {
            if (k == 200) return std::numeric_limits<Real>::quiet_NaN();
            hi += step;
        }
        Real v = std::clamp(v_guess, lo, hi);
        for (int k = 0; k < 100; ++k) {
            const Real f = charge(p, v) - q;
            if (f == Real{0}) return v;
            (f > Real{0} ? hi : lo) = v;
            Real vn = v - f / capacitance(p, v);
            if (!(vn > lo && vn < hi)) vn = Real{0.5} * (lo + hi);
            if (std::abs(vn - v) <= Real{1e-15} * (Real{1} + std::abs(v))) return vn;
            v = vn;
        }
        return v;
    }

    /// The energy a charge-conserving jump a → b dissipates in the path
    /// that carries it: ∫ₐᵇ (b − v)·C(v) dv = ∫ₐᵇ (Q(v) − Q(a)) dv
    /// (½·C·(b − a)² for a linear law). Closed form for the junction law,
    /// Gauss–Legendre (5 points × 32 panels) for the others.
    [[nodiscard]] static Real jump_energy(const Params& p, Real a, Real b) noexcept {
        const Real qa = charge(p, a);
        if (p.law == ChargeLawKind::Junction && p.m != Real{2} &&
            u_of(p, a) == a / p.V0 && u_of(p, b) == b / p.V0) {
            const Real e = Real{1} - p.m, e2 = Real{2} - p.m;
            auto F = [&](Real v) {   // ∫Q dv
                return p.C0 * p.V0 / e *
                       (p.V0 * std::pow(Real{1} + v / p.V0, e2) / e2 - v);
            };
            return F(b) - F(a) - qa * (b - a);
        }
        constexpr Real xg[5] = {0.0, -0.5384693101056831, 0.5384693101056831,
                                -0.9061798459386640, 0.9061798459386640};
        constexpr Real wg[5] = {0.5688888888888889, 0.4786286704993665,
                                0.4786286704993665, 0.2369268850561891,
                                0.2369268850561891};
        constexpr int n = 32;
        const Real hp = (b - a) / n;
        Real s = 0;
        for (int j = 0; j < n; ++j) {
            const Real c = a + (j + Real{0.5}) * hp;
            for (int g = 0; g < 5; ++g) s += wg[g] * (charge(p, c + Real{0.5} * hp * xg[g]) - qa);
        }
        return Real{0.5} * hp * s;
    }

    /// The CHARGE-EQUIVALENT capacitance over [0, V]: Q(V)/V. This
    /// is the number that predicts a dead time; a datasheet's
    /// C(V) is not.
    [[nodiscard]] static Real charge_equivalent(const Params& p,
                                                 Real V) noexcept {
        if (std::abs(V) < Real{1e-30}) {
            return p.law == ChargeLawKind::Junction ? p.C0 : capacitance(p, Real{0});
        }
        return charge(p, V) / V;
    }
};

}  // namespace pulsim::models
