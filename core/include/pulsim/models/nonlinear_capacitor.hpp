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

#include <cmath>
#include <format>
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

    static void validate(const Params& p) {
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
        return p.C0 / std::pow(Real{1} + u_of(p, v), p.m);
    }

    /// Stored charge Q(v) = ∫₀ᵛ C dv' [C], in closed form:
    ///   Q = C0·V0/(1−m)·[(1+v/V0)^(1−m) − 1]
    /// The m → 0 case reduces to C0·v, exactly.
    [[nodiscard]] static Real charge(const Params& p, Real v)
        noexcept {
        if (p.law == ChargeLawKind::Table) return p.table->charge(v);
        if (p.law == ChargeLawKind::VdmosCgd) return vdmos_charge_(p, v);
        const Real u = u_of(p, v);
        const Real e = Real{1} - p.m;
        return p.C0 * p.V0 / e
               * (std::pow(Real{1} + u, e) - Real{1});
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
