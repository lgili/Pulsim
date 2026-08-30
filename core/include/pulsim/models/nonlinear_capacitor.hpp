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

#include "pulsim/numeric/types.hpp"

#include <cmath>
#include <stdexcept>

namespace pulsim::models {

/// Junction-law nonlinear capacitor: C(v) = C0 / (1 + v/V0)^m.
///
/// `m = 0` is an ordinary linear capacitor, and the formulas below
/// stay exact there — useful as a self-check and as a way to ask
/// for "same device, linear Coss" in an A/B.
struct NonlinearCapacitor {
    struct Params {
        Real C0 = Real{1e-12};   //!< zero-bias capacitance [F]
        Real V0 = Real{1};       //!< junction potential [V]
        Real m  = Real{0.5};     //!< grading coefficient
        /// Reverse-bias floor. Real devices do not have infinite
        /// capacitance at v → −V0; clamping the argument keeps the
        /// model finite if a transient rings below zero.
        Real v_floor = Real{-0.9};
    };

    static void validate(const Params& p) {
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
        return p.C0 / std::pow(Real{1} + u_of(p, v), p.m);
    }

    /// Stored charge Q(v) = ∫₀ᵛ C dv' [C], in closed form:
    ///   Q = C0·V0/(1−m)·[(1+v/V0)^(1−m) − 1]
    /// The m → 0 case reduces to C0·v, exactly.
    [[nodiscard]] static Real charge(const Params& p, Real v)
        noexcept {
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
            return p.C0;
        }
        return charge(p, V) / V;
    }
};

}  // namespace pulsim::models
