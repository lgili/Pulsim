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
    template <numeric::FloatingPoint S>
    [[nodiscard]] static S forward_current_(
        const S& V_OV, const S& V_DS, const Params& p) noexcept {
        const S alpha_on =
            numeric::logistic(S{p.kappa * V_OV});
        const S beta_triode =
            numeric::logistic(S{p.kappa * (V_OV - V_DS)});
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
    template <numeric::FloatingPoint S>
    [[nodiscard]] static S current(
        const S* v, const Params& p) noexcept {
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
        if (V_DS < S{0}) {
            return -forward_current_<S>(V_OV - V_DS, -V_DS, p);
        }
        return forward_current_<S>(V_OV, V_DS, p);
    }

};

}  // namespace pulsim::models
