#pragma once

// =============================================================================
// Pulsim v2 — Layer 2 V13: MOSFET Shichman-Hodges Level 1
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
// REVERSE CONDUCTION CAVEAT: the raw triode polynomial
//   K·(2·V_OV·V_DS − V_DS²)
// is a DOWNWARD parabola in V_DS that goes negative for both
// V_DS < 0 and V_DS > 2·V_OV. The β sigmoid handles the
// upper boundary by blending to i_sat. The LOWER boundary
// (V_DS < 0) is NOT handled by the model — Newton can land
// on this spurious root during big linearisation steps when
// e.g. a series-L forces V_drain below V_source briefly.
//
// The fix is at the TOPOLOGY level: add an explicit anti-
// parallel body diode (anode = source, cathode = drain) in
// the user's circuit. Real MOSFETs have this physically; for
// SH1 it must be added explicitly. The diode clamps V_DS to
// approximately −V_F (small negative), keeping the MOSFET in
// its well-behaved positive-V_DS regime.

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/numeric/concepts.hpp"
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

    /// Current from drain → source. Terminal ordering:
    ///   v[0] = V(drain)
    ///   v[1] = V(source)
    ///   v[2] = V(gate)
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

        // Sigmoid α: 0 when V_OV << 0 (cutoff), 1 when
        // V_OV >> 0 (on).
        const S alpha_on = S{1} /
            (S{1} + exp(-p.kappa * V_OV));

        // Sigmoid β: 1 when V_DS << V_OV (triode), 0 when
        // V_DS >> V_OV (saturation).
        const S beta_triode = S{1} /
            (S{1} + exp(-p.kappa * (V_OV - V_DS)));

        const S clm = S{1} + p.lambda * V_DS;

        const S i_triode = p.K *
            (Real{2} * V_OV * V_DS - V_DS * V_DS) * clm;
        const S i_sat    = p.K * V_OV * V_OV * clm;

        const S i_on = beta_triode * i_triode +
                       (S{1} - beta_triode) * i_sat;
        return alpha_on * i_on;
    }
};

}  // namespace pulsim::models
