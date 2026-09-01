#pragma once

// =============================================================================
// Pulsim — Layer 2 V14: IGBT Level 1 (nonlinear)
// =============================================================================
//
// 3-terminal IGBT model with linear-conduction physics:
//
//   I_C = α(V_GE) · (V_CE − V_CE_sat) / R_CE_sat
//
// where:
//   V_GE = V_gate − V_emitter        (gate-emitter)
//   V_CE = V_collector − V_emitter   (collector-emitter)
//   α(V_GE) = 1/(1 + exp(−κ·(V_GE − V_T)))   (cutoff sigmoid)
//
//   V_CE_sat = collector-emitter saturation voltage (the
//              "knee" — IGBT doesn't conduct until V_CE
//              exceeds this), typically 1–2.5 V
//   R_CE_sat = on-state slope resistance, typically 10–100 mΩ
//   V_T      = gate threshold voltage, typically 4–6 V
//   κ        = cutoff sigmoid sharpness, typical 5–20
//
// AN IGBT CANNOT CONDUCT IN REVERSE (v2.0, audit C.1). It is a
// minority-carrier device: there is no channel to run backwards
// the way a MOSFET's does, and the collector junction blocks. The
// law above does not know that — below the knee it simply goes
// negative, and it does so with the full on-state slope:
//
//     V_CE (V)     I_C (A), gate at 15 V, V_CE_sat = 1.5, R = 50 mΩ
//       -10          -230        <- 230 A backwards through a
//        -5          -130           device that physically blocks
//         0           -30        <- both terminals shorted, and it
//       1.5             0           still sources 30 A from nothing
//         5           +70        <- the only correct row
//
// This is not a corner case: freewheeling an inductive load is
// what the low-side device does every switching cycle in every
// voltage-source inverter, and the current belongs to the
// anti-parallel FWD. The header used to wave the region off as
// "in normal operation V_CE >> V_CE_sat during conduction" — but
// during freewheeling V_CE is NEGATIVE, which is precisely when
// the model is wrong.
//
// The clamp is a smooth max, so Newton keeps a continuous
// derivative:
//
//     I_C = α(V_GE) · smoothmax0(V_CE − V_CE_sat) / R_CE_sat
//     smoothmax0(x) = (x + sqrt(x² + v_knee²)) / 2
//
// which is C^∞, strictly monotone and strictly positive, tends to
// x for x >> v_knee and to 0 for x << −v_knee.
//
// `v_knee` is set from the ACCURACY side, not the smoothing side.
// Widening it rounds the corner more but costs forward accuracy
// exactly where a lightly-loaded device sits — measured relative
// error on (V_CE − V_CE_sat):
//
//   v_knee    at 0.112 V    at 1.5 V    at 3.5 V    I_C at V_CE=0
//    0.1       17 %         0.11 %      0.02 %       3.3e-2 A
//    0.01      0.2 %        0.001 %     0.0002 %     3.3e-4 A
//
// so 0.01 V blocks just as well (30 A of reverse conduction
// becomes a third of a milliamp) while leaving conduction alone.
// 0.1 V would have moved an existing 2.24 A operating point by
// 17 %, which is why it is not the default.
//
// CONSEQUENCE FOR EXISTING CIRCUITS. A circuit that was silently
// freewheeling through the IGBT now has no path for that current,
// and will fail to converge instead of reporting a wrong answer.
// That is the intended trade. Give the device its anti-parallel
// diode — `add_igbt_level1(..., with_fwd=true)` — which is what
// the real module has co-packaged.
//
// Physical behavior:
//   * V_GE < V_T:                   I_C ≈ 0 (cutoff)
//   * V_GE > V_T, V_CE < V_CE_sat:  I_C → 0 (blocking; the FWD
//                                            takes the current)
//   * V_GE > V_T, V_CE > V_CE_sat:  I_C = (V_CE−V_CE_sat)/R
//                                       (active conduction)
//
// Architecture: same as MosfetLevel1 — 3-terminal nonlinear
// device with collector/emitter/gate node references.
// `evaluate_current_and_jacobian` returns the 3-element AD
// gradient (∂I/∂V_collector, ∂I/∂V_emitter, ∂I/∂V_gate).

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/logistic.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>

namespace pulsim::models {

struct IgbtLevel1 {
    struct Params {
        Real V_CE_sat = Real{1.5};   // [V] saturation voltage
        Real R_CE_sat = Real{0.05};  // [Ω] on-state slope R
        Real V_T      = Real{5.0};   // [V] gate threshold
        Real kappa    = Real{10.0};  // [1/V] sigmoid sharpness
        /// [V] width of the collector knee. Rounds the corner at
        /// V_CE = V_CE_sat and enforces I_C >= 0 below it. Also
        /// the model's only non-monotone-free knob: it must stay
        /// positive, and 0 would restore reverse conduction.
        Real v_knee   = Real{0.01};
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    static constexpr Size num_terminals = 3;  // collector, emitter, gate
    static constexpr bool is_linear = false;

    /// Collector current. Terminal ordering:
    ///   v[0] = V(collector)
    ///   v[1] = V(emitter)
    ///   v[2] = V(gate)
    template <numeric::FloatingPoint S>
    [[nodiscard]] static S current(
        const S* v, const Params& p) noexcept {
        using std::exp;
        // Branch voltages.
        const S V_GE = v[2] - v[1];
        const S V_CE = v[0] - v[1];

        // Cutoff sigmoid: α=0 below threshold, α=1 above.
        const S alpha_on =
            numeric::logistic(S{p.kappa * (V_GE - p.V_T)});

        // Linear conduction above the knee, blocking below it.
        // smoothmax0(x) = (x + sqrt(x² + v_knee²))/2 — see the
        // header: an IGBT has no reverse channel, so the raw
        // (V_CE − V_CE_sat) form conducting backwards was wrong
        // by the whole load current during freewheeling.
        using std::sqrt;
        const S x  = V_CE - p.V_CE_sat;
        const S eps2 = S{p.v_knee * p.v_knee};
        const S i_on = (x + sqrt(x * x + eps2))
                       / (Real{2} * p.R_CE_sat);

        return alpha_on * i_on;
    }
};

}  // namespace pulsim::models
