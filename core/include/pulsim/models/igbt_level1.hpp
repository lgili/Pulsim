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
// This is simpler than the MOSFET SH1 — the on-state I-V is
// LINEAR in V_CE (not quadratic), so there's no spurious
// negative-current root. Newton handles it cleanly.
//
// Physical behavior:
//   * V_GE < V_T:                I_C ≈ 0 (cutoff)
//   * V_GE > V_T, V_CE < V_CE_sat:  I_C < 0 (unphysical — but
//                                            in normal operation
//                                            V_CE >> V_CE_sat
//                                            during conduction)
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

        // Linear conduction with V_CE_sat offset.
        const S i_on = (V_CE - p.V_CE_sat) / p.R_CE_sat;

        return alpha_on * i_on;
    }
};

}  // namespace pulsim::models
