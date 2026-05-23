#pragma once

// =============================================================================
// Pulsim v2 — Layer 2: IdealDiode device model
// =============================================================================
//
// `pulsim-v2-device-models-ad-driven` Phase 5.
//
// Smooth-blend Norton-shifted ideal diode. The nonlinear "AD killer
// test" — if AD works here, it works for every device in Pulsim's
// catalogue.
//
// Model:
//
//   v_diode = v[0] - v[1]                          (anode - cathode)
//   alpha   = 1 / (1 + exp(-kappa · (v_diode - V_F0)))
//                                                  (sigmoid on/off)
//   i_on    = alpha · (v_diode - V_F0) / R_d       (Norton-shifted)
//   i_off   = (1 - alpha) · v_diode · G_off        (reverse leakage)
//   current = i_on + i_off
//
// Reverse-biased (v_diode ≪ V_F0): alpha ≈ 0 → current ≈ v_diode·G_off
//   (small leakage, typically pA range)
// Forward-biased (v_diode ≫ V_F0): alpha ≈ 1 → current ≈ (v_diode -
//   V_F0)/R_d (linear past V_F0, with slope 1/R_d)
// Threshold (v_diode ≈ V_F0): alpha ≈ 0.5, smooth blend
//
// `kappa` controls the sigmoid sharpness. Typical: 20-50/V. Too
// sharp → Newton struggles at the transition; too soft → off-state
// leakage bleeds into the on regime. The v1 audit chose kappa=20
// after the float-precision underflow study.
//
// The whole function is templated on `numeric::FloatingPoint S`.
// Instantiates for:
//   * Real (= double) — forward evaluation, used by tests + Layer 5
//   * ADRealN<2> — Jacobian via AD; Layer 3 reads the partials
//     and stamps the off-diagonal Newton terms.
//
// One function. Two instantiations. No duplication.

#include "pulsim/ad/ad_scalar.hpp"
#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

#include <cmath>

namespace pulsim::models {

struct IdealDiode {
    struct Params {
        Real V_F0  = 0.7;     // forward drop [V]
        Real R_d   = 0.01;    // forward slope resistance [Ω]
        Real G_off = 1e-9;    // reverse leakage conductance [S]
        Real kappa = 20.0;    // sigmoid sharpness [1/V]
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Nonlinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear = false;

    /// Current from terminal 0 (anode) to terminal 1 (cathode).
    /// Templated on any FloatingPoint type; ADL handles `exp` via
    /// `using std::exp` so `S = Real` and `S = ADRealN<N>` both
    /// dispatch correctly.
    template <numeric::FloatingPoint S>
    [[nodiscard]] static S current(const S* v, const Params& p) noexcept {
        using std::exp;
        const S v_diode = v[0] - v[1];
        const S delta = v_diode - p.V_F0;
        const S alpha = S{1} / (S{1} + exp(-p.kappa * delta));
        const S i_on  = alpha * delta * (S{1} / p.R_d);
        const S i_off = (S{1} - alpha) * v_diode * p.G_off;
        return i_on + i_off;
    }
};

}  // namespace pulsim::models
