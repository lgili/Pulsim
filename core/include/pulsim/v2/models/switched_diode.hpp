#pragma once

// =============================================================================
// Pulsim v2 — Layer 2: SwitchedDiode (binary-state, auto-commutating)
// =============================================================================
//
// `pulsim-v2-ideal-diode-auto-commutation` Phase 1.
//
// The binary-state PLECS-style ideal diode. Two states:
//   * ON  — conducting, `g_on` between anode and cathode.
//   * OFF — open,       `g_off` between anode and cathode.
//
// State transitions (decided per simulation step by Layer 5 V2):
//   OFF → ON   iff (v_anode − v_cathode) ≥ V_th
//   ON  → OFF  iff i_diode ≤ 0
//
// V_th defaults to 0 (perfectly ideal). Pass a non-zero value
// (e.g. 0.7) for a Si-behavioral diode.
//
// Topologically the diode IS a Switch — same combinatorial axis
// in Layer 4's cache, same stamp (g_on or g_off between
// terminals depending on the bit). The new bit is the per-step
// state-decision logic exposed by `decide_next_state`.
//
// SwitchedDiode is DISTINCT from the existing `models::IdealDiode`
// in `ideal_diode.hpp`, which is the smooth-blend nonlinear (AD-
// driven Newton) model. SwitchedDiode is the linear PWL variant
// that Layer 4's cache can swallow at one segment per state.

#include "pulsim/v2/numeric/concepts.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/topology/graph.hpp"

namespace pulsim::v2::models {

struct SwitchedDiode {
    struct Params {
        Real g_on;   // forward conductance (typ. 1e3 S)
        Real g_off;  // reverse conductance (typ. 1e-9 S)
        Real V_th;   // forward threshold (0 for ideal, 0.7 for Si)
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Switch;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;
    static constexpr bool is_dynamic = false;

    /// Decide the NEXT step's state given the current step's
    /// state and the just-computed (v_diode, i_diode).
    ///
    /// Transition rules (no hysteresis besides V_th):
    ///   OFF → ON   iff v_diode ≥ V_th
    ///   ON  → OFF  iff i_diode ≤ 0
    ///   Otherwise the state is unchanged.
    ///
    /// `v_diode` is `v_anode − v_cathode` (positive when forward-
    /// biased). `i_diode` is the current from anode to cathode
    /// (positive when conducting forward).
    [[nodiscard]] static bool decide_next_state(bool currently_on,
                                                  Real v_diode,
                                                  Real i_diode,
                                                  const Params& p) noexcept {
        if (currently_on) {
            return i_diode > Real{0};
        }
        return v_diode >= p.V_th;
    }

    /// Static current contract — diodes are stamped via
    /// `stamping::stamp_switch_fixed` like any other switch, not
    /// via `stamp_device`. Return 0 to satisfy the DeviceModel
    /// concept.
    template <numeric::FloatingPoint S>
    [[nodiscard]] static constexpr S current(const S* /*v*/,
                                              const Params& /*p*/) noexcept {
        return S{0};
    }
};

}  // namespace pulsim::v2::models
