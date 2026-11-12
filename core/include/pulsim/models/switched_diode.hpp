#pragma once

// =============================================================================
// Pulsim — Layer 2: SwitchedDiode (binary-state, auto-commutating)
// =============================================================================
//
// `pulsim-v2-ideal-diode-auto-commutation` Phase 1.
//
// The binary-state PLECS-style ideal diode. Two states:
//   * ON  — conducting, `g_on` between anode and cathode.
//   * OFF — open,       `g_off` between anode and cathode.
//
// PSIM-style: while ON the diode is `g_on` in series with a forward
// drop `V_f` (i = g_on·(v − V_f)); OFF is plain `g_off` (no offset).
//
// State transitions (decided per simulation step by Layer 5 V2):
//   OFF → ON   iff (v_anode − v_cathode) ≥ V_on = turn_on_voltage(p)
//   ON  → OFF  iff i_diode = g_on·(v − V_f) ≤ 0
//
// V_on = V_th when V_f == 0 (bit-identical to the pre-V_f law, negative
// V_th included), else max(V_th, V_f): turning on below V_f would turn on
// into negative current. V_th > V_f is a hysteresis band.
// Every engine decides through `decide` / `crossing_signal`; computing
// i = g·v by hand silently lets an ON diode conduct backwards.
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

#include <algorithm>

#include "pulsim/numeric/concepts.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::models {

struct SwitchedDiode {
    struct Params {
        Real g_on;   // forward conductance (typ. 1e3 S)
        Real g_off;  // reverse conductance (typ. 1e-9 S)
        Real V_th;   // turn-on threshold (effective max(V_th, V_f))
        Real V_f = Real{0};  // forward drop while ON (PSIM "Diode Voltage Drop")
    };

    [[nodiscard]] static Real turn_on_voltage(const Params& p) noexcept {
        return p.V_f == Real{0} ? p.V_th : std::max(p.V_th, p.V_f);
    }

    /// Anode→cathode branch current of the stamped state.
    [[nodiscard]] static Real branch_current(bool on, Real v,
                                             const Params& p) noexcept {
        return on ? p.g_on * (v - p.V_f) : p.g_off * v;
    }

    /// Next state from the solved terminal voltage v = v_a − v_k.
    [[nodiscard]] static bool decide(bool on, Real v,
                                     const Params& p) noexcept {
        return on ? branch_current(true, v, p) > Real{0}
                  : v >= turn_on_voltage(p);
    }

    /// Signal whose sign change marks the commutation from `on`.
    [[nodiscard]] static Real crossing_signal(bool on, Real v,
                                              const Params& p) noexcept {
        return on ? v - p.V_f : v - turn_on_voltage(p);
    }

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Switch;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;
    static constexpr bool is_dynamic = false;

    /// Decide the NEXT step's state given the current step's
    /// state and the just-computed (v_diode, i_diode).
    ///
    /// Transition rules (no hysteresis besides V_th − V_f):
    ///   OFF → ON   iff v_diode ≥ turn_on_voltage(p)
    ///   ON  → OFF  iff i_diode ≤ 0
    /// `i_diode` MUST come from `branch_current` (it carries V_f).
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
        return v_diode >= turn_on_voltage(p);
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

}  // namespace pulsim::models
