#pragma once

// =============================================================================
// Pulsim v2 — Layer 2: Capacitor device model (trap companion)
// =============================================================================
//
// `pulsim-v2-trapezoidal-companion` Phase 1.
//
// Linear two-terminal dynamic device:  i_C = C · dv_C/dt.
//
// At each timestep n → n+1, the trapezoidal rule yields:
//
//     v_{n+1} − v_n = (dt / 2C) · (i_n + i_{n+1})
//
// Solving for i_{n+1}:
//
//     i_{n+1}  =  (2C/dt) · v_{n+1}  −  [(2C/dt) · v_n + i_n]
//              =  G_eq   · v_{n+1}   −  I_hist
//
//     G_eq    = 2C / dt
//     I_hist  = G_eq · v_n + i_n
//
// G_eq goes into the MNA matrix at assembly time (same 4-entry
// conductance block as a resistor with G = G_eq).
//
// I_hist depends on the previous-step solution — Layer 5's
// HistoryState tracks it and contributes to b_extra at solve
// time, NOT during assembly.
//
// Terminal 0 is the "positive" terminal (the from-side). Current
// flows from terminal 0 to terminal 1 when i_C > 0, i.e. when v_C
// is increasing.

#include "pulsim/v2/numeric/concepts.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/topology/graph.hpp"

namespace pulsim::v2::models {

struct Capacitor {
    struct Params {
        Real C;     // farads
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::PassiveLinear;
    static constexpr Size num_terminals = 2;
    static constexpr bool is_linear  = true;   // linear in (v, i)
    static constexpr bool is_dynamic = true;   // needs companion stamp

    /// Norton-equivalent conductance for the trap companion.
    /// Stamped into the MNA matrix as a resistor-like
    /// 4-entry block (G_eq, −G_eq, −G_eq, G_eq) at the
    /// (from, to) coordinates.
    [[nodiscard]] static Real g_eq(Real dt, const Params& p) noexcept {
        return Real{2} * p.C / dt;
    }

    /// History term for step n+1 given previous-step state
    /// (v_n, i_n). Contributes to b_extra at solve time as:
    ///   b_extra(from) += −I_hist
    ///   b_extra(to)   += +I_hist
    /// (which produces a current-source equivalent of I_hist
    /// flowing from terminal 0 to terminal 1, parallel to G_eq).
    [[nodiscard]] static Real history_term(Real v_prev, Real i_prev,
                                            Real dt,
                                            const Params& p) noexcept {
        return g_eq(dt, p) * v_prev + i_prev;
    }

    /// Static current contract — capacitors don't fit the "current
    /// as a function of voltage" model; companion stamping handles
    /// them separately. Returns 0 to satisfy the DeviceModel
    /// concept.
    template <numeric::FloatingPoint S>
    [[nodiscard]] static constexpr S current(const S* /*v*/,
                                              const Params& /*p*/) noexcept {
        return S{0};
    }
};

}  // namespace pulsim::v2::models
