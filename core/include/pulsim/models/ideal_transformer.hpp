#pragma once

// =============================================================================
// Pulsim — Phase 4 C.4: ideal transformer (N-winding-ready two-port)
// =============================================================================
//
// The one magnetics element the kernel did not have. `add_transformer`
// is a pair of COUPLED INDUCTORS, v_i = Σ_j M_ij di_j/dt — linear by
// construction, with the magnetising inductance folded into L_p and
// M. There is no place in that model for a core, so nothing in it
// can saturate: pushed to 3× B_sat the example flyback still returns
// a tidy output voltage (measured: 21.6 A peak primary, 1.14 T
// implied on an N87 core that saturates at 0.35 T, not a warning).
//
// The T-model separates what is linear from what is not:
//
//     p_from ──[ L_leak,p ]──┬──● IDEAL n ●──[ L_leak,s ]── s_from
//                          [L_m]
//     p_to   ────────────────┴──●         ●──────────────── s_to
//
// Leakage inductances are plain linear inductors. The magnetising
// branch L_m is a flux device λ(i) — the saturable inductor's
// machinery, with a law generated from core geometry — and it is the
// ONLY nonlinear thing. What is left in the middle is this element:
//
//     v_s = n · v_p            n = N_s / N_p
//     i_p = −n · i_s           (power in = power out, exactly)
//
// with i_p, i_s the currents INTO the dotted (from) terminals. Note
// that it transforms DC — an ideal transformer has no frequency
// dependence at all; it is L_m, in parallel with the primary, that
// carries the DC short. That is physically right and it is why the
// DC operating point stamps this element unchanged.
//
// MNA: the SECONDARY is the branch (Source kind, own current unknown
// i_s, exactly like a VCVS with gain n sensing the primary nodes).
// The PRIMARY has no branch of its own — its current is determined —
// so it appears as a current-controlled current source −n·i_s
// injected between p_from and p_to. Four Jacobian entries a VCVS
// does not have; everything else is the VCVS stamp verbatim.

#include "pulsim/numeric/types.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::models {

struct IdealTransformer {
    struct Params {
        Real n = Real{1};   //!< turns ratio N_s / N_p (> 0)
    };

    static constexpr topology::BranchKind kind =
        topology::BranchKind::Source;
    static constexpr bool is_linear = true;
};

}  // namespace pulsim::models
