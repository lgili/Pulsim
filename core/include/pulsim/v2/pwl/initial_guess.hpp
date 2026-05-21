#pragma once

// =============================================================================
// Pulsim v2 — Layer 4 V10: smart initial-guess helper
// =============================================================================
//
// `pulsim-v2-pseudo-transient` Phase 2 / 3.
//
// The κ=20 stiff sinusoidal rectifier from `x = 0` is THE
// motivating challenge from V4 → V9. After exhaustive
// exploration of Newton globalizations (line search, LM,
// κ-homotopy, V_F0-homotopy, combined homotopy, and PTC),
// the empirical finding is unambiguous:
//
//   ▸ The κ=20 smooth-blend sigmoid creates multiple
//     operating-point branches near v_diode = V_F0.
//   ▸ NO single-shot Newton variant (line search, LM, PTC
//     with any dt schedule we tried) can robustly navigate
//     past the sigmoid wall from a far warm-start.
//   ▸ The ONLY tool that consistently works is a smart
//     PHYSICAL warm-start that puts Newton in the correct
//     basin of attraction.
//
// This header ships `make_diode_aware_initial_guess` — a
// helper that walks the DevicePool, sets node voltages
// according to source values + b_extra modulation, and
// assumes diodes are OFF (current = 0, with v_anode set by
// the source). For circuits with the usual source → diode →
// load → GND topology, this produces a load-line warm-start
// that's INSIDE the correct Newton basin.
//
// Used in run_transient and standalone tests to crack the
// stiff κ=20 rectifier from a programmatic starting point.

#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/topology/graph.hpp"

namespace pulsim::v2::pwl {

/// Build an initial-guess vector that puts Newton in the
/// correct basin for typical diode-load circuits.
///
/// Algorithm:
///   1. Start with x = zero.
///   2. For each voltage source: read its EFFECTIVE voltage
///      (`pool.V` minus the b_extra contribution on the
///      source's constraint row) and assign it to v_from.
///      v_to (if a non-ground node) is set to zero by default
///      (i.e. assume the source is referenced to ground or
///      to an unset node).
///   3. (Reserved for future extensions: diode load-line
///      computation walking nonlinear branches.)
///
/// For the canonical sinusoidal rectifier
/// (V_sine → diode → R_load → GND), this initialises
/// v_n0 = V_sine(t) and leaves v_n1 = 0. The latter is the
/// "diode-off" assumption, which is INSIDE the correct
/// basin for both positive (where Newton will pull v_n1 up
/// to V_sine − V_F0) and negative half-cycles (where
/// v_n1 ≈ 0 IS the answer).
///
/// This is a structural / lightweight heuristic — it does
/// not solve anything. It just gives Newton a sensible
/// starting point.
[[nodiscard]] inline Vector make_diode_aware_initial_guess(
    const topology::Graph& graph,
    const DevicePool& pool,
    const Vector& b_extra) {

    const Size n = pool.state_size(graph);
    Vector x_init = Vector::Zero(static_cast<Index>(n));

    // Iterate graph branches; for each one stored as a
    // VoltageSource in the pool, write the effective voltage
    // onto the "from" node of the branch. The effective
    // voltage is pool.V minus b_extra[constraint_row], since
    // the MNA constraint reads
    //   (v_from − v_to) − V_constant − b_extra[row] = 0
    // and we want v_from ≈ V_effective when v_to = 0.
    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        const auto kind = pool.kind_of(branch.id);
        if (kind != DevicePool::StoredKind::VoltageSource) {
            continue;
        }
        const auto& vs =
            pool.voltage_source_params(branch.id);
        const Index src_var =
            pool.branch_var_id_for_source(branch.id, graph);
        Real v_effective = vs.V;
        if (src_var >= 0 &&
            src_var < static_cast<Index>(b_extra.size())) {
            v_effective -= b_extra[src_var];
        }
        if (branch.from != graph.ground() &&
            branch.from >= 0 &&
            branch.from < static_cast<Index>(n)) {
            x_init[branch.from] = v_effective;
        }
    }

    return x_init;
}

}  // namespace pulsim::v2::pwl
