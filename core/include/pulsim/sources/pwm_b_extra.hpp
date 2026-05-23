#pragma once

// =============================================================================
// Pulsim — Layer 2 V4: PWM b_extra computation helper
// =============================================================================
//
// At each `run_transient` time step, this helper walks the
// DevicePool for `PWMVoltageSource` entries and computes the
// b_extra contribution that overlays the time-varying PWM
// value on top of the (V=0 baseline) source's constraint
// row.
//
// Math:
//   The PWM source is assembled with V=0 baseline. The MNA
//   constraint row reads:
//       (v_from - v_to) - V_pwm(t) = 0
//   With V_baseline = 0, this becomes:
//       (v_from - v_to) - 0 = 0
//   We need to subtract V_pwm(t) from the residual, which
//   in v2's "cache.solve does -(b_constant + b_extra)"
//   convention means:
//       b_extra[branch_var_row] = -V_pwm(t)
//   Then solving J·x = -(b_constant + b_extra) gives:
//       (v_from - v_to) = V_pwm(t)  ✓
//
// This mirrors the established pattern used by the V8 layer5_v2
// half-wave rectifier test (sinusoidal source via b_extra).

#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::sources {

/// Walk all `PWMVoltageSource` entries in `pool` and
/// return a b_extra vector that overlays each one's
/// time-varying value at simulation time `t`.
///
/// `state_size` is the full state-vector size (number of
/// nodes + voltage sources + inductors). The returned
/// vector has all zeros except at PWM sources' branch-var
/// rows, where it holds `-V_pwm(t)` (matching the v2
/// solve-convention).
[[nodiscard]] inline Vector compute_pwm_b_extra(
    const pwl::DevicePool& pool,
    const topology::Graph& graph,
    Real t) {
    const Size state_size = pool.state_size(graph);
    Vector b_extra =
        Vector::Zero(static_cast<Index>(state_size));

    // Iterate graph branches looking for PWMVoltageSource
    // entries. The pool doesn't keep a dedicated "PWM
    // sources" list, so we scan all Source branches and
    // check the stored kind.
    for (Index b_id = 0;
         b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        if (branch.kind != topology::BranchKind::Source) {
            continue;
        }
        const auto k = pool.kind_of(branch.id);
        if (k != pwl::DevicePool::StoredKind::PWMVoltageSource) {
            continue;
        }
        const auto& p =
            pool.pwm_voltage_source_params(branch.id);
        const Index src_var =
            pool.branch_var_id_for_source(branch.id, graph);
        const Real v_pwm =
            models::PWMVoltageSource::value_at(p, t);
        b_extra[src_var] += -v_pwm;
    }
    return b_extra;
}

}  // namespace pulsim::sources
