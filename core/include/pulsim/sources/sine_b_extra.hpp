#pragma once

// =============================================================================
// Pulsim — Layer 2 V11: Sine b_extra computation helper
// =============================================================================
//
// Walks the DevicePool for `SineVoltageSource` entries and
// computes the b_extra contribution that overlays the time-
// varying sine value on top of the (V=0 baseline) source's
// constraint row.
//
// Math: the source is assembled with V_baseline = 0. The MNA
// constraint row reads (v_from − v_to) − V_sine(t) = 0.
// With V_baseline = 0 stamped statically, we add
// b_extra[branch_var_row] = −V_sine(t) so that solving
// J·x = −(b_constant + b_extra) yields (v_from − v_to) =
// V_sine(t) at every step.
//
// Same wiring pattern as `compute_pwm_b_extra` (V4).

#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/topology/graph.hpp"

namespace pulsim::sources {

/// Overlay each SineVoltageSource's instantaneous value
/// into b_extra at simulation time `t`. Returns a vector
/// of size `pool.state_size(graph)` with all zeros except
/// at the sine-source branch-var rows.
/// Output-parameter overload (hot-path friendly): fill `out` — resized to
/// `state_size` and zeroed — reusing the caller's buffer instead of
/// allocating one per call.
inline void compute_sine_b_extra(
    const pwl::DevicePool& pool,
    const topology::Graph& graph,
    Real t,
    Vector& out) {
    const Size state_size = pool.state_size(graph);
    if (out.size() != static_cast<Index>(state_size)) {
        out = Vector::Zero(static_cast<Index>(state_size));
    } else {
        out.setZero();
    }

    for (Index b_id = 0;
         b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        if (branch.kind != topology::BranchKind::Source) {
            continue;
        }
        const auto k = pool.kind_of(branch.id);
        if (k != pwl::DevicePool::StoredKind::SineVoltageSource) {
            continue;
        }
        const auto& p =
            pool.sine_voltage_source_params(branch.id);
        const Index src_var =
            pool.branch_var_id_for_source(branch.id, graph);
        const Real v_sine =
            models::SineVoltageSource::value_at(p, t);
        out[src_var] += -v_sine;
    }
}

/// Allocating convenience overload — returns a fresh vector. Delegates to the
/// output-parameter overload above.
[[nodiscard]] inline Vector compute_sine_b_extra(
    const pwl::DevicePool& pool,
    const topology::Graph& graph,
    Real t) {
    Vector out;
    compute_sine_b_extra(pool, graph, t, out);
    return out;
}

}  // namespace pulsim::sources
