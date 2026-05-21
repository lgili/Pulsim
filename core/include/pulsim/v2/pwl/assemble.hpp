#pragma once

// =============================================================================
// Pulsim v2 — Layer 4: assemble_segment (build one per-state matrix)
// =============================================================================
//
// `pulsim-v2-pwl-state-space-cache` Phase 3 (Layer 4 V0).
// `pulsim-v2-trapezoidal-companion` Phase 4 (Layer 4 V1 — dt-aware).
//
// For one `SwitchStateMask`, build the MNA matrix `J` + constant
// RHS `b` by stamping every branch in the graph using Layer 3's
// stampers + Layer 2's device-model parameters from DevicePool.
//
// V1 scope: PassiveLinear (Resistor / Capacitor / Inductor),
// Source, and Switch branches. Nonlinear branches are SKIPPED.
//
// dt convention:
//   * dt == 0  → static-only build (V0 backwards compat). Cap /
//                Inductor branches are SKIPPED.
//   * dt  > 0  → V1 dynamic build. Caps / Inductors stamp their
//                trap-companion g_eq into J. History contributions
//                are NOT stamped here — Layer 5 adds them via
//                b_extra at solve time.

#include "pulsim/v2/models/resistor.hpp"
#include "pulsim/v2/models/voltage_source.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/stamping/branch_coord.hpp"
#include "pulsim/v2/stamping/stamp_companion.hpp"
#include "pulsim/v2/stamping/stamp_device.hpp"
#include "pulsim/v2/stamping/stamp_switch.hpp"
#include "pulsim/v2/stamping/stamp_voltage_source.hpp"
#include "pulsim/v2/topology/graph.hpp"
#include "pulsim/v2/topology/switch_state.hpp"

namespace pulsim::v2::pwl {

// -----------------------------------------------------------------------------
// assemble_segment — build (J, b) for one switch state.
//
// Sizes J + b to `pool.state_size(graph)`, zeroes them, then
// iterates every branch in branch_id order. For each branch the
// dispatch by BranchKind picks the right Layer 3 stamper.
//
// PassiveLinear branches dispatch by pool kind: Resistor uses the
// generic stamp_device; Capacitor + Inductor use companion stamps
// IF dt > 0.
//
// V1 still uses `x = Vector::Zero(state_size)` during stamping —
// every device class supported here is linear, so the stamp
// doesn't depend on the operating point. The voltage source's
// `-V` contribution to `b` is captured because
// `stamp_voltage_source` reads V from its parameter, not from `x`.
// Capacitor / Inductor companions contribute 0 to b at assembly;
// the history term lives in `b_extra` at solve time.
// -----------------------------------------------------------------------------
inline void assemble_segment(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask,
                              Real dt,
                              sparse::Matrix& J,
                              Vector& b) {
    const Size state_size = pool.state_size(graph);

    // Reset / size J and b.
    J = sparse::Matrix(static_cast<Index>(state_size),
                        static_cast<Index>(state_size));
    b = Vector::Zero(static_cast<Index>(state_size));

    if (state_size == 0) {
        return;
    }

    // Zero state vector: V1 stamping is linear, so the result
    // doesn't depend on x.
    const Vector x = Vector::Zero(static_cast<Index>(state_size));

    // Counter advances only on Switch-kind branches; matches the
    // SwitchStateMask bit-i = i-th Switch convention.
    Size switch_idx = 0;

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        const stamping::BranchCoord coord{branch.from, branch.to,
                                            branch.id};

        switch (branch.kind) {
        case topology::BranchKind::PassiveLinear: {
            // V1 dispatch: PassiveLinear may be Resistor, Capacitor,
            // or Inductor. Use the pool's StoredKind to discriminate.
            const auto k = pool.kind_of(branch.id);
            switch (k) {
            case DevicePool::StoredKind::Resistor: {
                const auto& p = pool.resistor_params(branch.id);
                stamping::stamp_device<models::Resistor>(J, b, x,
                                                          coord, p);
                break;
            }
            case DevicePool::StoredKind::Capacitor: {
                if (dt > Real{0}) {
                    const auto& p = pool.capacitor_params(branch.id);
                    const Real g_eq = models::Capacitor::g_eq(dt, p);
                    stamping::stamp_capacitor_companion(J, b, coord,
                                                         g_eq);
                }
                // dt == 0: V0 backwards-compat path — skip.
                break;
            }
            case DevicePool::StoredKind::Inductor: {
                if (dt > Real{0}) {
                    const auto& p = pool.inductor_params(branch.id);
                    const Real g_eq_inv =
                        models::Inductor::g_eq_inv(dt, p);
                    const Index branch_var_id =
                        pool.branch_var_id_for_inductor(branch.id, graph);
                    stamping::stamp_inductor_companion(
                        J, b, coord, branch_var_id, g_eq_inv);
                }
                break;
            }
            default:
                // Unexpected PassiveLinear kind — should be
                // unreachable. Silently skip.
                break;
            }
            break;
        }
        case topology::BranchKind::Source: {
            const auto& p = pool.voltage_source_params(branch.id);
            const Index branch_var_id =
                pool.branch_var_id_for_source(branch.id, graph);
            stamping::stamp_voltage_source(J, b, x, coord,
                                            branch_var_id, p.V);
            break;
        }
        case topology::BranchKind::Switch: {
            // The topological `Switch` kind covers both
            // user-controlled `IdealSwitch` and auto-commutating
            // `SwitchedDiode`. Use the pool's StoredKind to fetch
            // the right (g_on, g_off).
            const bool closed = mask.get(switch_idx);
            Real g_on  = Real{0};
            Real g_off = Real{0};
            const auto k = pool.kind_of(branch.id);
            if (k == DevicePool::StoredKind::Diode) {
                const auto& p = pool.diode_params(branch.id);
                g_on  = p.g_on;
                g_off = p.g_off;
            } else {
                g_on  = pool.switch_g_on(branch.id);
                g_off = pool.switch_g_off(branch.id);
            }
            stamping::stamp_switch_fixed(J, b, x, coord, closed,
                                          g_on, g_off);
            ++switch_idx;
            break;
        }
        case topology::BranchKind::Nonlinear:
            // Still deliberately skipped — per-segment Newton on top
            // of the cached factor lands in
            // `pulsim-v2-nonlinear-segment-newton`.
            break;
        }
    }
}

// -----------------------------------------------------------------------------
// V0 backwards-compat overload — static-only assembly (dt = 0).
//
// Caps and Inductors are silently SKIPPED, matching V0 behaviour.
// New callers that have dynamic devices SHOULD use the dt-aware
// overload.
// -----------------------------------------------------------------------------
inline void assemble_segment(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask,
                              sparse::Matrix& J,
                              Vector& b) {
    assemble_segment(graph, pool, mask, Real{0}, J, b);
}

}  // namespace pulsim::v2::pwl
