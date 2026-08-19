#pragma once

// =============================================================================
// Pulsim — Layer 4: assemble_segment (build one per-state matrix)
// =============================================================================
//
// `pulsim-v2-pwl-state-space-cache` Phase 3 (Layer 4 V0).
// `pulsim-v2-trapezoidal-companion` Phase 4 (Layer 4 V1 — dt-aware).
// v2.0 Phase 1: (G, C, b) split assembly.
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
//
// (G, C, b) split (v2.0 Phase 1, audit finding A.5): every
// dt-dependent trap-companion entry in this kernel is EXACTLY
// linear in 1/dt —
//   * capacitor block      ±(2·C_cap)/dt
//   * inductor constraint  −(2·L)/dt on the branch-var diagonal
//   * transformer cross    −(2·M)/dt between winding branch rows
// — so the full matrix decomposes as
//       J(mask, dt) = G(mask) + (1/dt) · C
// with `G` holding every dt-independent stamp (including the
// inductor's ±1 incidence terms, which the dt-monolithic V1 path
// only emitted when dt > 0) and `C` holding the 1/dt COEFFICIENTS
// (2·C_cap, −2·L, −2·M). `C` does not depend on the switch mask.
// `b` receives source stamps only and is dt-independent.
//
// `assemble_segment_split` exposes the split directly (consumed by
// the cache's event-dt solver reuse and the EXACT continuous-LTI
// extraction); the dt-aware `assemble_segment` recombines
// J = G + (1/dt)·C so every caller shares one stamping loop.

#include "pulsim/models/resistor.hpp"
#include "pulsim/models/transformer.hpp"
#include "pulsim/models/voltage_source.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/stamping/stamp_companion.hpp"
#include "pulsim/stamping/stamp_current_source.hpp"
#include "pulsim/stamping/stamp_device.hpp"
#include "pulsim/stamping/stamp_switch.hpp"
#include "pulsim/stamping/stamp_vcvs.hpp"
#include "pulsim/stamping/stamp_voltage_source.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

namespace pulsim::pwl {

namespace detail {

// -----------------------------------------------------------------------------
// assemble_impl — the ONE stamping loop behind both public entry
// points.
//
// `S` is the static-stamp target (the caller's G, or the legacy
// static-only J). `C_out` selects the mode:
//   * C_out != nullptr → full split: static stamps land in S,
//     1/dt-coefficient stamps land in *C_out, and the inductor's
//     ±1 incidence terms land in S (they are dt-independent).
//   * C_out == nullptr → legacy V0 static-only assembly: Cap /
//     Inductor / transformer-coupling stamps are SKIPPED ENTIRELY
//     (including the inductor incidence — at dt = 0 the inductor
//     branch is absent from the system, matching V0 behaviour).
//
// V1 still uses `x = Vector::Zero(state_size)` during stamping —
// every device class supported here is linear, so the stamp
// doesn't depend on the operating point. The voltage source's
// `-V` contribution to `b` is captured because
// `stamp_voltage_source` reads V from its parameter, not from `x`.
// Capacitor / Inductor companions contribute 0 to b at assembly;
// the history term lives in `b_extra` at solve time.
// -----------------------------------------------------------------------------
inline void assemble_impl(const topology::Graph& graph,
                           const DevicePool& pool,
                           const topology::SwitchStateMask& mask,
                           sparse::Matrix& S,
                           sparse::Matrix* C_out,
                           Vector& b) {
    const Size state_size = pool.state_size(graph);
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
                stamping::stamp_device<models::Resistor>(S, b, x,
                                                          coord, p);
                break;
            }
            case DevicePool::StoredKind::Capacitor: {
                if (C_out != nullptr) {
                    const auto& p = pool.capacitor_params(branch.id);
                    // Coefficient of 1/dt: g_eq(dt) = 2C/dt →
                    // stamp the block with 2C into C_out.
                    stamping::stamp_capacitor_companion(
                        *C_out, b, coord, Real{2} * p.C);
                }
                // Static-only: V0 backwards-compat path — skip.
                break;
            }
            case DevicePool::StoredKind::Inductor: {
                if (C_out != nullptr) {
                    const auto& p = pool.inductor_params(branch.id);
                    const Index branch_var_id =
                        pool.branch_var_id_for_inductor(branch.id, graph);
                    // Incidence (±1) → S; diagonal coefficient
                    // (−2L) → C_out.
                    stamping::stamp_inductor_companion_split(
                        S, *C_out, coord, branch_var_id,
                        Real{2} * p.L);
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
            // The topological `Source` kind covers
            // VoltageSource, CurrentSource, AND
            // PWMVoltageSource (a time-varying voltage
            // source). Dispatch on the pool's StoredKind.
            const auto src_kind = pool.kind_of(branch.id);
            if (src_kind == DevicePool::StoredKind::CurrentSource) {
                const auto& p =
                    pool.current_source_params(branch.id);
                stamping::stamp_current_source(b, coord, p.I);
            } else if (src_kind ==
                       DevicePool::StoredKind::PWMVoltageSource) {
                // Structurally a VoltageSource with V=0
                // baseline. The time-varying PWM value is
                // overlaid via b_extra at runtime by
                // run_transient's PWM pass.
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(S, b, x, coord,
                                                branch_var_id, Real{0});
            } else if (src_kind ==
                       DevicePool::StoredKind::SineVoltageSource) {
                // V11: same V=0 baseline pattern; sine
                // value comes from b_extra each step.
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(S, b, x, coord,
                                                branch_var_id, Real{0});
            } else if (src_kind ==
                       DevicePool::StoredKind::PulseVoltageSource) {
                // V12: same V=0 baseline + b_extra overlay.
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(S, b, x, coord,
                                                branch_var_id, Real{0});
            } else if (src_kind ==
                       DevicePool::StoredKind::VCVS) {
                // V15: voltage-controlled voltage source.
                // Constraint row references the sense inputs.
                const auto& vp = pool.vcvs_params(branch.id);
                const auto [in_pos, in_neg] =
                    pool.vcvs_input_nodes(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_vcvs(S, b, x, coord,
                                       in_pos, in_neg,
                                       branch_var_id, vp.gain);
            } else {
                const auto& p = pool.voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(S, b, x, coord,
                                                branch_var_id, p.V);
            }
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
            stamping::stamp_switch_fixed(S, b, x, coord, closed,
                                          g_on, g_off);
            ++switch_idx;
            break;
        }
        case topology::BranchKind::Nonlinear: {
            // V17: Saturable inductors live as Nonlinear
            // branches but use the inductor branch-var
            // numbering. Their constraint row would be
            // ALL-ZERO in the linear assembly (the Newton
            // refresh stamps everything per-iteration),
            // which makes KLU's factorisation fail. Add a
            // tiny "G_min" diagonal here so the cache builds;
            // the Newton refresh's stamping dwarfs G_min in
            // the combined Jacobian.
            const auto src_kind = pool.kind_of(branch.id);
            if (src_kind ==
                    DevicePool::StoredKind::SaturableInductor) {
                const Index branch_var_id =
                    pool.branch_var_id_for_inductor(
                        branch.id, graph);
                S.coeffRef(branch_var_id, branch_var_id) +=
                    Real{1e-12};
            }
            break;
        }
        }
    }

    // ----- Layer 2 V2: transformer cross-coupling pass ---------------------
    //
    // For each registered (primary, secondary) inductor pair,
    // add the mutual-inductance cross-terms:
    //   J[p_row, s_col] += -(2M/dt)  →  C[p_row, s_col] += -2M
    //   J[s_row, p_col] += -(2M/dt)  →  C[s_row, p_col] += -2M
    //
    // This runs AFTER per-branch stamping so each inductor's
    // self-inductance diagonal is already in place. Skipped in
    // static-only mode (couplings have no static contribution;
    // transformers behave as open circuits at DC, matching the
    // inductor's static path).
    if (C_out != nullptr) {
        for (const auto& tc : pool.transformer_couplings()) {
            const Index p_row =
                pool.branch_var_id_for_inductor(
                    tc.primary_branch_id, graph);
            const Index s_row =
                pool.branch_var_id_for_inductor(
                    tc.secondary_branch_id, graph);
            const Real cross =
                Real{2} * models::TwoWindingTransformer::
                              mutual_inductance(tc.params);
            C_out->coeffRef(p_row, s_row) += -cross;
            C_out->coeffRef(s_row, p_row) += -cross;
        }
    }
}

}  // namespace detail

// -----------------------------------------------------------------------------
// assemble_segment_split — build (G, C, b) for one switch state.
//
// `G` holds every dt-independent stamp; `C` holds the 1/dt
// coefficients, so J(dt) = G + (1/dt)·C for ANY dt > 0. Both are
// returned COMPRESSED. `C` is mask-invariant (switches only touch
// G) — callers may cache it across masks if the pool's L/C/M
// parameters have not changed.
// -----------------------------------------------------------------------------
inline void assemble_segment_split(const topology::Graph& graph,
                                    const DevicePool& pool,
                                    const topology::SwitchStateMask& mask,
                                    sparse::Matrix& G,
                                    sparse::Matrix& C,
                                    Vector& b) {
    const auto n = static_cast<Index>(pool.state_size(graph));
    G = sparse::Matrix(n, n);
    C = sparse::Matrix(n, n);
    b = Vector::Zero(n);
    detail::assemble_impl(graph, pool, mask, G, &C, b);
    G.makeCompressed();
    C.makeCompressed();
}

// -----------------------------------------------------------------------------
// assemble_segment — build (J, b) for one switch state.
//
// dt > 0 recombines the split: J = G + (1/dt)·C (single stamping
// loop shared with assemble_segment_split — no drift between the
// monolithic and split assemblies). dt == 0 is the V0 static-only
// path: Caps and Inductors are skipped entirely.
// -----------------------------------------------------------------------------
inline void assemble_segment(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask,
                              Real dt,
                              sparse::Matrix& J,
                              Vector& b) {
    const auto n = static_cast<Index>(pool.state_size(graph));
    if (dt > Real{0}) {
        sparse::Matrix G, C;
        assemble_segment_split(graph, pool, mask, G, C, b);
        J = G + (Real{1} / dt) * C;
        return;
    }
    J = sparse::Matrix(n, n);
    b = Vector::Zero(n);
    detail::assemble_impl(graph, pool, mask, J, nullptr, b);
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

}  // namespace pulsim::pwl
