#pragma once

// =============================================================================
// Pulsim — Layer 4 V2: DC operating-point MNA assembly
// =============================================================================
//
// `pulsim-v2-dc-operating-point` Phase 1.
//
// `dc_assemble` is a sibling of `assemble_segment` that builds
// the DC MNA matrix for a given switch state. Differences from
// the trap-companion assembly:
//   * Capacitors are SKIPPED (g_eq = 0 at DC, no contribution).
//   * Inductors are stamped as v_from − v_to = 0 short circuits
//     (with the branch-current unknown still present so i_L is
//     recoverable from the solution).
//   * Resistors, VoltageSources, Switches, Diodes stamp as
//     usual (their stamps are dt-independent).
//
// `compute_dc_op` solves the DC system and returns the state
// vector. Throws on singular matrices.

#include "pulsim/models/current_source.hpp"
#include "pulsim/models/pulse_voltage_source.hpp"
#include "pulsim/models/pwm_voltage_source.hpp"
#include "pulsim/models/resistor.hpp"
#include "pulsim/models/sine_voltage_source.hpp"
#include "pulsim/models/vcvs.hpp"
#include "pulsim/models/voltage_source.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/nonlinear_solve.hpp"
#include "pulsim/pwl/segment.hpp"
#include "pulsim/sparse/matrix.hpp"
#include "pulsim/sparse/solver.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/stamping/stamp_current_source.hpp"
#include "pulsim/stamping/stamp_device.hpp"
#include "pulsim/stamping/stamp_switch.hpp"
#include "pulsim/stamping/stamp_vcvs.hpp"
#include "pulsim/stamping/stamp_voltage_source.hpp"
#include "pulsim/topology/graph.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <stdexcept>

namespace pulsim::pwl {

/// Stamp an inductor as a DC short: v_from − v_to = 0 with i_L
/// as a branch unknown. The constraint row reads "+1 at from,
/// −1 at to, 0 at i_L" (the i_L coefficient is zero, but the
/// branch-current unknown still exists in `x` so we can read
/// i_L from it). The KCL contributions of i_L on the terminal
/// rows are identical to the trap case.
inline void stamp_inductor_dc(sparse::Matrix& J, Vector& b,
                                const stamping::BranchCoord& coord,
                                Index branch_var_id) noexcept {
    (void)b;
    const bool from_active = stamping::node_is_active(coord.from);
    const bool to_active   = stamping::node_is_active(coord.to);

    // KCL contributions of i_L on terminal rows.
    if (from_active) {
        J.coeffRef(coord.from, branch_var_id) += Real{1};
    }
    if (to_active) {
        J.coeffRef(coord.to, branch_var_id) -= Real{1};
    }

    // Constraint row (v_from − v_to = 0).
    if (from_active) {
        J.coeffRef(branch_var_id, coord.from) += Real{1};
    }
    if (to_active) {
        J.coeffRef(branch_var_id, coord.to) -= Real{1};
    }
    // No diagonal entry on branch_var_id — DC inductor is just
    // a v=0 constraint, no L · di/dt term. (The system is
    // structurally singular along that diagonal until the v
    // constraint forces a unique solution via the rest of the
    // matrix.)
    //
    // To avoid singularity from missing diagonals we add a
    // tiny epsilon — physically equivalent to a near-zero
    // resistance in series with the inductor. KLU handles it.
    J.coeffRef(branch_var_id, branch_var_id) += Real{-1e-12};
}

/// Build the DC MNA matrix. `t_eval` is the evaluation time
/// for time-varying sources (PWM / Sine / Pulse): the DC
/// operating point is computed treating each source as if it
/// were held at its instantaneous value at t = t_eval. Default
/// 0 reproduces the historical behaviour for circuits with
/// only DC sources (VoltageSource / CurrentSource).
///
/// Time-varying sources dispatched here:
///   * VoltageSource      — V = p.V (DC)
///   * CurrentSource      — I = p.I (DC, no branch unknown)
///   * PWMVoltageSource   — V = PWMVoltageSource::value_at(p, t_eval)
///   * SineVoltageSource  — V = SineVoltageSource::value_at(p, t_eval)
///   * PulseVoltageSource — V = PulseVoltageSource::value_at(p, t_eval)
///   * VCVS               — linear gain stamp (no time)
inline void dc_assemble(const topology::Graph& graph,
                         const DevicePool& pool,
                         const topology::SwitchStateMask& mask,
                         sparse::Matrix& J,
                         Vector& b,
                         Real t_eval = Real{0}) {
    const Size state_size = pool.state_size(graph);
    J = sparse::Matrix(static_cast<Index>(state_size),
                        static_cast<Index>(state_size));
    b = Vector::Zero(static_cast<Index>(state_size));

    if (state_size == 0) {
        return;
    }

    const Vector x = Vector::Zero(static_cast<Index>(state_size));
    Size switch_idx = 0;

    for (Index b_id = 0; b_id < graph.num_branches(); ++b_id) {
        const auto& branch = graph.branch(b_id);
        const stamping::BranchCoord coord{branch.from, branch.to,
                                            branch.id};

        switch (branch.kind) {
        case topology::BranchKind::PassiveLinear: {
            const auto k = pool.kind_of(branch.id);
            switch (k) {
            case DevicePool::StoredKind::Resistor: {
                const auto& p = pool.resistor_params(branch.id);
                stamping::stamp_device<models::Resistor>(J, b, x,
                                                          coord, p);
                break;
            }
            case DevicePool::StoredKind::Capacitor:
                // Skip — open circuit at DC.
                break;
            case DevicePool::StoredKind::Inductor: {
                const Index branch_var_id =
                    pool.branch_var_id_for_inductor(branch.id, graph);
                stamp_inductor_dc(J, b, coord, branch_var_id);
                break;
            }
            default:
                break;
            }
            break;
        }
        case topology::BranchKind::Source: {
            const auto k = pool.kind_of(branch.id);
            switch (k) {
            case DevicePool::StoredKind::VoltageSource: {
                const auto& p = pool.voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                stamping::stamp_voltage_source(J, b, x, coord,
                                                branch_var_id, p.V);
                break;
            }
            case DevicePool::StoredKind::CurrentSource: {
                const auto& p =
                    pool.current_source_params(branch.id);
                stamping::stamp_current_source(b, coord, p.I);
                break;
            }
            case DevicePool::StoredKind::PWMVoltageSource: {
                const auto& p =
                    pool.pwm_voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const Real V_t =
                    models::PWMVoltageSource::value_at(p, t_eval);
                stamping::stamp_voltage_source(J, b, x, coord,
                                                branch_var_id, V_t);
                break;
            }
            case DevicePool::StoredKind::SineVoltageSource: {
                const auto& p =
                    pool.sine_voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const Real V_t =
                    models::SineVoltageSource::value_at(p, t_eval);
                stamping::stamp_voltage_source(J, b, x, coord,
                                                branch_var_id, V_t);
                break;
            }
            case DevicePool::StoredKind::PulseVoltageSource: {
                const auto& p =
                    pool.pulse_voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const Real V_t =
                    models::PulseVoltageSource::value_at(p, t_eval);
                stamping::stamp_voltage_source(J, b, x, coord,
                                                branch_var_id, V_t);
                break;
            }
            case DevicePool::StoredKind::VCVS: {
                const auto& p = pool.vcvs_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const auto [in_pos, in_neg] =
                    pool.vcvs_input_nodes(branch.id);
                stamping::stamp_vcvs(J, b, x, coord,
                                      in_pos, in_neg,
                                      branch_var_id, p.gain);
                break;
            }
            default:
                break;
            }
            break;
        }
        case topology::BranchKind::Switch: {
            const bool closed = mask.get(switch_idx);
            Real g_on, g_off;
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
            // Skipped (handled by future Newton OpenSpec).
            break;
        }
    }
}

inline Vector compute_dc_op(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask,
                              Real t_eval = Real{0}) {
    sparse::Matrix J;
    Vector b;
    dc_assemble(graph, pool, mask, J, b, t_eval);
    sparse::compress_in_place(J);

    auto solver = sparse::make_default_solver();
    if (!solver->analyze(J)) {
        throw std::runtime_error(
            "compute_dc_op: DC matrix structurally singular for "
            "mask " + mask.to_string());
    }
    if (!solver->factorize(J)) {
        throw std::runtime_error(
            "compute_dc_op: DC matrix numerically singular for "
            "mask " + mask.to_string());
    }

    Vector x;
    Vector rhs = -b;   // J · x = -b solves f = J·x + b = 0
    solver->solve(rhs, x);
    return x;
}

/// Phase-0 fix #2 — Newton DC operating point.
///
/// `compute_dc_op` stamps `BranchKind::Nonlinear` devices as OPEN
/// CIRCUITS (see the case label above), so for any circuit with a
/// smooth diode / MOSFET L1 / IGBT L1 it converges — with no
/// warning — to the operating point of a DIFFERENT circuit (audit
/// finding dc-op-skips-nonlinear-devices, CONFIRMED). This variant
/// runs the SAME Newton machinery the transient uses
/// (`solve_with_newton_b_extra` + the caller's NonlinearRefreshFn
/// chain) on the DC-assembled system:
///
///   * linear part: the dc_assemble matrix (caps open, inductors
///     near-short) — identical to `compute_dc_op`;
///   * nonlinear part: re-stamped per Newton iterate by `refresh`,
///     exactly as in the transient inner solve;
///   * warm start: the linear DC solution (nonlinear-open), which
///     is the natural continuation seed.
///
/// For a circuit with no Newton devices the refresh stamps nothing
/// and this returns the `compute_dc_op` answer after one iteration.
inline Vector compute_dc_op_newton(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh,
    Real t_eval = Real{0},
    Size max_iters = Size{50},
    Real tol_dx  = Real{1e-9},
    Real tol_res = Real{1e-9},
    bool enable_line_search = false,
    bool enable_lm = false) {
    // Warm start from the nonlinear-open linear solve.
    Vector x0 = compute_dc_op(graph, pool, mask, t_eval);
    if (!refresh) {
        return x0;
    }

    // Local DC "segment": same shape solve_with_newton_b_extra
    // consumes for transient steps (it reads only J, b_constant,
    // state_size — the pre-factored solver member is unused there).
    PwlSegment seg;
    dc_assemble(graph, pool, mask, seg.J, seg.b_constant, t_eval);
    sparse::compress_in_place(seg.J);
    seg.state_size = static_cast<Size>(seg.b_constant.size());

    const Vector b_extra =
        Vector::Zero(static_cast<Index>(seg.state_size));
    return solve_with_newton_b_extra(
        seg, refresh, graph, pool, x0, b_extra,
        max_iters, tol_dx, tol_res,
        enable_line_search, enable_lm);
}

}  // namespace pulsim::pwl
