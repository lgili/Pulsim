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
#include "pulsim/pwl/gmin.hpp"
#include "pulsim/pwl/preflight.hpp"
#include "pulsim/pwl/row_names.hpp"
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

#include <format>
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
///
/// `source_scale` multiplies every INDEPENDENT source amplitude — the
/// homotopy parameter for source stepping. Dependent sources (VCVS)
/// are deliberately left alone: their gain is a property of the
/// circuit, not an excitation level, and ramping it would change what
/// is being solved rather than how hard it is.
///
/// `gmin` is the conductance floor stamped node-to-ground; see
/// `gmin.hpp`. Zero (the default) leaves the matrix untouched.
inline void dc_assemble(const topology::Graph& graph,
                         const DevicePool& pool,
                         const topology::SwitchStateMask& mask,
                         sparse::Matrix& J,
                         Vector& b,
                         Real t_eval = Real{0},
                         Real gmin = Real{0},
                         Real source_scale = Real{1}) {
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
                                                branch_var_id,
                                                source_scale * p.V);
                break;
            }
            case DevicePool::StoredKind::CurrentSource: {
                const auto& p =
                    pool.current_source_params(branch.id);
                stamping::stamp_current_source(b, coord,
                                                source_scale * p.I);
                break;
            }
            case DevicePool::StoredKind::PWMVoltageSource: {
                const auto& p =
                    pool.pwm_voltage_source_params(branch.id);
                const Index branch_var_id =
                    pool.branch_var_id_for_source(branch.id, graph);
                const Real V_t = source_scale *
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
                const Real V_t = source_scale *
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
                const Real V_t = source_scale *
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

    // v2.0 Phase 2 (B.2): the conductance floor, stamped last so it
    // lands on the node block only — see `gmin.hpp` for why the
    // branch-current rows must be left alone. Defaults to zero: the
    // assembler stays neutral and the SOLVER opts in, so AC / LTI /
    // sweep consumers that build their own matrices are unaffected.
    stamp_gmin(J, graph.num_nodes(), gmin);
}

/// A rank deficiency of the DC system that no amount of
/// conditioning may cover for.
struct DCStructuralDefect {
    bool present = false;
    Index row = kInvalidIndex;   //!< the offending unknown, if any
    std::string detail;          //!< " — ..." suffix for a message
};

/// Decide whether the DC system is structurally rank-deficient — and
/// therefore whether a conductance floor would be CONDITIONING the
/// matrix or INVENTING equations for it.
///
/// Two defects qualify, and they are found different ways:
///
///   1. **A galvanically isolated subnet.** Its nodes have plenty of
///      stamps, so no column is empty; the block simply has no
///      reference, and only reachability sees that. A floor would
///      quietly supply the reference and report an operating point
///      for a circuit whose isolation is the entire point of its
///      design. `preflight.hpp` owns this repair, with a report.
///   2. **An unknown with no equation at all** — an empty MNA column
///      or row. Checked before the next one because it names a more
///      specific mechanism for the same node.
///   3. **A subnet with no DC path to ground.** Galvanically
///      connected — through a coupling capacitor, or fed by an ideal
///      current source — but rank-deficient in the DC system, where
///      those devices contribute nothing. Emptiness cannot see this:
///      the subnet's own internal resistors populate every column.
///
/// Neither probe may libel a healthy circuit, so both are careful:
/// reachability unions EVERY branch regardless of kind (a diode is a
/// galvanic connection even though it does not conduct at DC), and
/// the emptiness probe is taken on the linear stamps UNIONED with
/// the nonlinear ones. `dc_assemble` skips `BranchKind::Nonlinear`,
/// so an interior node of a diode chain has an empty LINEAR column
/// while being perfectly well determined once the diodes are
/// stamped — the same false accusation Phase 1's review caught in
/// the singular-matrix message.
///
/// Nonlinear structure is taken at x = 0: sparsity is a property of
/// which devices touch which nodes, not of the operating point.
[[nodiscard]] inline DCStructuralDefect dc_structural_defect(
    const topology::Graph& graph,
    const DevicePool& pool,
    const topology::SwitchStateMask& mask,
    const NonlinearRefreshFn& refresh,
    Real t_eval) {
    DCStructuralDefect out;

    // 1. GALVANIC reachability: a subnet with no path to ground
    //    through ANY device has no voltage reference at all.
    const auto isolated = detail::components_without_ground(
        graph, [](const topology::Branch&) { return true; });
    if (!isolated.empty() && !isolated.front().empty()) {
        out.present = true;
        out.row = isolated.front().front();
        out.detail = std::format(
            " — {} is in a {}-node subnet with no connection to "
            "ground through any device, so its voltage is undefined "
            "rather than merely hard to compute. A conductance floor "
            "would invent a reference and report a confident answer; "
            "run the topology preflight (Python: it is on by "
            "default) to insert an explicit 1 GΩ tie, or add one "
            "yourself",
            node_label(graph, out.row), isolated.front().size());
        return out;
    }

    // 2. Emptiness: an unknown that appears in no equation.
    sparse::Matrix J;
    Vector b;
    dc_assemble(graph, pool, mask, J, b, t_eval, Real{0});
    sparse::compress_in_place(J);
    if (refresh) {
        const Index n = static_cast<Index>(b.size());
        sparse::Matrix J_nl(n, n);
        Vector f_nl = Vector::Zero(n);
        const Vector x0 = Vector::Zero(n);
        (void)refresh(x0, J_nl, f_nl, graph, pool);
        sparse::compress_in_place(J_nl);
        if (J_nl.nonZeros() > 0) {
            J += J_nl;
            sparse::compress_in_place(J);
        }
    }
    const Index col = sparse::first_empty_column(J);
    const Index row = (col == kInvalidIndex)
        ? sparse::first_empty_row(J) : col;
    if (row != kInvalidIndex) {
        out.present = true;
        out.row = row;
        out.detail = explain_singular(graph, pool, J, nullptr);
        return out;
    }

    // 3. DC reachability. Galvanic connection is not enough: a
    //    subnet whose only path to ground runs through capacitors
    //    (open at DC) or ideal current sources (no conductance) has
    //    a rank-deficient DC block even though every one of its
    //    columns is populated by its own internal devices. Probe 2
    //    above cannot see that — emptiness is not rank — and with
    //    the floor stamped the block becomes invertible and the
    //    solver reports 0 V, or I/(2·gmin) volts, as an operating
    //    point.
    //
    //    Nonlinear branches conduct here IF AND ONLY IF a refresh
    //    will stamp them. `dc_assemble` skips them, which is why
    //    `conducts_at_dc` says no; but Newton puts them back, and
    //    treating an interior node of a diode chain as floating
    //    would be the same false accusation this function exists to
    //    avoid.
    const bool nonlinear_is_stamped = static_cast<bool>(refresh);
    const auto dc_floating = detail::components_without_ground(
        graph, [&](const topology::Branch& br) {
            if (br.kind == topology::BranchKind::Nonlinear) {
                return nonlinear_is_stamped;
            }
            return detail::conducts_at_dc(pool, br);
        });
    if (!dc_floating.empty() && !dc_floating.front().empty()) {
        out.present = true;
        out.row = dc_floating.front().front();
        out.detail = std::format(
            " — {} is in a {}-node subnet with no DC path to ground: "
            "it is connected only through devices that are open at "
            "DC (capacitors, ideal current sources), so the DC "
            "system is rank-deficient there. Its columns are all "
            "populated by the subnet's own devices, so no emptiness "
            "probe can see this, and a conductance floor would "
            "silently supply the missing rank. Run the topology "
            "preflight (Python: it is on by default) or add a "
            "bleeder resistance to ground",
            node_label(graph, out.row), dc_floating.front().size());
        return out;
    }

    return out;
}

/// Solve the DC operating point of the LINEAR part of the circuit.
///
/// `gmin` is the conductance floor (S) stamped from every non-ground
/// node to ground before factorization — see `gmin.hpp`. It is on by
/// default because a DC matrix is where near-open devices (every
/// diode in a bridge reverse-biased, a MOSFET below threshold) leave
/// pivots that are technically nonzero and numerically worthless.
/// Pass 0 to reproduce the un-augmented system exactly.
///
/// NOTE the floor is never allowed to substitute for topology: the
/// structural probe below runs on the UN-augmented matrix, so a node
/// that genuinely has no equation still produces the named Phase-1
/// error rather than a confident 0 V.
inline Vector compute_dc_op(const topology::Graph& graph,
                              const DevicePool& pool,
                              const topology::SwitchStateMask& mask,
                              Real t_eval = Real{0},
                              Real gmin = kDefaultGmin) {
    sparse::Matrix J;
    Vector b;
    dc_assemble(graph, pool, mask, J, b, t_eval);
    sparse::compress_in_place(J);

    if (gmin > Real{0}) {
        // The floor may CONDITION a matrix; it may not INVENT
        // equations for one. Check first and let the diagnostic win.
        const auto defect =
            dc_structural_defect(graph, pool, mask, {}, t_eval);
        if (defect.present) {
            throw std::runtime_error(std::format(
                "compute_dc_op: DC matrix structurally singular for "
                "mask {}{}",
                mask.to_string(), defect.detail));
        }
        stamp_gmin(J, graph.num_nodes(), gmin);
        sparse::compress_in_place(J);
    }

    auto solver = sparse::make_default_solver();
    if (!solver->analyze(J)) {
        throw std::runtime_error(std::format(
            "compute_dc_op: DC matrix structurally singular for "
            "mask {}{}",
            mask.to_string(),
            explain_singular(graph, pool, J, solver.get())));
    }
    if (!solver->factorize(J)) {
        // v2.0 Phase 1 (audit finding
        // `singular-errors-dont-name-the-node`): say WHERE, not just
        // that it failed. Capacitors are open at DC, so a node hung
        // off nothing but a capacitor produces a genuinely empty
        // column here — by far the most common cause, and now named.
        throw std::runtime_error(std::format(
            "compute_dc_op: DC matrix numerically singular for "
            "mask {}{}",
            mask.to_string(),
            explain_singular(graph, pool, J, solver.get())));
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
    bool enable_lm = false,
    Real gmin = kDefaultGmin) {
    if (!refresh) {
        return compute_dc_op(graph, pool, mask, t_eval, gmin);
    }

    // Structural check against the system Newton actually solves.
    // Doing it here rather than inheriting `compute_dc_op`'s
    // linear-only probe is the difference between naming a genuinely
    // floating node and libelling every interior node of a diode
    // chain (whose LINEAR column is empty by construction).
    {
        const auto defect =
            dc_structural_defect(graph, pool, mask, refresh, t_eval);
        if (defect.present) {
            throw std::runtime_error(std::format(
                "compute_dc_op_newton: DC system structurally "
                "singular for mask {}{}",
                mask.to_string(), defect.detail));
        }
    }

    // Warm start from the nonlinear-open linear solve. That system
    // can be singular on its own merits — nodes reachable only
    // through the nonlinear devices we just proved DO stamp them —
    // in which case a cold start is the honest fallback.
    Vector x0;
    try {
        x0 = compute_dc_op(graph, pool, mask, t_eval, gmin);
    } catch (const std::exception&) {
        x0 = Vector::Zero(
            static_cast<Index>(pool.state_size(graph)));
    }

    // Local DC "segment": same shape solve_with_newton_b_extra
    // consumes for transient steps (it reads only J, b_constant,
    // state_size — the pre-factored solver member is unused there).
    PwlSegment seg;
    dc_assemble(graph, pool, mask, seg.J, seg.b_constant, t_eval,
                 gmin);
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
