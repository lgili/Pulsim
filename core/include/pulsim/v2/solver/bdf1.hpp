// SPDX-License-Identifier: MIT
//
// Pulsim v2 — Phase B.2: BDF1 (implicit Euler) integrator.
//
// The trap (Tustin) companion in the existing kernel is 2nd-order
// and A-stable, but NOT L-stable: a stiff system with a fast pole
// produces a NUMERICAL OSCILLATION (alternating-sign noise) on top
// of the analytic decay. BDF1 (= implicit Euler) is 1st-order but
// L-stable — it adds artificial damping that kills the oscillation,
// at the cost of slightly more numerical error on smooth solutions.
//
// Use BDF1 when:
//   * The plant has a stiff pole and you see ringing in the trap
//     output (the classic "Tustin ring").
//   * Sub-µs dynamics with large physical time constants — the L/C
//     time constants stress trap because dt is comparable to 1/p.
//   * Quick-and-dirty robustness check: if trap and BDF1 agree, the
//     answer is right; if they don't, you have a numerical issue.
//
// Discrete-time companion (per device, vs trap in [brackets]):
//
//   Capacitor (C·dv/dt = i):
//     BDF1:   G_eq   = C / dt        [trap: 2C/dt]
//             I_hist = G_eq · v_n    [trap: G_eq · v_n + i_n]
//
//   Inductor (v = L·di/dt):
//     BDF1:   G_eq_inv = dt / L      [trap: dt / (2L)]
//             b_row    = G_eq_diag · i_n   = (L/dt) · i_n
//                                       [trap: (2L/dt) · i_n + v_n]
//
// Note BDF1 drops the i_n term from caps' I_hist and the v_n term
// from inductors' b_row — that's the source of the L-stability.
//
// This is a SEPARATE driver path; the existing trap-based
// `run_transient(...)` is unchanged.
//
// Header-only, C++20.

#pragma once

#include "pulsim/v2/builder/circuit_builder.hpp"
#include "pulsim/v2/numeric/dense.hpp"
#include "pulsim/v2/numeric/types.hpp"
#include "pulsim/v2/pwl/dc_assemble.hpp"
#include "pulsim/v2/pwl/device_pool.hpp"
#include "pulsim/v2/solver/options.hpp"
#include "pulsim/v2/solver/result.hpp"
#include "pulsim/v2/solver/run_transient.hpp"   // SwitchScheduleFn / BExtraFn / StepObserverFn typedefs
#include "pulsim/v2/sparse/matrix.hpp"
#include "pulsim/v2/sparse/solver.hpp"
#include "pulsim/v2/stamping/stamp_companion.hpp"
#include "pulsim/v2/topology/graph.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsim::v2::solver {

// =============================================================================
// Bdf1HistoryEntry — per-device past state
// =============================================================================

struct Bdf1HistoryEntry {
    pwl::DevicePool::StoredKind kind;
    Index from = -1;
    Index to   = -1;
    Index inductor_branch_var_id = -1;
    Real  C_or_L = Real{0};
    // Last accepted (v_prev, i_prev). For BDF1 we only need ONE of
    // them depending on device, but we store both for simplicity +
    // for diagnostics.
    Real  v_prev = Real{0};
    Real  i_prev = Real{0};
};


// =============================================================================
// Bdf1HistoryState — populate at init, update each step
// =============================================================================

class Bdf1HistoryState {
public:
    Bdf1HistoryState(const topology::Graph& graph,
                       const pwl::DevicePool& pool) {
        state_size_ = pool.state_size(graph);
        const Index nb = graph.num_branches();
        for (Index bid = 0; bid < nb; ++bid) {
            const auto& branch = graph.branch(bid);
            if (branch.kind != topology::BranchKind::PassiveLinear) {
                continue;
            }
            const auto stored = pool.kind_of(branch.id);
            if (stored == pwl::DevicePool::StoredKind::Capacitor) {
                const auto& cp = pool.capacitor_params(branch.id);
                Bdf1HistoryEntry e{};
                e.kind = stored;
                e.from = branch.from;
                e.to   = branch.to;
                e.C_or_L = cp.C;
                entries_.push_back(e);
            } else if (stored == pwl::DevicePool::StoredKind::Inductor) {
                const auto& lp = pool.inductor_params(branch.id);
                Bdf1HistoryEntry e{};
                e.kind = stored;
                e.from = branch.from;
                e.to   = branch.to;
                e.inductor_branch_var_id =
                    pool.branch_var_id_for_inductor(branch.id, graph);
                e.C_or_L = lp.L;
                entries_.push_back(e);
            }
        }
    }

    void reset() noexcept {
        for (auto& e : entries_) {
            e.v_prev = Real{0};
            e.i_prev = Real{0};
        }
    }

    void seed_from_dc_op(const Vector& dc_x) {
        for (auto& e : entries_) {
            const Real v_from =
                stamping::read_node_voltage(dc_x, e.from);
            const Real v_to   =
                stamping::read_node_voltage(dc_x, e.to);
            const Real v_new  = v_from - v_to;
            if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                e.v_prev = v_new;
                e.i_prev = Real{0};
            } else {
                e.v_prev = Real{0};
                e.i_prev = dc_x[e.inductor_branch_var_id];
            }
        }
    }

    /// BDF1 b_extra contribution for this step:
    ///   Cap: I_hist = (C/dt) · v_n   →
    ///        b_extra[from] -= I_hist, b_extra[to] += I_hist
    ///   Ind: b_extra[branch_var_id] += (L/dt) · i_n
    [[nodiscard]] Vector compute_b_extra(Real dt) const {
        Vector b_extra =
            Vector::Zero(static_cast<Index>(state_size_));
        if (entries_.empty() || dt <= Real{0}) {
            return b_extra;
        }
        for (const auto& e : entries_) {
            if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                const Real i_hist = e.C_or_L * e.v_prev / dt;
                if (stamping::node_is_active(e.from)) {
                    b_extra[e.from] -= i_hist;
                }
                if (stamping::node_is_active(e.to)) {
                    b_extra[e.to]   += i_hist;
                }
            } else {
                const Real L_over_dt = e.C_or_L / dt;
                b_extra[e.inductor_branch_var_id] +=
                    L_over_dt * e.i_prev;
            }
        }
        return b_extra;
    }

    /// After solving for x_{n+1}, read off v_C,n+1 and i_L,n+1 and
    /// stash them as the new (v_prev, i_prev).
    void update_from_state(const Vector& x, Real dt) {
        if (entries_.empty() || dt <= Real{0}) {
            return;
        }
        for (auto& e : entries_) {
            const Real v_from =
                stamping::read_node_voltage(x, e.from);
            const Real v_to   =
                stamping::read_node_voltage(x, e.to);
            const Real v_new  = v_from - v_to;
            if (e.kind == pwl::DevicePool::StoredKind::Capacitor) {
                // BDF1 cap current: i_new = (C/dt)·(v_new − v_prev).
                e.i_prev = e.C_or_L * (v_new - e.v_prev) / dt;
                e.v_prev = v_new;
            } else {
                // Inductor: i_L is a direct unknown in x.
                e.i_prev = x[e.inductor_branch_var_id];
                e.v_prev = v_new;
            }
        }
    }

    [[nodiscard]] Size state_size() const noexcept {
        return state_size_;
    }

private:
    std::vector<Bdf1HistoryEntry> entries_;
    Size state_size_ = 0;
};


// =============================================================================
// Bdf1 segment assembly — like assemble_segment but with BDF1
// companion stamps for C and L (g_eq scaled by 1/2 vs trap)
// =============================================================================

/// BDF1 segment assembly = dc_assemble (R + sources + switches +
/// inductor v=0 stub) PATCHED to add BDF1 companion stamps for L/C.
///
/// Strategy: dc_assemble handles everything except the dynamic
/// terms (it leaves capacitors as open and inductors with a v=0
/// constraint plus a tiny ε on the diagonal for numerical
/// non-singularity). We then walk the L/C branches a second time
/// and:
///   * Capacitor: add the 4-entry conductance block with G_eq=C/dt.
///   * Inductor: undo the ε and add −L/dt on the diagonal.
inline void bdf1_assemble_segment(const topology::Graph& graph,
                                       const pwl::DevicePool& pool,
                                       const topology::SwitchStateMask& mask,
                                       Real dt,
                                       sparse::Matrix& J,
                                       Vector& b,
                                       Real t_eval = Real{0}) {
    // Step 1: dc_assemble for the algebraic part. dc_assemble
    // doesn't add the trap companion for L/C — caps are treated as
    // open, inductors get a v=0 constraint with a tiny diagonal ε
    // for numerical non-singularity.
    pwl::dc_assemble(graph, pool, mask, J, b, t_eval);

    if (dt <= Real{0}) return;

    // Step 2: add BDF1 companion stamps for capacitors + inductors.
    const Index nb = graph.num_branches();
    constexpr Real dc_assemble_inductor_epsilon = Real{-1e-12};
    for (Index bid = 0; bid < nb; ++bid) {
        const auto& branch = graph.branch(bid);
        if (branch.kind != topology::BranchKind::PassiveLinear) continue;
        const auto stored = pool.kind_of(branch.id);
        const stamping::BranchCoord coord{branch.from, branch.to};
        if (stored == pwl::DevicePool::StoredKind::Capacitor) {
            const auto& p = pool.capacitor_params(branch.id);
            const Real g_eq = p.C / dt;  // BDF1
            stamping::stamp_capacitor_companion(J, b, coord, g_eq);
        } else if (stored == pwl::DevicePool::StoredKind::Inductor) {
            const auto& p = pool.inductor_params(branch.id);
            const Index brid =
                pool.branch_var_id_for_inductor(branch.id, graph);
            // Undo dc_assemble's tiny ε and add −L/dt on the diagonal.
            J.coeffRef(brid, brid) -= dc_assemble_inductor_epsilon;
            J.coeffRef(brid, brid) -= p.L / dt;
        }
    }
    // Force re-compression after our coefficient touches.
    sparse::compress_in_place(J);
    (void)mask;
}


// =============================================================================
// run_transient_bdf1 — driver loop
// =============================================================================

[[nodiscard]] inline SimulationResult run_transient_bdf1(
    const builder::CircuitBuilder& builder,
    const SimulationOptions& opts,
    const SwitchScheduleFn& switch_fn,
    const BExtraFn& b_extra_fn = {},
    bool start_from_dc_op = false,
    const StepObserverFn& step_observer = {}) {

    if (!opts.valid()) {
        throw std::invalid_argument(
            "run_transient_bdf1: invalid SimulationOptions");
    }
    if (!switch_fn) {
        throw std::invalid_argument(
            "run_transient_bdf1: switch_fn is required");
    }

    const auto& graph = builder.graph();
    const auto& pool  = builder.pool();
    const Size state_size = pool.state_size(graph);
    const Size n_steps = opts.expected_step_count();

    SimulationResult result;
    result.reserve(n_steps);

    Vector x = Vector::Zero(static_cast<Index>(state_size));
    Bdf1HistoryState history{graph, pool};
    history.reset();

    // DC OP bootstrap.
    if (start_from_dc_op) {
        auto mask = switch_fn(opts.t_start);
        x = pwl::compute_dc_op(graph, pool, mask, opts.t_start);
        history.seed_from_dc_op(x);
    }

    // Per-(mask, t) we rebuild b for time-varying sources; we cache
    // the J factor by mask since BDF1 stamps don't depend on t
    // (only sources do, and sources only affect b).
    //
    // For Phase B.2 V1 we don't cache at all — assemble + factor on
    // every step. That's slow but simple; users who want speed can
    // use the trap path (which has the proper PwlStateSpaceCache).
    // The point of BDF1 is robustness on stiff problems, not raw
    // throughput.

    // Sample 0.
    result.times.push_back(opts.t_start);
    result.states.push_back(x);
    result.event_iteration_count.push_back(0);

    if (step_observer) {
        step_observer(opts.t_start, x);
    }

    for (Size k = 1; k < n_steps; ++k) {
        const Real t = opts.t_start +
                          static_cast<Real>(k) * opts.dt;

        if (step_observer) {
            step_observer(t, x);
        }

        const auto mask = switch_fn(t);

        // Assemble J + b for this (mask, t).
        sparse::Matrix J;
        Vector b_static;
        bdf1_assemble_segment(graph, pool, mask, opts.dt,
                                  J, b_static, t);

        // BDF1 history term.
        const Vector b_hist = history.compute_b_extra(opts.dt);
        // User b_extra.
        const Vector b_user = b_extra_fn
            ? b_extra_fn(t)
            : Vector::Zero(static_cast<Index>(state_size));
        const Vector b_total = b_static + b_hist + b_user;

        auto solver = sparse::make_default_solver();
        if (!solver->analyze(J)) {
            throw std::runtime_error(
                "run_transient_bdf1: structurally singular at t="
                + std::to_string(t));
        }
        if (!solver->factorize(J)) {
            throw std::runtime_error(
                "run_transient_bdf1: numerically singular at t="
                + std::to_string(t));
        }
        Vector rhs = -b_total;
        solver->solve(rhs, x);

        history.update_from_state(x, opts.dt);

        result.times.push_back(t);
        result.states.push_back(x);
        result.event_iteration_count.push_back(0);
    }
    return result;
}

}  // namespace pulsim::v2::solver
