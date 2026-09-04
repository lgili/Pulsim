#pragma once

// =============================================================================
// Pulsim — Layer 4 V17: Saturable-inductor history tracker
// =============================================================================
//
// Parallel to `HistoryState` (which tracks linear inductors + capacitors),
// this stand-alone tracker holds the previous-step state of each
// saturable inductor: `(i_L_old, V_L_old)`. The Newton refresh function
// reads from here at each iteration; `update_from_state` is called by
// `run_transient` after each successful Newton solve.
//
// Architecturally separate from `HistoryState` to avoid disturbing the
// existing trap-companion machinery in Layer 4. Saturable inductors
// live as `BranchKind::Nonlinear` (assemble skips them); their
// constraint row + KCL are stamped 100 %% via the refresh.
//
// THE STATE IS THE FLUX, NOT THE CURRENT. `lambda_old` is committed
// from the model's own λ(i) at the converged point, never
// re-integrated from voltages, so the stored flux cannot disagree
// with the current the circuit was actually solved with. `i_L_old`
// and `V_L_old` are kept alongside it because the trapezoidal
// residual reads V_L_old directly and callers report i_L_old.
//
// TR-BDF2 SECOND STAGE. The composite step needs λ at the γ point:
//
//     trapezoidal   λ_new − λ_n     = (h/2)(v_new + v_n)
//     BDF2 stage 2  c1·λ_new + c2·λ_γ + c3·λ_n = h·v_new
//
// λ is what makes the second stage WORTH HAVING — not what makes
// it expressible. That distinction is worth stating precisely,
// because the stronger claim is tempting and false.
//
// Since c1 + c2 + c3 = 0, the stage-2 history is
// (c1(λ_new − λ_n) + c2(λ_γ − λ_n))/h. Only the first increment
// contains the unknown; the second is pure history, contributing
// nothing to the Jacobian. And stage 1 has already enforced, at
// its own convergence, λ_γ − λ_n = (γh/2)(v_γ + v_n) — so that
// second increment folds into ONE history scalar of exactly the
// same shape as V_L_old, and the old rectangle stamp could have
// carried a BDF2 stage after all:
//
//   R = v_new − (c1/h)·L(i_new)·(i_new − i_n)
//             − (c2·γ/2)·(v_γ + v_n)
//
// That form was built and run: it converges at every step and
// returns 1.025 / 0.481 / 0.234 / 0.115 A of residual DC on a
// closed excursion at h = 2e-6 … 2.5e-7, halving with h. FIRST
// ORDER, and rectifying. The exact-λ form returns 3e-11 A.
//
// So the real argument is the accuracy one, and it is enough:
// L(i_new)·Δi is a right-endpoint rectangle rule for λ_new − λ_n
// with error −(1/2)·L'(i)·Δi². The residual divides that by h, and
// Δi = O(h), so it is an O(h) local truncation error — it drags a
// second-order, L-stable TR-BDF2 stage down to first order and
// brings the rectified DC drift with it.

#include "pulsim/models/saturable_inductor.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <vector>

namespace pulsim::pwl {

class SaturableInductorHistory {
public:
    struct Entry {
        Index branch_id;
        Index from;
        Index to;
        Index branch_var_id;   // absolute row in state vector
        Real  i_L_old = Real{0};
        Real  V_L_old = Real{0};
        //! λ(i_L_old) from the model's own law — the actual state.
        Real  lambda_old = Real{0};
        //! λ at the γ stage point; read only by the BDF2 stage, and
        //! only between the two stages of one step.
        Real  lambda_gamma = Real{0};
        models::SaturableInductor::Params params;
    };

    /// Build the entry list from the device pool + graph.
    /// Called once at the start of `run_transient`.
    void init(const topology::Graph& graph,
               const DevicePool& pool) {
        entries_.clear();
        for (const Index b_id :
                 pool.saturable_inductor_branches()) {
            const auto& branch = graph.branch(b_id);
            Entry e;
            e.branch_id = b_id;
            e.from = branch.from;
            e.to   = branch.to;
            e.branch_var_id =
                pool.branch_var_id_for_inductor(b_id, graph);
            e.params =
                pool.saturable_inductor_params(b_id);
            entries_.push_back(e);
        }
    }

    /// λ at the current carried by `x`, from the model's own law.
    [[nodiscard]] static Real lambda_at(const Entry& e,
                                        const Vector& x) noexcept {
        return models::SaturableInductor::flux(x[e.branch_var_id],
                                                e.params);
    }

    /// Snapshot λ at the γ stage point; the BDF2 history term reads
    /// it. Must run between the two stages of a step.
    void capture_gamma(const Vector& x_gamma) {
        for (auto& e : entries_) {
            e.lambda_gamma = lambda_at(e, x_gamma);
        }
    }

    /// Read the freshly-converged `x` and stash the new
    /// (λ, i_L, V_L) as the next step's history.
    /// Called by `run_transient` after each Newton solve, and by the
    /// TR-BDF2 stepper only once a step is ACCEPTED.
    void update_from_state(const Vector& x) {
        for (auto& e : entries_) {
            const Real v_from = stamping::read_node_voltage(x, e.from);
            const Real v_to   = stamping::read_node_voltage(x, e.to);
            e.i_L_old = x[e.branch_var_id];
            e.V_L_old = v_from - v_to;
            // From the model's law at the converged current, so the
            // committed flux and the committed current can never
            // describe different states.
            e.lambda_old =
                models::SaturableInductor::flux(e.i_L_old, e.params);
        }
    }

    /// Roll the flux back. The TR-BDF2 stepper never needs this — it
    /// commits only on accept — but the FIXED engine re-takes a
    /// non-converging step at dt/2 after the history has already
    /// moved, and without this the retry would integrate from a
    /// state the circuit never reached.
    [[nodiscard]] std::vector<Entry> snapshot() const { return entries_; }
    void restore(const std::vector<Entry>& snap) { entries_ = snap; }

    /// Flat (λ, i, v) triples per device, for state export / resume.
    /// `lambda_gamma` is not carried: it lives only between the two
    /// stages of one step, and a snapshot is always taken at a step
    /// boundary.
    [[nodiscard]] std::vector<Real> to_flat() const {
        std::vector<Real> out;
        out.reserve(entries_.size() * 3);
        for (const auto& e : entries_) {
            out.push_back(e.lambda_old);
            out.push_back(e.i_L_old);
            out.push_back(e.V_L_old);
        }
        return out;
    }
    void from_flat(const std::vector<Real>& flat) {
        if (flat.size() != entries_.size() * 3) return;
        Size p = 0;
        for (auto& e : entries_) {
            e.lambda_old = flat[p++];
            e.i_L_old    = flat[p++];
            e.V_L_old    = flat[p++];
        }
    }

    /// Seed `(i_L_old, V_L_old)` from a warm-start state vector (the
    /// `initial_state` passed to `run_transient`), so the first Newton
    /// refresh sees consistent previous-step values instead of zeros.
    /// Mirrors `HistoryState::seed_from_dc_op`; reads the same fields as
    /// `update_from_state`. No-op when the circuit has no saturable
    /// inductors. Call AFTER `init()`.
    void seed_from_dc_op(const Vector& x) { update_from_state(x); }

    [[nodiscard]] const std::vector<Entry>&
    entries() const noexcept { return entries_; }

    [[nodiscard]] bool empty() const noexcept {
        return entries_.empty();
    }

private:
    std::vector<Entry> entries_;
};

}  // namespace pulsim::pwl
