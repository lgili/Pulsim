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

#include "pulsim/models/jiles_atherton.hpp"
#include "pulsim/models/saturable_inductor.hpp"
#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/pwl/device_pool.hpp"
#include "pulsim/pwl/trbdf2_stage.hpp"
#include "pulsim/stamping/branch_coord.hpp"
#include "pulsim/topology/graph.hpp"

#include <utility>
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
        //! Jiles-Atherton step-start state (H_n, M_n), meaningful
        //! only when params.ja is set. Committed on accept; a
        //! rejected step never touches it (evaluations integrate
        //! from it and never write it).
        models::JilesAthertonCore::State ja_state{};
        //! Jiles-Atherton state at the γ point (H_γ, M_γ) — the base
        //! the BDF2 stage integrates from. Written only by
        //! capture_gamma, from the SAME evaluation that fills
        //! lambda_gamma, so the two cannot disagree; lives between
        //! the two stages of one attempt like lambda_gamma does.
        models::JilesAthertonCore::State ja_gamma{};
        models::SaturableInductor::Params params;
    };

    /// The JA base state for a stage. The trapezoidal stage — and
    /// every fixed-engine solve, probe, landing and seed — integrates
    /// from the committed (H_n, M_n). The BDF2 stage integrates from
    /// the stage-1 converged (H_γ, M_γ): the three-point formula
    /// assumes λ_n, λ_γ, λ_{n+1} sample ONE trajectory, and for a
    /// path functional that trajectory runs THROUGH γ. Starting stage
    /// 2 from (H_n, M_n) coincides only while no reversal lies inside
    /// the step; with one, the n→γ irreversible leg is dropped — an
    /// O(h) error at every loop tip, twice a cycle, always shrinking
    /// the loop. Found by an adversarial check before this shipped.
    [[nodiscard]] static const models::JilesAthertonCore::State& ja_base(
        const Entry& e, TrBdf2Stage stage) noexcept {
        return stage == TrBdf2Stage::Bdf2Stage2 ? e.ja_gamma : e.ja_state;
    }

    /// λ and L at a trial current under whichever law the entry
    /// carries, in the given stage. The hysteretic law integrates
    /// from that stage's base state; the other two are stateless.
    [[nodiscard]] static std::pair<Real, Real> flux_and_inductance(
        const Entry& e, Real i,
        TrBdf2Stage stage = TrBdf2Stage::Trapezoidal) noexcept {
        if (e.params.ja) {
            const auto ev = models::JilesAthertonCore::evaluate(
                *e.params.ja, ja_base(e, stage), i);
            return {ev.lambda, ev.L};
        }
        return {models::SaturableInductor::flux(i, e.params),
                models::SaturableInductor::inductance(i, e.params)};
    }

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
            if (e.params.ja) {
                // Remanent state at zero current: H = 0, M = M0, and
                // the flux it links, so the first step sees no
                // phantom dλ/dt.
                e.ja_state.H = Real{0};
                e.ja_state.M = e.params.ja->M0;
                e.ja_state.n_sub = e.params.ja->substeps_min;
                e.lambda_old = models::JilesAthertonCore::flux_of(
                    *e.params.ja, Real{0}, e.params.ja->M0);
            }
            entries_.push_back(e);
        }
    }

    /// λ at the current carried by `x`, from the model's own law.
    [[nodiscard]] static Real lambda_at(
        const Entry& e, const Vector& x,
        TrBdf2Stage stage = TrBdf2Stage::Trapezoidal) noexcept {
        return flux_and_inductance(e, x[e.branch_var_id], stage).first;
    }

    /// Snapshot λ at the γ stage point; the BDF2 history term reads
    /// it. Must run between the two stages of a step. For the
    /// hysteretic law it also captures (H_γ, M_γ) — from the same
    /// evaluation — as the base the BDF2 stage integrates from.
    void capture_gamma(const Vector& x_gamma) {
        for (auto& e : entries_) {
            if (e.params.ja) {
                const auto ev = models::JilesAthertonCore::evaluate(
                    *e.params.ja, e.ja_state, x_gamma[e.branch_var_id]);
                e.lambda_gamma = ev.lambda;
                e.ja_gamma.H = ev.H;
                e.ja_gamma.M = ev.M;
                e.ja_gamma.n_sub = e.ja_state.n_sub;
                // The step's direction stays in force through both
                // stages (it is per STEP, not per stage).
                e.ja_gamma.delta_hint = e.ja_state.delta_hint;
            } else {
                e.lambda_gamma = lambda_at(e, x_gamma);
            }
        }
    }

    /// Read the freshly-converged `x` and stash the new
    /// (λ, i_L, V_L) as the next step's history.
    /// Called by `run_transient` after each Newton solve, and by the
    /// TR-BDF2 stepper only once a step is ACCEPTED.
    void update_from_state(const Vector& x,
                           TrBdf2Stage stage = TrBdf2Stage::Trapezoidal) {
        for (auto& e : entries_) {
            const Real v_from = stamping::read_node_voltage(x, e.from);
            const Real v_to   = stamping::read_node_voltage(x, e.to);
            e.i_L_old = x[e.branch_var_id];
            e.V_L_old = v_from - v_to;
            // From the model's law at the converged current, so the
            // committed flux and the committed current can never
            // describe different states. For the hysteretic law this
            // is also where the magnetisation history advances: the
            // (H, M) reached at the converged current becomes the
            // next step's start.
            if (e.params.ja) {
                // Commit from the SAME base the converged residual
                // used — (H_n, M_n) after a trapezoidal solve,
                // (H_γ, M_γ) after a BDF2 stage — or the committed
                // λ would not be the λ the circuit was solved with.
                const auto& base = ja_base(e, stage);
                const auto ev = models::JilesAthertonCore::evaluate(
                    *e.params.ja, base, e.i_L_old);
                // The coming step's sub-step count comes from THIS
                // step's whole field excursion (twice it, for
                // headroom): fixed for the step so every evaluation
                // inside it is one smooth function of the trial
                // current.
                const Real span = Real{2} * std::abs(ev.H - e.ja_state.H);
                // Next step's direction = this step's actual leg (a
                // zero leg keeps the current direction).
                const Real leg = ev.H - e.ja_state.H;
                e.ja_state.delta_hint =
                    leg > Real{0} ? Real{1}
                    : leg < Real{0} ? Real{-1} : e.ja_state.delta_hint;
                e.ja_state.n_sub =
                    models::JilesAthertonCore::substeps_for(*e.params.ja, span);
                e.ja_state.H = ev.H;
                e.ja_state.M = ev.M;
                e.lambda_old = ev.lambda;
            } else {
                e.lambda_old =
                    models::SaturableInductor::flux(e.i_L_old, e.params);
            }
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
        out.reserve(entries_.size() * 5);
        for (const auto& e : entries_) {
            out.push_back(e.lambda_old);
            out.push_back(e.i_L_old);
            out.push_back(e.V_L_old);
            // Hysteresis state; zeros for the stateless laws.
            out.push_back(e.ja_state.H);
            out.push_back(e.ja_state.M);
        }
        return out;
    }
    void from_flat(const std::vector<Real>& flat) {
        if (flat.size() != entries_.size() * 5) return;
        Size p = 0;
        for (auto& e : entries_) {
            e.lambda_old = flat[p++];
            e.i_L_old    = flat[p++];
            e.V_L_old    = flat[p++];
            e.ja_state.H = flat[p++];
            e.ja_state.M = flat[p++];
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
