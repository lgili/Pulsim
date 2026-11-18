#pragma once

// =============================================================================
// Pulsim — Layer 5: SimulationResult (transient output container)
// =============================================================================
//
// `pulsim-v2-solver-and-events` Phase 1.
//
// Value-type aggregate holding the recorded `(t, x)` pairs from
// `run_transient`. Two parallel vectors:
//   * `times[k]`  → simulation time of sample k
//   * `states[k]` → state vector at `times[k]`
//
// `reserve(n)` pre-allocates both; `run_transient` calls it
// before the time-stepping loop to avoid per-step reallocation.
//
// v2.0 Phase 1 (audit finding `waveform-storage-vector-of-vectors`):
// `states` is no longer a `std::vector<Vector>` (one heap block per
// sample) but a `StateTrajectory` — ONE contiguous row-major
// buffer, allocated once, handed to Python as a zero-copy 2-D
// numpy view. The read API (`states[k]`, `size()`, `back()`,
// range-for) is unchanged except that element access yields an
// `Eigen::Map<const Vector>` by value; bind with `const auto&` or
// `const Vector&`, never a non-const `auto&`.
//
// Decimation now exists too: `SimulationOptions::store_every`
// records every m-th step (a PURE stride — steps 0, m, 2m, …, so
// the recorded grid stays strictly uniform at m·dt; the final step
// is included only when it lands on the stride), so a long
// high-fidelity run no longer has to hold every sample.

#include "pulsim/numeric/dense.hpp"
#include "pulsim/numeric/types.hpp"
#include "pulsim/solver/state_trajectory.hpp"
#include "pulsim/topology/switch_state.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

namespace pulsim::solver {

/// Layer 5 V2.2 — sub-step commutation timing diagnostic.
/// `t_estimated` is the linear-interpolated zero crossing of the
/// watched signal (v_diode − turn_on_voltage for both directions on
/// the pwl engine, see run_transient's interp_commutation_time)
/// between two grid points; with the fixed-step event split on it is
/// the located crossing the step was split at (not a grid index). The
/// state vectors
/// themselves remain at the dt grid; only the COMMUTATION
/// timestamp is sub-step accurate.
/// add-branch-form-ideal-switch D4: one impulsive re-initialisation. The
/// mode entered at `t` makes `states` dependent (sum_j s_j x_j = d); the
/// states jump by the projection x+ = x- - M^-1 K^T lambda with M = L (a
/// cut-set: lambda is the flux in V*s) or C (a capacitor loop: the charge
/// in C), and `dW = lambda * (K x- - d) / 2` J leaves the circuit.
struct ReinitRecord {
    Real t = 0;                      ///< start of the step that carried it
    std::string kind;                ///< "cut_set" | "capacitor_loop"
    std::vector<std::string> devices;  ///< the ideal elements involved
    std::vector<std::string> states;   ///< the inductors / capacitors
    Real lambda = 0;                 ///< V*s (cut-set) or C (loop), signed
    Real dW = 0;                     ///< energy removed [J], >= 0
    Real projection_gap = 0;         ///< max |x_proj - x(t+h)| over states
    long sample = -1;                ///< stored sample of the pulse, or -1
};

struct CommutationEvent {
    Real  t_estimated;
    Index branch_id;
    bool  new_state;
};

/// Phase-0 fix #9 — one record per time step where the diode
/// event-iteration loop could NOT reach a stable diode state:
/// either the iteration budget ran out, or a mask CYCLE was
/// detected (mask A → B → A …, the classic un-hysteresed diode
/// pair around a resonant node, which no budget resolves).
///
/// Pre-fix behaviour was a hard throw that discarded the whole
/// run ("minutes of compute lost at t = 37 ms"). The solver now
/// accepts the last consistent solve, records this breach, and
/// continues; the Python layer surfaces a loud warning. Set
/// `SimulationOptions::strict_event_iterations = true` to
/// restore the old throw.
struct EventIterationBreach {
    Real t;                 ///< step time of the breach
    Size iterations;        ///< solves attempted this step
    bool cycle_detected;    ///< true = mask cycle, false = budget
};

/// v2.0 Phase 2 (B.4) — a step the solver could not take at the
/// nominal dt, and had to re-take as 2^`halvings` sub-steps.
///
/// The output grid is UNCHANGED: sub-steps are internal, and a
/// sample is still only recorded at a nominal grid point. What this
/// records is that the trajectory between two samples was integrated
/// more finely than the user asked for — which is a change in
/// accuracy, not in the sampling, and the user should be told.
struct DtRetry {
    Real t;             ///< step time that had to be re-taken
    Size halvings;      ///< 2^halvings sub-steps were used
    std::string reason; ///< what the nominal-dt attempt reported
};

/// v2.0 Phase 2 — a post-solve guard OVERWROTE the solver's answer
/// for an inductor's branch current.
///
/// `inductor_freeze_di_max` and `inductor_abs_clamp` do not solve
/// anything: they replace a number the solver produced with one the
/// user configured. That is sometimes the only way to get a run to
/// finish, but it means the current being plotted is the LIMIT, not
/// the circuit — on the drive this was written for, the reported
/// line current peaks at exactly 100.000 A because the clamp is
/// 100 A. A guard that fires is evidence the model is missing
/// something (usually a snubber across a path that opens), and the
/// user has to be told, or they will read the limit as physics.
///
/// One record per inductor that fired, not per step.
struct InductorGuardAction {
    Index branch_id = kInvalidIndex;
    Size freeze_count = Size{0};   ///< steps snapped back to i_prev
    Size clamp_count  = Size{0};   ///< steps hard-limited
    Real t_first = Real{0};        ///< when it first fired
    Real worst_solved = Real{0};   ///< largest |i| the solver produced
    Real reported_limit = Real{0}; ///< the |i| the user sees instead

    [[nodiscard]] Size total() const noexcept {
        return freeze_count + clamp_count;
    }
};

/// v2.0 Phase 4 — the COMPLETE state of a run at one instant.
///
/// `initial_state=` restores only the MNA vector and then INVENTS
/// the companion history from it (a capacitor gets i_prev = 0, an
/// inductor v_prev = 0), so resuming from it does not reproduce
/// the run: a continuous 2T RLC and a T-then-resume differ by
/// 2.3e-4 where a true resume is ~1e-15. A snapshot carries the
/// parts that were missing, so `resume_from=` is exact on the
/// fixed-step engine. TR-BDF2 fills one and resumes from one too,
/// to the run's own tolerance: its step controller's state (last
/// accepted h, the LTE history) is not carried, so the resumed
/// segment restarts from h_init.
struct SolverSnapshot {
    Real t = Real{0};
    /// MNA unknowns at `t`.
    Vector x;
    /// Trapezoidal companion history: 2 reals (v_prev, i_prev)
    /// per dynamic device, in HistoryState::entries() order.
    std::vector<Real> history;
    /// Which switched diodes were conducting. Solver-owned bits
    /// that a mask alone cannot reconstruct.
    std::vector<bool> diode_on;
    /// The switch / diode mask STAMPED for the solve that produced
    /// `x` — one bit per switch-kind branch, in SwitchStateMask
    /// order: the record's last row
    /// (`SimulationResult::switch_state_bits`) whenever that row is
    /// the sample at `t`. It is not `diode_on` + switch_fn(t): where a
    /// run ends exactly on a gate edge, the variable-step engine
    /// stores the edge sample under the PRE-edge mask and then
    /// re-decides the diodes under the post-edge one, so `diode_on`
    /// (what the NEXT step starts from) and switch_fn(t) both describe
    /// the post-edge circuit while `x` is the pre-edge solve. A resume
    /// records this mask for its row 0, solved; empty when unknown
    /// (a hand-built or older snapshot), and that row is then flagged
    /// unsolved.
    std::vector<bool> stamped_mask;
    /// The five stateful-device histories, each in its own
    /// to_flat() layout; empty when the circuit has none. Before
    /// they were carried, a resume restored x and the linear
    /// companions while every stateful device restarted from its
    /// init value and the first Newton step reconciled the two by
    /// jumping — measured: a saturable inductor lost 8 mWb of flux
    /// at the seam (171.85 A continuous vs 27.14 A resumed), a Coss
    /// went from 109.85 V to 0.28 V on the first resumed step, an
    /// IGBT tail of 17.7 A vanished. No error was raised. Each
    /// history's from_flat() refuses a size mismatch by name.
    ///
    /// 7 per device: lambda_old, i_L_old, V_L_old, then the JA
    /// hysteresis state H, M, branch direction, sub-step count
    /// (zeros for the stateless laws). Covers the saturable,
    /// gapped-core and hysteretic-core inductors and the saturable
    /// transformer's magnetising branch — they share one history.
    std::vector<Real> saturable_history;
    /// 3 per device: v_prev, q_prev, i_prev (charge-based Coss).
    std::vector<Real> coss_history;
    /// 2 per device: q_M_prev, f_prev (Lauritzen stored charge).
    std::vector<Real> lauritzen_history;
    /// 2 per device: q_prev, f_prev (IGBT turn-off tail charge).
    std::vector<Real> igbt_tail_history;
    /// 9 per machine: lambda_abc, i_abc, v_abc (MNA PMSM).
    std::vector<Real> pmsm_history;
    /// Restart steps still owed (discontinuity_restart): the rule takes
    /// the step of a discontinuity and the next two as SDIRK(2,2)
    /// steps, so a run stopped inside that window resumes it. 0 = none.
    Size restart_steps_left = 0;
    /// True once populated by a run.
    bool valid = false;
};

/// `m` as one bool per bit (SolverSnapshot::stamped_mask).
[[nodiscard]] inline std::vector<bool> mask_bits(
    const topology::SwitchStateMask& m) {
    std::vector<bool> out(static_cast<std::size_t>(m.size()));
    for (Size i = 0; i < m.size(); ++i) {
        out[static_cast<std::size_t>(i)] = m.get(i);
    }
    return out;
}

/// The snapshot's `stamped_mask` as a mask of `num_switches` bits, or
/// false when it carries none of that width (a hand-built or older
/// snapshot).
[[nodiscard]] inline bool snapshot_stamped_mask(
    const SolverSnapshot& s, Size num_switches,
    topology::SwitchStateMask& out) {
    if (s.stamped_mask.size() != static_cast<std::size_t>(num_switches)) {
        return false;
    }
    out = topology::SwitchStateMask(num_switches);
    for (Size i = 0; i < num_switches; ++i) {
        out.set(i, s.stamped_mask[static_cast<std::size_t>(i)]);
    }
    return true;
}

/// Newton linear-algebra counters of one run
/// (update-newton-persistent-solver). Zero on a run without Newton.
struct NewtonCounters {
    std::uint64_t solves            = 0;  ///< Newton calls
    std::uint64_t iterations        = 0;  ///< Newton iterations
    std::uint64_t analyses          = 0;  ///< symbolic analyses
    std::uint64_t factorizations    = 0;  ///< full numeric factorisations
    std::uint64_t partial_refactors = 0;  ///< low-rank refactors (stage B)
    std::uint64_t lm_attempts       = 0;  ///< LM factorisation attempts
    /// Iterations of each Newton call, in call order (saturates at 65535).
    std::vector<std::uint16_t> iterations_per_solve;
};

struct SimulationResult {
    std::vector<Real> times;

    /// Newton counters of the run (update-newton-persistent-solver).
    NewtonCounters newton;

    /// v2.0 Phase 4 — the run's final state, complete enough to
    /// resume from exactly (`simulate(resume_from=...)`).
    SolverSnapshot final_snapshot;

    /// Recorded state samples, contiguous (v2.0 Phase 1).
    /// `states[k]` is the state at `times[k]`.
    StateTrajectory states;

    /// v2.0 Phase 2 — steps re-taken at a reduced dt. Empty on a run
    /// that never needed one.
    std::vector<DtRetry> dt_retries;

    /// Steps taken as SDIRK(2,2) restart steps
    /// (SimulationOptions::discontinuity_restart). 0 with the rule off.
    Size n_restart_steps = 0;
    /// Factorisations made for the restart stages (one per mask at
    /// 2γ·dt, more only if the segment byte budget evicts). 0 off.
    Size n_restart_factorizations = 0;
    /// Steps split at an event inside them
    /// (SimulationOptions::enable_substep_state_correction). 0 off.
    Size n_split_steps = 0;
    /// update-fixed-step-loop-bookkeeping loop counters (`res.counters`):
    /// `switch_fn` evaluations in the step loop, segment lookups the
    /// last-segment memo did not absorb, and rollback snapshots taken.
    /// add-fixed-step-resolution-diagnostics (`pwl` dynamic path).
    struct RingingRecord {
        Index var = kInvalidIndex;   ///< state-vector entry
        Real t_start = 0;            ///< end of the run's first alternating step
        Size length = 0;             ///< alternating steps in the run
        Real max_delta = 0;          ///< largest |x_k − x_{k−1}| in the run
    };
    struct Resolution {
        bool enabled = false;
        /// Per stored sample: max over the steps since the previous one
        /// of max_i (h/12)|f_{k+1} − 2f_k + f_{k−1}| / running peak |y_i|
        /// (f = dy/dt of capacitor voltage / inductor current). 0 where
        /// the stencil is invalid (mask change, restart, split, retry,
        /// source jump, first two steps).
        std::vector<Real> lte;
        Real lte_max = 0;
        Index lte_branch = kInvalidIndex;
        Real lte_time = 0;
        Size n_branches_covered = 0;
        /// Fraction of the steps with no estimate (stencil invalid): an
        /// unestimable run is not a resolved one.
        Real unestimated_fraction = 0;
        std::vector<RingingRecord> ringing;
    };
    Resolution resolution;
    Size switch_fn_calls = 0;
    Size segment_searches = 0;
    Size rollback_snapshots = 0;
    // Restart stage-2 fallback: backward-Euler re-tries, and those kept.
    Size stage2_retries = 0;
    Size stage2_be_kept = 0;
    Size stage2_retry_failures = 0;
    // Event splitting: steps whose gate changed with no edge oracle to
    // locate it (taken at the next grid point), and steps whose event
    // list overflowed (kMaxStepEvents or the per-step search cap).
    Size unlocated_gate_steps = 0;
    Size split_event_overflows = 0;
    // add-branch-form-ideal-switch D3: structural passes run (one per
    // distinct mode of a circuit with an ideal branch-form element).
    Size mode_structure_checks = 0;
    /// Ideal diodes turned OFF because they closed a loop of sources and
    /// closed ideal elements (add-branch-form-ideal-switch D3).
    Size ideal_loop_releases = 0;
    /// add-branch-form-ideal-switch D4: impulsive re-initialisations.
    std::vector<ReinitRecord> reinit_records;

    /// v2.0 Phase 2 — inductors whose current a post-solve guard
    /// overwrote. Empty unless `inductor_freeze_di_max` or
    /// `inductor_abs_clamp` was set AND actually fired.
    std::vector<InductorGuardAction> inductor_guard_actions;

    /// Phase-0 fix #9 — steps where diode event iteration hit a
    /// cycle or the budget. Empty on a fully-converged run.
    std::vector<EventIterationBreach> event_iteration_breaches;

    /// Per-step count of how many `cache.solve` invocations were
    /// needed before the diode state stabilised. Parallel to
    /// `times` and `states`. Zero means "first solve was already
    /// consistent" (no diode flips, or no diodes in the circuit
    /// at all).
    ///
    /// Diagnostic only — Layer 5 V2.1 populates it when diodes
    /// are present; otherwise it's left empty (Layer 5 V0/V1
    /// behaviour).
    std::vector<Size> event_iteration_count;

    /// Layer 5 V2.2 — list of estimated commutation times for
    /// every diode flip detected during the simulation. Ordered
    /// by `t_estimated`. Empty if no diodes are in the circuit
    /// or no flips occurred.
    std::vector<CommutationEvent> commutation_events;

    // -----------------------------------------------------------------
    // Per-sample record of the STAMPED switch / diode mask.
    // -----------------------------------------------------------------
    //
    // `switch_state_bits` holds, for every stored sample k, the
    // SwitchStateMask the solver STAMPED for the solve that produced
    // `states[k]` — the mask whose conductances are consistent with
    // the stored vector, so a switched branch's current is exactly
    //     i_k = (bit ? g_on : g_off) · v_k
    // (minus g_on·V_f for an ON SwitchedDiode with forward drop V_f).
    // It is NOT the mask the step after it starts with: a PWL diode
    // whose voltage crosses V_th in the last solve of a step is
    // re-decided by `update_from_state` AFTER that solve, and on the
    // variable-step engine an event landing is committed under the
    // PRE-commutation mask before the bits flip. The voltage alone
    // cannot recover it (a conducting diode with g_on = 1e3 carrying
    // 48 A sits at 48 mV, inside 0 < v_D < V_th, where ON and OFF
    // are both self-consistent), which is why it is recorded here.
    //
    // Layout: flat, `switch_state_stride` = ceil(n/8) BYTES per
    // sample, bit i of the mask in byte i/8 at bit i%8 (LSB first —
    // numpy's unpackbits(bitorder='little')). Bit i is the
    // SwitchStateMask index `switch_index_of` returns (Switch-kind
    // branches in builder order; a MOSFET's body diode is its own
    // bit, interleaved with the switches). Parallel to `times`:
    // bits.size() == times.size() * stride. Byte-packed rather than
    // word-packed so a converter with a handful of switches pays one
    // byte per sample, not eight.
    //
    // `switch_state_unsolved` lists the (few) sample indices whose
    // state was NOT produced by a solve under the recorded mask —
    // today only a sample 0 seeded from `initial_state`, for which
    // the recorded mask is the bits the solver CARRIED into the run,
    // not a solve (the state may come from a solve with a diode ON
    // that the carried bits have OFF). A `resume_from` sample 0 is
    // not listed when the snapshot carries `stamped_mask` (the mask of
    // the solve that produced its state, recorded as that row); one
    // without it (hand-built, or from before the field) is. Sparse,
    // so it costs nothing on a long run.
    //
    // Decimation (`SimulationOptions::store_every` > 1): row k is the
    // mask of the LAST solve before sample k, not of the whole stored
    // interval (t[k-1], t[k]], which spans several solves. KCL against
    // the record stays exact at every stored sample, but a time
    // fraction (duty) or a time integral over the stored grid then
    // aliases: a 50 kHz buck at dt = T/200 read M1 duty 0.5537 and
    // P_avg +14 % with store_every = 67 (0.5001 at store_every = 1).
    //
    // `has_switch_states` is false on a result no engine recorded
    // (a default-constructed or hand-built result); consumers must
    // then fall back and say so.
    std::vector<std::uint8_t> switch_state_bits;
    Size switch_state_stride = 0;
    Size num_switch_bits = 0;
    std::vector<Size> switch_state_unsolved;
    bool has_switch_states = false;

    /// Transient Newton solves that ended at the iteration budget and
    /// were accepted at roundoff level (add-newton-roundoff-acceptance:
    /// stagnated ||dx|| below 1e-6·|x|, row-wise backward error at most
    /// 8 eps), the largest backward error among them, and its MNA row.
    Size newton_roundoff_accepts = 0;
    Real newton_roundoff_omega_max = 0;
    Index newton_roundoff_row = kInvalidIndex;

    /// Start the record for a circuit with `num_switches` mask bits.
    /// Call once, before the first sample is stored.
    void begin_switch_state_record(Size num_switches, Size n_reserve = 0) {
        num_switch_bits = num_switches;
        switch_state_stride = (num_switches + 7) / 8;
        switch_state_bits.clear();
        switch_state_unsolved.clear();
        if (n_reserve > 0 && switch_state_stride > 0) {
            switch_state_bits.reserve(n_reserve * switch_state_stride);
        }
        has_switch_states = true;
    }

    /// Append the mask stamped for the sample just pushed onto
    /// `times` / `states`. `solved = false` marks a sample whose
    /// state is not a solve under `m` (see `switch_state_unsolved`).
    void record_switch_state(const topology::SwitchStateMask& m,
                             bool solved = true) {
        if (!has_switch_states) {
            return;
        }
        const Size nw = m.num_words();
        for (Size b = 0; b < switch_state_stride; ++b) {
            const Size w = b / 8;
            const std::uint64_t word = w < nw ? m.word(w) : 0ULL;
            switch_state_bits.push_back(static_cast<std::uint8_t>(
                (word >> (8 * (b % 8))) & 0xFFULL));
        }
        if (!solved && !times.empty()) {
            // Index of the sample just recorded (call this right
            // after pushing onto `times`).
            switch_state_unsolved.push_back(times.size() - 1);
        }
    }

    /// Bit `bit` of the mask stamped for sample `k`.
    [[nodiscard]] bool switch_state(Size k, Size bit) const {
        const Size stride = switch_state_stride;
        if (!has_switch_states || bit >= num_switch_bits ||
            (k + 1) * stride > switch_state_bits.size()) {
            throw std::out_of_range(
                "SimulationResult::switch_state: no recorded mask bit "
                "for this (sample, switch)");
        }
        return ((switch_state_bits[k * stride + bit / 8] >> (bit % 8)) &
                1U) != 0U;
    }

    /// True when the record is present and parallel to `times`.
    [[nodiscard]] bool switch_states_complete() const noexcept {
        return has_switch_states &&
               switch_state_bits.size() ==
                   times.size() * switch_state_stride;
    }

    [[nodiscard]] Size num_steps() const noexcept {
        return times.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return times.empty();
    }

    /// Pre-allocate space for `n` samples on all internal
    /// vectors. Does NOT change `num_steps()` — capacity only.
    ///
    /// Pass `state_size` (v2.0 Phase 1) so the contiguous sample
    /// buffer can be allocated in ONE shot for the whole run; the
    /// no-size overload leaves it to the first recorded sample.
    void reserve(Size n) {
        times.reserve(n);
        states.reserve(n);
        event_iteration_count.reserve(n);
        if (switch_state_stride > 0) {
            switch_state_bits.reserve(n * switch_state_stride);
        }
    }

    void reserve(Size n, Index state_size) {
        states.set_state_size(state_size);
        reserve(n);
    }

    /// Estimated bytes held by the recorded waveform (contiguous
    /// sample buffer + the parallel time / diagnostic vectors).
    [[nodiscard]] std::size_t approx_bytes() const noexcept {
        return states.bytes() +
               times.size() * sizeof(Real) +
               event_iteration_count.size() * sizeof(Size) +
               switch_state_bits.size() +
               newton.iterations_per_solve.size() * sizeof(std::uint16_t) +
               switch_state_unsolved.size() * sizeof(Size) +
               commutation_events.size() * sizeof(CommutationEvent) +
               event_iteration_breaches.size() *
                   sizeof(EventIterationBreach) +
               inductor_guard_actions.size() *
                   sizeof(InductorGuardAction) +
               // Each DtRetry carries the full Newton failure text
               // (a few hundred bytes with the row naming Phase 1
               // added), so the strings dominate the structs.
               [this] {
                   std::size_t n = 0;
                   for (const auto& r : dt_retries) {
                       n += sizeof(DtRetry) + r.reason.capacity();
                   }
                   return n;
               }();
    }
};

/// Thrown when a transient cannot continue — and it brings the run
/// with it.
///
/// v2.0 Phase 2. A simulation that dies at 90 % used to return
/// nothing at all: the exception carried a message and every sample
/// computed before the failure was destroyed with the stack. On a
/// run that takes minutes that is the difference between "here is
/// where it broke, and here is the waveform leading into it" and
/// "start again".
///
/// The default remains an exception, deliberately. Returning a
/// truncated result as if it were a whole one is exactly the silent
/// wrong answer this project keeps removing; a caller who wants the
/// partial trace has to ask for it by catching this type.
class SimulationAborted : public std::runtime_error {
public:
    SimulationAborted(const std::string& what,
                       SimulationResult partial,
                       Real t_failed)
        : std::runtime_error(what),
          partial_(std::move(partial)),
          t_failed_(t_failed) {}

    /// Everything that WAS computed, up to but not including the
    /// step that failed.
    [[nodiscard]] const SimulationResult& partial() const noexcept {
        return partial_;
    }

    /// The step time the solver could not reach.
    [[nodiscard]] Real t_failed() const noexcept { return t_failed_; }

private:
    SimulationResult partial_;
    Real t_failed_;
};

}  // namespace pulsim::solver
