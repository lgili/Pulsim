# Decisions — refactor-unified-transient-stepper

Recorded while implementing. The proposal and design stand; each entry
says where the implementation departs from them or what it settled.

1. **Strangler order, not skeleton first.** Both drivers are edited by
   other tracks while this change runs (TR-BDF2 track 8 on
   `trbdf2_transient.hpp`, the ideal-switch and device-interface tracks
   on `run_transient.hpp`). The policies are extracted as free types
   under `core/include/pulsim/solver/step/` that the old loops call, with
   the loop bodies left in place; the loop bodies are replaced by the
   skeleton only last. Each commit names the regions of the two files it
   touched so the other tracks can merge.

2. **The byte-identity gate is local and four-part.** Baselines recorded
   on `feat/core-integration` 7ff35556 (macOS arm64, Apple clang, Eigen
   5): the golden set (21 cases with their kernel counters, i.e. the
   counter gate: factorisations, solves, cache hits, Newton counters),
   the 40-case stateful byte compare, the engines byte compare, and a
   step-identity oracle (`times` / `states` / `switch_states` hashes, the
   whole `_trbdf2_stats` dict, `commutation_events`, `dt_retries`,
   `inductor_guard_actions`, `event_iteration_count`) over the golden
   cases, the four TR-BDF2 turn-off-invariant cases and three extra
   configurations (tight TR-BDF2, restart + split on the fixed engine).
   "Byte-identical" below means all four compare equal.

3. **`StageSpec` carries what the drivers use, not the design's full
   struct.** `{formula, factor_dt, ctx_h, t_off}`: the step context the
   devices are told (`pwl::StepCtx` = `{ctx_h, formula}`), the factor key
   and where the stage ends. The design's `a`, `c2`, `c3` and `t_end`
   are added when a solve point takes the history right-hand side from
   the spec (task 2). `h_elapsed` is replaced by `t_off` (offset of the
   stage end from the step start): "the last stage's `t_off` is `h`" is
   exact in floating point, "the stage `h_elapsed` sum to `h`" is not.

4. **Each table value is the driver's own expression.** `TrBdf2::stage`
   computes `trbdf2_gamma() * h` (what `hg` was), the fixed engine's
   restart stage is `BackwardEuler::on_factor(for_dt)` with
   `ctx_h = for_dt * 0.5` (what `step_ctx.h` was). The fixed engine's
   SDIRK γs = 1 − 0.70710678118654752440 and the TR-BDF2 engine's
   γ/2 = (2 − √2)/2 are NOT unified: they differ in the last bit and
   unifying them changes results. The SDIRK(2,2) restart table is left
   for the integrator that needs it (task 15.2).

5. **The TR-BDF2 SDIRK stage keeps `StepCtx{γh, Trapezoidal}`.** Its
   linear history is backward Euler (derivative slots zeroed) but the
   step context says trapezoidal and stage 2 says `Bdf2Stage2`. Not a
   defect today: no stamp reads it, because SDIRK steps never run on a
   circuit with a stateful device (`sdirk_possible`, and a D4
   re-initialisation on such a circuit is refused,
   `update-trbdf2-post-event-restart` DECISIONS 9). Carried over
   unchanged; an `Sdirk22` table (task 15.2) must say `BackwardEuler`
   before that refusal is lifted.

6. **The controller state.** `step::ControllerState {kind, v}` is the
   snapshot's controller block: a kind tag (`None`, `Fixed`, `Lte`) and a
   flat vector in that controller's layout. `LteController::save()`
   writes `[h, h_mem_gate[0], h_mem_gate[1], pending_gate_slot,
   diode_slot (−1 = none), h_mem_diode...]`; `load()` refuses another
   kind. Names for the device-interface snapshot work
   (`add-unified-device-interface`, controller-state block):
   `SolverSnapshot::controller` of type `step::ControllerState`.
   The fixed engine needs none (its grid is `t_start + k·dt` and the
   restart count is already in the snapshot).

7. **Task 1 (one source accumulator) is a number change, so it waits.**
   The two drivers add the same terms in different association. Per
   right-hand-side row, the fixed engine computes
   `(history + b_extra_fn) + (0 + (−v_src))` (rows added in place, then
   the current waveforms, then the loss injection), the TR-BDF2 engine
   `history + ((0 + b_extra_fn) + (−v_src))` (a zeroed overlay of dense
   per-kind vectors added to the history at the end). One implementation
   must pick one association and moves the last bit of every circuit with
   both a source and a `b_extra_fn` (or a history term) on one row, on
   one engine. That belongs to the number-changing release, not to this
   refactor (proposal: "nothing moves until the legacy drivers are
   deleted"). Kept: each driver's accumulation as it is.

8. **Task 12, C++ side: the controller block rides in the snapshot
   object; it is not pickled yet.** `SolverSnapshot::controller`
   (`step::ControllerState`) is written by `run_transient_trbdf2` at the
   end of every run and loaded on a resume from a snapshot whose block is
   `Kind::Lte` (an explicit `h_init` still wins; any other kind is
   ignored, so pwl ↔ trbdf2 resumes are unchanged). The pickle format is
   being changed on `feat/core-p3-device-interface` (version 4 carries
   the tick grid); the block joins the pickle when that lands, so a
   pickled snapshot resumes as before (to tolerance). Measured, a
   TR-BDF2 run split at a time the continuous run stepped to (dt = h_max
   given, so the step bounds do not depend on the span), the second half
   resumed from the first's snapshot:

   | circuit | before (h restarts at h_max/100) | with the block |
   |---|---|---|
   | buck (PWM, diode) | first step differs, 1164 vs 1157 steps | times and states bit-identical, 1157 steps |
   | transformer | first step differs, 1004 vs 998 | bit-identical, 998 |
   | RLC + sine source | first step differs, 1004 vs 998 | times identical; states 1.2e-13 off |

   The RLC residue is the FIRST half, not the resume: its last step is
   clamped to `t_end − t`, which differs from the continuous run's `h`
   in the last bit although `t + h` rounds to the same time, so the
   first half ends 6e-15 V away. Making that exact (keep the proposal
   when `t + h` rounds to `t_end`) changes the last step of ordinary
   runs: a number change, not done here.

   What a bit-identical TR-BDF2 → TR-BDF2 resume still needs beyond the
   step memory (all in `run_transient_trbdf2`, none in the snapshot):
   the zero-time diode settle at the start of a resumed run (it probes
   and may re-decide bits the stopped run had settled; `x_sig` restarts
   as `x`), `gates_decided` / `gate_prev`, the gate-period learning
   (`gate_pulse_min`, `gate_bits`, `gate_frac`, `gate_sample_w`,
   `gate_jitter`), the probe floor learned (`h_probe_learned`,
   `last_probe_*`), the chatter guards (`chatter_*`, `held_redecide_t`),
   the restart counters (`sdirk_left`, `sdirk_event`, `reinit_left`; the
   fixed engine already carries its own as `restart_steps_left`), the
   post-event sample state (`event_sample_t`, `post_event_pending`) and
   the queue's edge cache. The buck and the
   transformer above resume bit-identically because none of these was
   live at the split; a split inside a restart window or right after an
   edge would not.

9. **Review of the controller block (fresh reviewer, 2026-11-18), fixed.**
   (a) `steady_state(engine='trbdf2')` resumed its base period from the
   warm-up snapshot (with the block) and its probe periods from
   hand-built snapshots (without it): the monodromy columns would have
   differenced two step sequences. The base period now starts from a
   hand-built snapshot too on that engine (pwl unchanged, byte for byte).
   (b) A run whose `t_end` cut its last step saved the controller's
   reaction to the cut step (down to ~h_min); it now saves
   max(that, the proposal the end cut short). (c) The gate-edge parity
   restarted at 0 in a resumed run, so the remembered post-edge step was
   applied to the other edge kind: the block now carries
   `gate_parity` (layout `[h, h_mem_gate[0..1], pending_gate_slot,
   diode_slot, gate_parity, h_mem_diode...]`) and the slot is
   `(n_gate_events + gate_parity) & 1` (0 on a fresh run: unchanged).
   (d) `load` validates (finite values, h > 0, slots in range) and
   changes nothing on refusal. (e) The FixedClock unit test compares two
   runtime expressions (a folded constant may contract differently).
