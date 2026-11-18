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
