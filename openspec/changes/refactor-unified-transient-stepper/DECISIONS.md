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

10. **The skeleton exists before any engine runs on it.**
    `solver/step/stepper.hpp` is the D4 order as a function template,
    `step_once<I>(controller, body)`: poll → `begin` (instant
    processing) → `clamp(propose)` → [rollback copy] → `take` (the
    stage solves through the body's one `solve_stage(spec, s, frame)`;
    default `run_stages<I>`) → `error` → the controller's
    accept / retry / abort → `after_solve` (the event policy: accept, or
    truncate the frame and re-solve) → `commit` → `on_step`. Controllers:
    `FixedController` (the grid `t0 + k·dt`; the frame's `t1` is the grid
    point, never `t0 + h`; a step the event policy cut short is committed
    and the rest of the grid interval follows, `k` advancing only at the
    grid point; no error call and no retry branch compiled, and the
    rollback copy only when the body declares `localizes`) and
    `LteClock<I>` (the factor laws with the integrator's `err_exponent`;
    a failed solve or a non-finite error retries at h/4; below `h_min`
    aborts, rolled back). A step lands on its limit (grid point, `t_end`)
    or on the event time the body wrote, never on `t0 + h` recomputed;
    the next proposal comes from the step's full attempt, not from the
    sliver an event cut it to; a step re-solved 200 times aborts. The body is a duck-typed set of members, not a
    virtual interface (D1); the concepts `Integrator` and
    `StepController` are plain requires-expressions. Proven on a toy body
    (`core/tests/step/test_stepper.cpp`): orders 2 / 1 / 2 / 2 for
    trapezoidal, backward Euler, TR-BDF2 and SDIRK(2,2) on the fixed
    clock; both 2-stage methods under the LTE clock with a rejected first
    step; a NaN error retried; the phase order; abort before commit; the
    event truncation on both clocks (fixed: the remainder of the grid
    interval is integrated). Reviewed (fresh reviewer, 2026-11-18): the
    first version committed a cut fixed step and skipped the rest of the
    grid interval, never ended with the default `h_max = 0`, accepted a
    NaN error, took the next proposal from the event sliver, skipped the
    restore on an abort and recomputed `t0 + h` instead of landing on
    the event time; all fixed as above.
    Neither driver runs on it yet: the two loop bodies are under edit by
    three other tracks with large unmerged diffs (device interface −628
    lines of `run_transient.hpp`, ideal switch +256, TR-BDF2 track 8
    +271), so carving them into body members now would make every one of
    those merges a rewrite. They move after integration batch 13 lands.
    Planned mapping of the fixed loop (`run_transient.hpp` dynamic path):
    `poll` = the `should_continue` check; `begin` = step start through
    `mask_pre` (observer, split events, V3 snapshots); `take` = the solve
    state, the restart / split / event-iteration lambdas and the nominal
    attempt with the halving ladder (the event split stays inside `take`:
    it is pwl's sub-interval policy and continues from each event, while
    the skeleton's cut re-solves from the step start); `after_solve` =
    cycle handling and commutation records; `commit` = freeze guard,
    deferred history commit, restart bookkeeping, loss tracker accept;
    `on_step` = diagnostics, hooks, `store_every`, the structure record.
    The per-step locals that cross those members are hoisted (reset at the
    same program point) and the loop-scope lambdas defined once before the
    loop, with their text unchanged so the other tracks' hunks still apply.

11. **`steady_state(engine='trbdf2', dt=1e-7)` on the 24 V buck: not the
    step sequence; a derivative slot over a sliver step.** Reproducer:
    `probes/ss_trbdf2_sliver.py`. Every monodromy probe (relative 1e-4,
    1e-6, 1e-8) runs on exactly the base run's accepted grid, so the
    map is evaluated with a frozen step sequence already. What is not
    smooth is history slot 3, the capacitor's `i_prev` (the trapezoidal
    derivative slot, `2C/h·Δv − i_prev`): the user's `switch_fn` puts the
    edge 1e-9·T before the period boundary, TR-BDF2 lands on it, and the
    period ends with a 9.2e-15 s step (`h_post_event_min` in
    `_trbdf2_stats`; the 6.9e-14 s gap after the mid-period edge in
    `np.diff(times)` may be an event record sample, not a step), over
    which `2C/h` multiplies the rounding of `v` by ~1e10. The slot's
    column entry scales as 1/probe (a fixed jump of ~5.8e-7 A in multiples
    of one size, not a derivative). The size is ~60x below one ulp of
    `v` times `2C/h`, so the exact path of the rounding is not pinned:
    what is measured is a quantised, probe-independent jump in a slot the
    map otherwise forgets (its own column entry is 0: the slot is
    re-derived, not carried). Columns are usable only for probes 1e-4 to
    1e-6 (at 1e-8 the physical columns pick up the noise too). One
    period from the linear-solve answer moves that slot by 4.9e-3 while
    the physical slots move 1e-7. With the edges exactly on the boundary
    a second defect shows: a run resumed AT an edge time misses that edge
    (slot 0 moves −24 V in the second period), the "resume at a gate
    edge" gap of the event queue's edge cache (DECISIONS 8). Owners: the
    TR-BDF2 track (a gate edge within rounding of `t_end`, or the history
    commit after a sub-`h_min` post-edge step), the event queue (edge
    cache in the snapshot), or, without an engine change, `steady_state`
    itself (shoot on the physical slots only — capacitor `v_prev`,
    inductor `i_prev` — and let the period re-derive the derivative
    slots; the PSS owner). Not fixed here: each changes results.

12. **Bit-identical TR-BDF2 resume (task 12): measured, the settle stays,
    the rest waits for the loop.** Splits at a step time of the
    continuous run whose first half ends with the continuous step
    unchanged (no clamp effect, same state at the split; script in the
    worktree's `.scratch/resume_exact.py`, buck / RLC / bridge of
    `test_trbdf2_resume_controller.py` / an RLC with a sine source /
    `test_bridge_ground_truth._bridge`):

    | circuit | such splits | resumed run bit-identical |
    |---|---|---|
    | RLC + sine | 24 | 23 |
    | diode bridge | 2 | 2 |
    | buck, opaque `switch_fn` | 44 | 0 |
    | closed loop (`bind_pi_to_switch`) | — | refused by name: "a control graph resumes between two of its ticks" (event queue's tick origin) |

    The buck's first difference is the landing of the next gate edge
    (`1.99999802e-07` vs `1.99999800e-07`: the edge is located from the
    learned gate state, `gate_pulse_min` differs in the 14th digit) or a
    noise-held diode decision (`n_noise_held_decisions` 1 vs 2) — the
    gate-learning and hold state of DECISIONS 8, which are locals of the
    loop. Tried and refuted: carrying `x_sig` in the controller block and
    skipping the zero-time settle on a resume. It left every split above
    unchanged and broke
    `test_resume_just_before_a_gate_edge_matches_the_edge` (auto, 1 / 50 /
    1000 ulp): the settle reads the gates where the first step will
    (`t + h_min/2`), which the stopped run never did, so it is part of a
    correct resume, not a re-decision to skip. Reverted. The block grows
    when the loop's state is owned by the skeleton; the pickle carries it
    when `feat/core-p3-device-interface` defines the block (not there at
    385b4229).

13. **Slice 3a: the fixed engine's dynamic loop on the skeleton, as a
    twin; the phase map (the contract for slice 3b).** The loop body of
    `run_transient`'s dynamic path is cut, in place, into five lambdas
    that BOTH drivers call (nothing copied; each block's text moved
    only by its lambda header, so the ideal-switch track's hunks apply
    once and reach both paths). The per-step locals that cross phases
    are hoisted above them and (re)set where the body declared them
    (`commit_backward_euler`, used by take and commit, moved up with
    them). The legacy loop (`for k { if (step_cancelled()) break;
    step_begin(k); step_take(); step_after_solve(); step_commit();
    step_record(k); }`) stays the default; the twin is
    `step::run<step::Trapezoidal>(FixedController{fixed_grid, k_off+1,
    k_off+n_steps}, step::phase_body(...))`, selected only by the
    internal environment variable `PULSIM_PWL_STEPPER_TWIN=1` (read once
    per run; no option, no binding).

    | old-loop phase (dynamic path; line ranges at f69a2cbd) | skeleton slot | shared / copied |
    |---|---|---|
    | setup: IC / DC start sample 0, observer priming, `lt->begin`, `hooks->start`, restart window seed (`restart_left`), Newton round-off budget, resolution-diagnostics workspace (1574–1697) | before `run` (engine setup, not a slot) | shared (unchanged, runs before either driver) |
    | grid: `k = 1 .. n_steps−1`, `t = fixed_grid.at(k_off + k)` (1699, 1706–1709) | `FixedController` (clock = the run's `fixed_grid`, `k` global from `k_off + 1`); `begin` recomputes `t`, `t_prev` from the same clock at `f.k − k_off` | shared |
    | cancellation `should_continue` (1703) | `poll` | shared (`step_cancelled`) |
    | step start: `lt->advance`, `x_prev`, ideal-diode previous bits, observer / control ticks (queue), oracle and `b_extra_fn` jump caches reset, gate memo, the step's events (gate edges, source jumps, declared jumps; merge within `snap`), V3 rollback snapshots, `mask_pre`, unlocated-gate and overflow counters, diode combine (1710–1873) | `begin` | shared (`step_begin(k)`) |
    | — | `clamp` | identity: pwl places its events INSIDE the step (event split), never by shortening the frame |
    | solve: round-off note, `mode_solve`, `run_event_iteration` (diode event iteration + mode-selection sites), restart trigger and SDIRK(2,2) restart with the stage-2 fallback, `split_step` (sub-intervals per located event), the nominal attempt and the halving ladder (1891–2699) | `take` (returns true; a failed step throws, as the legacy loop) | shared (`step_take()`) |
    | — | `error`, retry branch, rollback copy | not compiled (`FixedController::uses_lte == false`, body does not localize) |
    | event-iteration breach (strict throw, or accept + bit resync + record), commutation records (2701–2830) | `after_solve` (never cuts: returns true) | shared (`step_after_solve()`) |
    | inductor freeze guard / clamp, deferred history commit (trapezoidal or backward Euler), restart counter, loss tracker `accept` (2832–2941) | `commit` | shared (`step_commit()`); twin adds `hooks->device_state` |
    | resolution diagnostics, `hooks->accepted`, `store_every` record, dependent-mode re-initialisation record (2943–3061) | `on_step` | shared (`step_record(k)`); twin adds `hooks->step` (R1) |
    | `++k` | `FixedController::advance` (k advances: no partial interval, `t1 == t_lim` always) | skeleton |
    | end: diagnostics finalisation, `restart_steps_left`, restart factorisations, outer catch, final snapshot `grid_step = k_off + k_reached` (3063–3075, ~3380) | after `run` | shared |
    | STATIC path (no dynamic device, 3075–3370) | not on the skeleton | the legacy loop on both settings |

    Copied: nothing of the loop; the twin's six one-line adapters
    (`tw_*`, `f.k − k_off`). For 3b (trbdf2) the map says what the
    skeleton must grow: `clamp` real (edge / crossing landing), `error`
    (`TrBdf2::lte`), the retry and rollback (the LTE clock), and
    `after_solve` cutting (localisation) — pwl needed none of them.
    Requests from other tracks, as of this slice:

    | request | slot | call |
    |---|---|---|
    | probes R1 (per accepted step, x_start + mask_start beside x_end) | `on_step` | yes, twin only: `StepHooks::step(t_prev, x_prev, mask_start, t, x, stamped_mask)`; `mask_start` (the previous stamp) copied only with a listener. Gap: pwl delivers the LEFT limit at t_prev (committed state, previous stamp); it solves no right limit at a grid-point edge (its re-initialisation happens inside the step), so the recorder's right limit at such an edge still needs a solve the engine does not make |
    | probes R2 (`on_event` after the settle) | inside `take` (pwl's events are sub-intervals of `split_step`) | no: needs a call in `split_step`, i.e. in shared code; not added |
    | probes R3 (pin currents at commit) | `commit` | via #31: a `device_state` listener calls `pwl/device_readings.hpp` |
    | device interface #31 (`on_device_state` after the histories' commit) | `commit` | yes, twin only: `StepHooks::device_state(t, x, hs)` |
    | device interface: `seed_held_machine_capacitors` hand PMSM refresh | setup (DC start), not the loop | not a slot; untouched |
    | ideal switch §6b L1–L7 | the mode helpers (`mode_lcp_site`, passivity flags) above the loop, called from `take` | shared code: applies once to both paths; not made (number changes) |
    | control graph: no queue instant at `t_end`; stamping hook `(x, h) → rows`; located crossing source | `begin` (queue) / `take` (stamping, crossing) | no |
    | high-order integrator hand-off 1–7 | variable-step slice | nothing added contradicts it (`PhaseBody` is pwl's; `run_stages` / `StageSpec` untouched) |

    Hooks only the skeleton calls are zero-cost without a listener (an
    emptiness test) and allocate nothing per step
    (`test_zero_alloc.cpp`, "fixed engine twin on the step skeleton").

    **Speed gate NOT met at first (resolved: cause and fix in 14).**
    Instructions retired per accepted step, in-process
    (`proc_pid_rusage` `ri_instructions` around `simulate()` only,
    min of 3–6 alternating runs; `.scratch/instr_inproc.py`; noise ≤ 0.2 %
    — f69a2cbd against itself reads 0.9990–1.0010):

    | case | f69a2cbd | default after | twin | default Δ | twin / default |
    |---|---|---|---|---|---|
    | buck_pwl | 3096 | 3138 | 3258 | +1.4 % | 1.038 |
    | vsi_pwl_restart | 4853 | 4966 | 5062 | +2.3 % | 1.019 |
    | vsi_pwl | 4705 | 4793 | 4894 | +1.9 % | 1.021 |
    | shockbuck_pwl | 18696 | 18724 | 18847 | +0.15 % | 1.007 |
    | rlc152_pwl | 18484 | 18494 | 18621 | +0.05 % | 1.007 |

    Gates: default ≤ 0.5 %, twin ≤ 1 %. Tried: (1) phases left to the
    inliner: default +1.9 % buck / +2.2 % VSI; (2) phases forced inline at
    both call sites: +0.4 % / +2.0 %, twin +4.4 % / +2.0 %; (3) (2) with
    the twin driver out of line: +1.0 % / +1.9 %; (4) (3) with the step
    times and `edge_restart` re-read into const locals per phase: no
    change. The default cost scales with switch / diode count (buck
    +32–42, VSI +88–113 instructions per step; none on the RLC ladder or
    the Shockley buck), so it is in per-switch work of a phase, not in the
    clock; not located. Next: compare the per-step disassembly of the
    default loop at f69a2cbd and here (`-S` of `ir_bindings.cpp`, the
    smallest TU that instantiates `run_transient`) around the
    commutation-record loop and `mask_pre`'s copy-assignment (a
    `SwitchStateMask` holds a `std::vector`; the loop body copy-
    CONSTRUCTED it per step).


14. **Slice 3a's speed cost, found: a hoisted mask's copy-ASSIGNMENT,
    and the twin's out-of-line adapters (2026-11-19; slice 3b must not
    repeat either).** Found by bisecting builds measured with the
    in-process probe (`.scratch/instr_inproc.py`, instructions retired
    per accepted step) and by the extension's disassembly (`otool -tvV`,
    the calls of the `run_transient` body). Two separate costs:

    (a) Default path. `mask_pre` was hoisted out of the loop body so
    every phase sees it. The body's `SwitchStateMask mask_pre =
    gate_at(gate_t);` (a copy-CONSTRUCTION from the reference `gate_at`
    returns into its memo: the empty heap vector costs a null test)
    became `mask_pre = gate_at(gate_t);`, a copy-ASSIGNMENT, whose
    defaulted form calls `std::vector<std::uint64_t>::__assign_with_size`
    OUT OF LINE on every copy, empty vectors included (13 such calls in
    the body against 12 at f69a2cbd). With the twin compiled out this
    one line was the whole default-path cost (buck 3141 → 3078, VSI
    4770 → 4712 once fixed). Fix at the cause:
    `SwitchStateMask::operator=(const&)` copies the two inline words
    and touches the vector only when either side holds heap words
    (the same mask bit for bit; `static_assert(kInlineWords == 2)`).
    The other per-step mask copy-assignments the PR line already had
    (`stamped_mask = mask`, `prev_gate = mask_pre`) get cheaper too, so
    the slice is also compared with f69a2cbd + the same class fix
    + fix A (the PR line + fix A, the loop this slice must equal) —
    "fair" below.

    (b) Twin. The five `tw_*` adapters were each emitted out of line
    (1200 / 1047 / 778 / 608 / 330 instructions), so the twin paid five
    calls per step and reached every hoisted local through two closure
    hops (1.038× buck, 1.019× VSI), and its out-of-line closure cost the
    default VSI loop +33 instructions per step. Fix: every adapter
    (`tw_*`, `PhaseBody`'s forwarders, `step_once`, `run`) is
    `PULSIM_STEP_PHASE` (forced inline), and EACH driver — the legacy
    loop (`run_legacy`) and the twin (`run_twin`) — is one out-of-line
    function (`PULSIM_STEP_DRIVER`, renamed from `PULSIM_STEP_TWIN`).
    The legacy loop alone in its function runs faster than inlined in
    `run_transient`'s 13k-instruction body (buck 3087 → 3034). Both
    macros are empty on MSVC, as the code base's other forced-inline
    macros (the gate is measured on clang).

    Instructions per accepted step, this tip (three runs each; f69a2cbd
    min of six; Shockley / RLC one run):

    | case | f69a2cbd | fair | default | twin / default |
    |---|---|---|---|---|
    | buck_pwl | 3080 | 3034 (3053–3056 typical) | 3040 / 3040 / 3042 | 1.0098 / 1.0096 / 1.0029 |
    | vsi_pwl | 4709 | 4693 | 4706 / 4704 / 4698 | 1.0060 / 1.0072 / 1.0077 |
    | vsi_pwl_restart | 4857 | 4843 | 4863 / 4868 / 4865 | 1.0060 / 1.0065 / 1.0079 |
    | shockbuck_pwl | 18689 | 18682 | 18652 | 1.0010 |
    | rlc152_pwl | 18499 | 18592 | 18416 | 1.0010 |

    Gates met: default ≤ f69a2cbd + 0.5 % (VSI + restart +0.2 %), and
    ≤ fair + 0.5 % (VSI +0.1–0.3 %, VSI + restart +0.4–0.5 %, the
    tightest row; buck, Shockley, RLC below it); twin ≤ 1.01× (buck 1.0098 is at the edge: the skeleton's own clock work,
    ~30 instructions per step). Noise on this loaded machine is larger
    than DECISIONS 13 measured: f69a2cbd against itself spans 0.6 % on
    the buck, and the SAME twin code moved 3050 → 3073 between two
    builds that changed only the legacy driver (code layout).

    For slice 3b (the TR-BDF2 loop): (1) a per-step object hoisted to
    cross phases turns its per-step initialisation into an ASSIGNMENT —
    check every class type's copy / move assignment for an out-of-line
    call (vector members) and prefer an in-place reset; (2) force every
    adapter between the skeleton and the phases inline and keep each
    driver ONE out-of-line function; (3) measure with the twin compiled
    out first (`if (false && twin)`): it splits the cut's cost from the
    twin's presence; (4) compare against the base with the same
    non-loop fixes, not only the PR line.

15. **Slice 3b: the variable-step loop on the skeleton, as a twin; the
    phase map (2026-11-19).** The TR-BDF2 loop body
    (`run_transient_trbdf2`, ~1,450 lines inside one `while`) is cut, in
    place, into eight lambdas that BOTH drivers call (`tb_live`, `tb_poll`,
    `tb_begin`, `tb_clamp`, `tb_take`, `tb_error`, `tb_after`,
    `tb_commit`; each block's text moved only by its lambda header, so the
    TR-BDF2 track's hunks apply once). The 21 locals that crossed phases
    are hoisted above them and (re)set where the body declared them (three
    `SwitchStateMask`s by move-assignment, `reinit_new` by `clear()`; the
    compiler listed them: `-fsyntax-only` after the cut, every
    "undeclared identifier"; a shadowing check found no phase-local that
    hides an outer name another phase uses). The legacy loop (`run_legacy`)
    stays the default; the twin (`run_twin`) is
    `step::run<step::TrBdf2>(BodyClock{tb_live, lte_ctrl, t, t_end},
    step::phase_body(tb_poll, tw_begin, tb_clamp, tb_take, tb_error,
    tb_after, tb_commit, tw_on_step))`, selected only by the internal environment
    variable `PULSIM_TRBDF2_STEPPER_TWIN=1` (read once per run).

    The inventory (`grep -n` of the loop's control transfers, a brace
    matcher for their targets; line numbers at the merge of b3bcfe32): the `while`
    has NINE exits from the straight path — `break` 2401 (cancellation),
    `break` 2727 (a singular final sliver ends the run), and `continue`
    2797 (conditioning floor: h = 4h), 2816 (Newton failure: h/4), 3021
    (non-finite: h/10), 3045 (LTE reject), 3079 (diode overrun: h/10),
    3355 / 3362 (chatter guard: probe-and-retake, then h/100), 3534
    (Newton failure on the landing: h_land/2). Every `continue` restarts
    the WHOLE body: the next attempt re-polls `should_continue`, counts
    `++steps`, re-runs the grind check, ticks, the caps and every landing
    for the new h, and nothing is rolled back — the attempt's learning
    (h, `h_cond_by_mode`, flipped diode bits, `chatter_*`, a dropped event
    sample, `sdirk_left`) is carried into the next attempt. A localised
    crossing is not a re-take either: `tb_after` solves the landing and
    COMMITS the shortened step itself (`commutated`), and the ordinary
    accept is skipped.

    | old-loop phase (lines before the cut) | skeleton slot | shared / copied |
    |---|---|---|
    | setup: options, DC / IC start, gate learning state, queue, histories, snapshot resume, lambdas (…–2396) | before `run` | shared (unchanged) |
    | loop condition `t < next(EndOfRun) − h_min && steps < max_steps` (2398) | `BodyClock::done` (= `!tb_live()`) | shared (`tb_live`) |
    | `should_continue` (2400) | `poll` | shared (`tb_poll`) |
    | `++steps`, grind refusal, ticks, monitors, `sw_now`, `h_lo`, `end_cut_h`, h clamp to `[h_lo, h_max]`, gate learning caps, `h_ctrl`, step-boundary settle (2403–2493) | `begin` | shared (`tb_begin`) |
    | tick landing, source-breakpoint landing, gate-edge landing (oracle or sampled bisection), no final sliver (2494–2649) | `clamp` (REAL: returns the landed h) | shared (`tb_clamp`) |
    | step mask, D4 entering test, SDIRK / TR stage 1, BDF2 / BE stage 2; conditioning floor and Newton failure → reject; singular final sliver → end (2651–2817) | `take` → Verdict Accept / Reject / End | shared (`tb_take`) |
    | LTE, Hosea–Shampine filter, round-off floor; non-finite → reject; forced accept (counted); LTE reject (2819–3046) | `error` → Verdict (the estimate AND the decision: the decision needs `h_lo`, the chatter window, `sdirk_event`, which are the body's) | shared (`tb_error`) |
    | crossing detection, overrun → reject, held / chatter guard, Illinois localisation on trial TRAP, the landing solve and its COMMIT + record (3048–3667) | `after_solve` → Verdict Accept (a commutated step is committed inside it) / Reject (a retake) | shared (`tb_after`) |
    | ordinary accept: history commit, `x`, `t += h`, record, `hooks->accepted`, D4 record, gate / breakpoint restart, next proposal (3669–3850) | `commit` (skipped by the body when `commutated`) | shared (`tb_commit`) |
    | — | `on_step` | twin only: the hooks (below); recording stays inside `commit` and the landing |
    | end: superseded sample, last monitors, max_steps refusal, stats (3852–3877) | after `run` | shared |

    What the skeleton grew (stepper.hpp): `Verdict::Reject` (the attempt
    is discarded; the next one starts at poll, no rollback) and
    `Verdict::End`, `StepStatus::Rejected`; `body_decides<B>()` (a body
    with `static constexpr bool decides = true`); in `step_once` a
    `decides` branch (one ATTEMPT: done → poll → begin → clamp → take →
    error → after_solve → commit → on_step; `run` loops on Accepted or
    Rejected); `BodyClock` (the body's time, end test and step memory
    `LteController`); `DecidingPhaseBody` + an 8-callable `phase_body` (the eighth is `on_step`). The
    fixed path's branch is the old `step_once` text in the `else` of an
    `if constexpr` (unchanged codegen).

    NOT grown, and why: the contract (13) expected a skeleton-owned retry
    with rollback (`save` / `restore`) and an `after_solve` that cuts and
    lets the skeleton re-take. The map says this loop has neither: its
    rejections carry state forward and re-run begin and clamp; its cut is
    solved and committed by the event policy itself. Moving the decision
    into the controller (`LteClock::on_result`) and a real rollback are a
    NUMBER change for this engine (the chatter window, the conditioning
    floor and the forced accepts are part of the decision), so they wait
    for the strategy work (task groups 14–15), not for this slice.

    Identity (this tip, macOS clang, local gate): the DEFAULT path is
    byte-identical to the merge baseline 9fb2f13e on the four compares
    (goldens incl. buck / VSI / RLC / Shockley / MOSFET / transformer
    trbdf2, stateful, engines, step sequences). The twin against the
    legacy loop, `benchmarks/tools/trbdf2_twin_identity.py`, every array
    and every stats key bit for bit: `--goldens --extra` 50/50 (7
    golden_baseline trbdf2 cases, 3 stateful rectifiers, the turn-off
    cases incl. the floating bridge, a closed loop with ticks AND a
    monitor (the monitor's call times compared too), 26 shipped examples,
    and resume across the drivers in all four combinations at two splits,
    one 1 ns after a turn-off edge, i.e. a run that ends inside the
    post-edge restart); `--reference --modes ',lcp,shadow'`: 327/327 (the 109
    reference circuits × default / 'lcp' / 'shadow'; 151 of them are the
    same refusal on both paths). A fresh reviewer found no divergence; its one
    real finding (on_step fired for an attempt that after_solve ended
    without a commit: a zero-time cascade or a retake at the same h) is
    fixed (no hook when t did not move).

    **Cost gate NOT met (default path).** Instructions retired per
    accepted step (`.scratch/instr_inproc.py`, min of 3; the base column
    is 9fb2f13e measured in the same process pattern, its own two columns
    agree within 0.06 %):

    | case | base | default | Δ | twin / default |
    |---|---|---|---|---|
    | buck_trbdf2 (events) | 10820 | 10906 | +0.80 % | 1.0016 |
    | bridge (floating bus, turn-off case) | 12577 | 12709 | +1.0 % | 0.9997 |
    | shockbuck_trbdf2 (Shockley, Newton) | 72214 | 72408 | +0.27 % | 1.0000 |

    Bisected with three builds: (A) twin compiled out (`if (false &&`):
    buck 10902, bridge 12718 — the twin's presence costs nothing, the cut
    itself does; (B) A + an inline `SwitchStateMask` move-assignment (the
    three hoisted masks are re-set by move): 10934 / 12705 — not the
    masks (reverted); (C) the legacy loop forced INLINE into the engine
    body instead of its own driver: 10964 / 12751 — worse, so rule 14 (2)
    holds here too. Remaining suspect, not yet measured: the 21 hoisted
    locals and the phase closures reached through the out-of-line
    driver's closure (reloaded after every opaque call: switch_fn, the
    solves), ~85–130 instructions per step independent of circuit size
    (Shockley's +194 is 0.27 %). Next: per-step disassembly of
    `run_legacy` against the base body (`otool -tvV`, count loads from the
    closure), then re-read hot hoisted scalars into const locals at each
    phase's top or pass them in one struct the driver owns.

    Because the gate is not met this tip is on
    `wip/core-p3-stepper-3b`, not on the track branch.

    Hooks (twin only, zero cost without a listener): R1 `StepHooks::step`
    and #31 `device_state` in on_step (after either commit). R2
    (`on_event` after the settle): StepHooks has no such member; not
    added (an API of the probes track). Event-queue request (a resume
    exactly at a gate edge misses the edge; the edge cache belongs in the
    snapshot): the cut makes it a one-place fix — the edge cache is read
    only in `tb_clamp` (`queue.next_after(edge_src, t)` and the sampled
    bisection) and written in `tb_commit` / `tb_after`, so seeding it from
    the snapshot is one statement in setup plus the snapshot field;
    behaviour not changed here.

16. **Slice 3b's speed cost, found: the out-of-line driver's closure
    hops; both TR-BDF2 drivers are forced inline (2026-11-19).** Found by
    the extension's disassembly (`otool -tvV`, base 9fb2f13e against the
    tip of 15) and two bisect builds measured with `.scratch/instr_inproc.py`.

    Cause. At the tip of 15 the legacy loop was its own out-of-line
    function (`run_legacy`, `PULSIM_STEP_DRIVER`, rule 14 (2)) with the
    phases inlined into it. Every engine local a phase touches (`t`, `h`,
    `x`, `st`, `opts`, the queue, the histories, the hoisted 21, …) is
    then reached as closure → phase closure → pointer → value: the driver
    spills its own closure pointer (`str x0, [sp, #0x68]`), keeps a few
    phase-closure pointers in callee-saved registers and RELOADS the
    captured pointer from the phase closure after every opaque call (the
    closures live in the engine frame and escape into the out-of-line
    driver, so no call can be proved not to change them). In the base the
    same access is one `ldr` off `sp`. Static histogram of the hot code
    (engine body + driver against the base body): `ldr` +342, `ldp` +197,
    `add` −329 (address formation `add xN, sp, #off` replaced by pointer
    loads), ~85–170 instructions per step. Not the masks, not the twin's
    presence (bisects A / B of 15).

    Fix. `run_legacy` AND `run_twin` are `PULSIM_STEP_PHASE` (forced
    inline into `run_transient_trbdf2`): no closure escapes, SROA removes
    them, and the locals are the engine frame's again (the engine body
    holds both loops, 26.3k instructions against 16.9k; no out-of-line
    lambda14 / 15; `ldr`:`add` back at the base's ratio). Build C of 15
    (legacy inline, twin still out of line) was WORSE because the twin's
    out-of-line closure still captured the phases, so they stayed
    escaped. The dispatch tests `!twin` first (the legacy loop is the
    fall-through: with the twin first, the twin hit the base exactly and
    the legacy loop sat at +0.4–0.6 %). Code layout only; one shared body,
    no copy of the loop. Rule 14 (2) is per engine: pwl's phases touch few
    locals and its drivers stay out of line.

    Instructions per accepted step (three runs each, min of 3 per run):

    | case | base 9fb2f13e | tip of 15 | this tip | Δ base | twin / default |
    |---|---|---|---|---|---|
    | buck_trbdf2 | 10816 / 10819 / 10824 | 10905 / 10922 / 10924 | 10854 / 10856 / 10856 | +0.35 % | 1.0001–1.0006 |
    | bridge (floating bus) | 12578 / 12591 / 12592 | 12698 / 12704 / 12709 | 12622 / 12623 / 12630 | +0.35 % | 1.0004–1.0007 |
    | shockbuck_trbdf2 | 72204 / 72205 / 72205 | 72391 / 72398 / 72414 | 72276 / 72277 / 72284 | +0.10 % | 0.9995–0.9997 |

    Gates met: default ≤ base + 0.5 %, twin ≤ 1.01× default. The fixed
    engine re-checked on the same two binaries (default unchanged; its
    codegen does not see the `decides` branch; tip vs base, min of 3): buck_pwl 3034 vs 3034,
    vsi_pwl 4703 vs 4707, vsi_pwl_restart 4863 vs 4864. Its twin / default
    on buck_pwl is 1.012 on BOTH (vsi 1.006–1.008): over 14's 1.01 since
    the merge of `fix/ci-147-l`, not this slice — open for the pwl twin.

    For the default-switch slice: the twin path IS the cheaper of the
    two when it is the fall-through (twin-first build: bridge 12580 = base,
    buck 10838 = +0.2 %), so making `step::run<TrBdf2>` the only driver is
    expected to cost nothing; measure it with the legacy loop deleted, not
    beside it. The inline macros are empty on MSVC and their placement on
    lambdas is compiled by Apple clang only: Linux / Windows unverified.
    Reviewer (fresh, read-only): no divergence; nits — a `Retry` / `Abort`
    verdict would stop the twin where legacy continues (no phase returns
    either), and `BodyClock::frame()` is sized from the proposal before
    `begin` changes `h` (no phase reads the frame).

17. **Slice 3c: the skeleton driver is THE driver on both engines; the
    static path on the skeleton; probes R2 (2026-11-19).** No number
    changes by construction: the default now calls the same phase
    lambdas through `step::run<…>` instead of the old loop.

    Switch. `run_transient` (dynamic AND static path) and
    `run_transient_trbdf2` call the skeleton driver unless the internal
    environment variable `PULSIM_PWL_STEPPER_LEGACY=1` /
    `PULSIM_TRBDF2_STEPPER_LEGACY=1` selects the old loop (read once per
    run; no option, no binding, no engine name — the brief overrides the
    design's public `pwl-legacy` / `trbdf2-legacy` names, which stay
    unbuilt; tasks 6.4 / 8.3 are done without them). The slice-3a/3b
    `…_STEPPER_TWIN` variables are accepted and ignored (nothing reads
    them; a script that sets them now compares the skeleton with itself).
    The skeleton is the dispatch's fall-through on both engines. The
    identity tools keep their names; `twin(on)` now clears / sets the
    LEGACY variable.

    Static path (no capacitor / inductor: a cache built at dt = 0, i.e.
    the raw `run_transient` binding; `simulate()` never reaches it). Its
    loop body is cut in place into `st_begin(k)`, `st_take()`,
    `st_after(k)`, `st_record(k)` (nothing to commit). Sample 0 (the first
    solve, `hooks->start`) runs before the skeleton's steps, as the dynamic
    path's IC sample; the skeleton runs k = 1 .. n_steps − 1 on the same
    `FixedController`. The static path never polled cancellation and still
    does not (`poll` is `true`). Both static drivers are forced inline into
    `run_transient` (out of line they cost +1.3–1.7 % per step on a 2.3k
    instruction step: the same closure hops as DECISIONS 16).

    Hooks, in the shared phases (both drivers) on the fixed engine; one
    test per step without a listener: the start mask for R1 / R2 is copied
    at the END of the previous step's record (under `hook_mask`), #31
    `device_state` moved into the record's existing `if (hooks)` before
    `accepted` (still after the commit; nothing between them changes x or
    the histories). TR-BDF2 keeps R1 / #31 in the skeleton's `on_step`
    (its legacy loop does not call them). R2 `StepHooks::on_event(t,
    x_left, mask_left, x_right, mask_right)`:
    TR-BDF2 at a gate edge (in `restart_after_gate_event`, after the
    settle: the committed solve at t under `x_mask`, the settled probe
    under the new stamp) and at a diode landing that commutated (the
    landed solve under the step's mask, the cascade's last probe).
    Fixed engine at the end of a step whose stamp differs from the
    start's: t = t_prev, x_left the committed state there under the old
    stamp, x_right the step's end under the new one (a bracket within one
    dt: pwl solves no right limit at the instant). Not reported (listed in
    step_hooks.hpp): TR-BDF2 edges with `refresh_after_event` off, a
    failed probe, entry into a dependent mode; fixed-engine changes that
    revert inside one step; a seeded start's first step (its stamp is a
    guess: `mask_start_known` from `stamped_known`, from the reviewer);
    the static path. Test: `test_switch_state_record.cpp` "Step hooks on
    the default driver" (fixed / TR-BDF2): every R2 side satisfies its own
    mask's device laws, gate events sit on (TR-BDF2) or within dt of
    (fixed) an edge, at least one right side violates the left mask's
    laws; R1 chains (fixed: always; TR-BDF2: the gate edge's restart
    commits inside the landing step's commit, so the chain does not break
    and that step's x_end is the post-edge state — the edge's left limit
    is R2's); #31 once per R1.

    **Cost gate NOT met on the fixed engine (trbdf2 met).** Instructions
    retired per accepted step (`.scratch/instr_inproc.py`, min of 3 per
    run; OLD = the starting tip 67fef260's default, the legacy loop, three
    runs; NEW = this tip's default, the skeleton).

    | case | OLD (3 runs) | NEW (3 runs) | NEW / OLD (min) |
    |---|---|---|---|
    | buck_pwl | 3034 / 3040 / 3042 | 3078 / 3077 / 3077 | 1.0140 |
    | vsi_pwl | 4698 / 4703 / 4706 | 4736 / 4741 / 4741 | 1.0082 |
    | vsi_pwl_restart | 4861 / 4868 / 4866 | 4907 / 4905 / 4901 | 1.0083 |
    | shockbuck_pwl | 18639 / 18643 / 18649 | 18676 / 18677 / 18675 | 1.0020 |
    | rlc152_pwl | 18395 / 18437 / 18428 | 18496 / 18590 / 18549 | 1.0055 |
    | static_pwl (raw binding, dt = 0 cache, 4 diodes + switch) | 2301 / 2317 / 2304 | 2304 / 2304 / 2304 | 1.0015 |
    | buck_trbdf2 | 10860 / 10859 / 10862 | 10863 / 10860 / 10864 | 1.0001 |
    | bridge_trbdf2 (floating bus) | 12633 / 12627 / 12649 | 12640 / 12638 / 12633 | 1.0004 |
    | shockbuck_trbdf2 | 72298 / 72286 / 72300 | 72258 / 72274 / 72270 | 0.9996 |

    The legacy loop of the same final build (the `twin` column with the
    LEGACY variable) reads buck 3045–3047, VSI 4704–4711: the shared-phase
    hooks cost the legacy loop ~+5–10 per step (one test per step), the
    skeleton driver itself another ~30.

    What was tried on pwl, each a build measured the same way (buck_pwl
    NEW, legacy loop of the same build in brackets; OLD 3034–3042):
    (s1) switch only, drivers out of line, R1/#31 in the skeleton adapters:
    3064 [3039]; (s3) + `FixedController::advance_whole` (frame times dead,
    `partial` constant): 3066 [3040], no change; (s4) both drivers forced
    inline (3b's layout): 3086 [3086] — worse for both, rule 14 (2) holds
    for pwl; (s5) drivers out of line, hooks moved into the shared phases
    so the skeleton driver's closure holds only the phases: 3055 [3055];
    (s6) one hook test per step: 3066 [3045]; (s7) a `whole_steps` fast
    path in `step_once` (the legacy loop's shape): byte-identical code to
    s6 (the generic path already folds), reverted with advance_whole.
    Disassembly (`otool`/`objdump` of the two out-of-line drivers, s6/s7):
    same size (4298 vs 4234 instructions) and callee set, but the skeleton
    driver reloads from the stack 253 times against the legacy loop's 152
    (`ldr x, [sp…]`; ldr total 1032 vs 943): register pressure from the
    skeleton's live state (clock counter and bound, the body's references,
    `k_off`), ~20–30 instructions per step — the same signature 3a found.
    Not removed in this slice. Also: identical code moves by ±0.7 % between
    builds that change unrelated code (layout), so the pwl gate sits at the
    noise floor; the honest reading is "skeleton +0.5–1.1 % on buck,
    +0.6–0.8 % on VSI, ≤ 0.5 % elsewhere". Next: give `FixedController` a
    loop-shaped `run` overload (k in a register, the bound a const local)
    or measure with the legacy loop deleted (its presence shapes the
    allocator, 3b's lesson).

    Engine function sizes (instructions, `nm` + c++filt): `run_transient`
    body 9252 → 12389 (+34 %: both static drivers now inline in it), the
    two out-of-line dynamic drivers 4204 / 4089 → 4298 / 4234;
    `run_transient_trbdf2` 26346 → 26418.

    Not done in this slice: the deletion of the legacy loops (next slice,
    after CI compiles this on GCC, Clang + libc++ and MSVC; the inline
    macros are empty on MSVC and their placement on lambdas is compiled by
    Apple clang only); the decision / rollback in the controller (groups
    14–15); a public `integrator=` option. Because the pwl cost gate is not
    met this tip is on `wip/core-p3-stepper-3c`, not on the track branch.


    Identity (this tip, macOS clang, local gate): default AND both-legacy
    byte-identical to the starting tip 67fef260's default on the four
    compares (goldens, stateful, engines, step sequences);
    `pwl_twin_identity.py --goldens --reference` 379/379;
    `trbdf2_twin_identity.py --goldens --extra` 50/50 (resume across the
    drivers, four ways) and `--reference --modes ',lcp,shadow'` 327/327.
    Static path (`.scratch/static_identity.py`: rectifier ± switch,
    `store_every=3`, the discontinuity rule, shadow, lcp, strict (a
    refusal, same on both), PWM source, cancellation, one step): skeleton
    vs legacy 11/11 bit for bit, and the new default 11/11 against the
    starting tip's binary.

    Reviewer (fresh, read-only, opus medium): no divergence of result,
    record or snapshot from the default switch — `store_every`,
    cancellation (`k_reached` / `grid_step`), resume across drivers,
    exceptions (no skeleton rollback, nothing driver-specific after the
    call), the static cut verbatim (n_steps 0 / 1 identical). Findings
    fixed: R2 at a seeded start's first step reported a guessed stamp
    (now `mask_start_known`); R2's unreported cases documented in
    step_hooks.hpp. Accepted: TR-BDF2's R1 / #31 stay skeleton-only (its
    legacy loop is silent); the `…_TWIN` variables are silently ignored.

18. **Slice 3c verified and landed; the pwl cost (2026-11-19).**

    Verification of b6600dd4 as it stood (owed by 17):
    - C++ targets `pulsim_core_step_tests` / `_dsed_tests` / `_perf_tests`,
      Eigen 5 and Eigen 3.4.0: all pass (step 27 cases / 518 assertions,
      dsed 128 / 36567 — the two new "Step hooks on the default driver"
      cases included — perf 13 / 2089, the edited zero-alloc case included).
    - pytest (repo root): default 5560 passed / 108 skipped / 8 xfailed;
      the same in each of PULSIM_PWL_STEPPER_LEGACY=1,
      PULSIM_TRBDF2_STEPPER_LEGACY=1 and both; `pytest -m reference` 377;
      `reference_status` exit 0; examples (PULSIM_RUN_EXAMPLES=1) 136;
      openspec strict 78/78; stubs and ruff clean. Static path 11/11.
      Nothing failed; nothing to fix.

    Cost, measured again (`.scratch/ab2.sh`: the old binary and the new
    interleaved in separate processes, default path only, each run the min
    of 6 simulate() calls; three runs; OLD = 67fef260 rebuilt from
    `git archive` with the same flags):
    | case | OLD | b6600dd4 | ratio |
    | buck_pwl | 3030/3034/3036 | 3056/3078/3069 | 1.0087 |
    | vsi_pwl | 4702/4702/4698 | 4739/4741/4737 | 1.0084 |
    | vsi_pwl_restart | 4867/4868/4861 | 4907/4904/4901 | 1.0082 |
    | shockbuck_pwl | 18633/18632/18623 | 18661/18658/18660 | 1.0019 |
    | rlc152_pwl | 18487/18446/18428 | 18488/18495/18429 | 1.0001 |

    Experiment (a), legacy fixed-step drivers (dynamic AND static) DELETED
    (throw-away build of b6600dd4 minus `run_legacy` / `st_run_legacy` /
    the env read), same harness:
    | buck_pwl | OLD 3040/3040/3024 | (a) 3076/3074/3076 | 1.0164 |
    | vsi_pwl | 4704/4702/4710 | 4741/4739/4748 | 1.0080 |
    | vsi_pwl_restart | 4866/4867/4871 | 4902/4904/4908 | 1.0072 |
    | shockbuck_pwl | 18626/18644/18643 | 18674/18660/18677 | 1.0018 |
    | rlc152_pwl | 18467/18472/18491 | 18483/18464/18499 | 0.9999 |
    The gap does not close: not an artefact of carrying both drivers.
    Disassembly (b6600dd4, `objdump` of the two out-of-line dynamic
    drivers): the skeleton driver's prologue loads its closure's nine
    captures (the five phase closures, cancellation, k_off, the grid,
    n_steps) ONCE, because the `tw_*` adapters and the PhaseBody are locals
    of the driver whose reference members SROA turns into SSA values; nine
    values live across the whole loop take the callee-saved registers, and
    they are spilled and reloaded from four stack slots 112 times (sp+0x98
    alone, the take phase's closure, 58). The legacy loop reaches the same
    closures through its one spilled `this` (21 reloads of sp+0x58). Stack
    reloads 202 vs 101.

    (b) b12 = adapters + PhaseBody built OUTSIDE the driver (its closure
    captures the body, the grid, k_off, n_steps; the body is memory the
    driver reaches through one pointer, as the legacy loop reaches its
    closures) + FixedClock::k0 (the skeleton's clock counts the run's own
    step k, so no adapter subtracts k_off): skeleton driver 4176
    instructions (legacy 4238), stack reloads 86 (legacy 101).
    | buck_pwl | OLD 3039/3030/3037 | b12 3060/3041/3059 | 1.0038 |
    | vsi_pwl | 4701/4702/4702 | 4721/4722/4722 | 1.0042 |
    | vsi_pwl_restart | 4867/4867/4866 | 4883/4887/4882 | 1.0033 |
    | shockbuck_pwl | 18630/18624/18624 | 18666/18654/18644 | 1.0011 |
    | rlc152_pwl | 18461/18452/18448 | 18471/18448/18468 | 1.0000 |
    b5 = b12 with the tw_* adapters removed: `step::IndexPhaseBody` (in
    stepper.hpp) holds the loop's OWN phase callables and calls them with
    the clock's step (`begin(f.k)`, `record(f.k)`, poll = !cancelled):
    driver 4170 instructions, 925 ldr (legacy 4238 / 945), 86 stack
    reloads (legacy 101).
    | buck_pwl | OLD 3039/3019/3032 | b5 3054/3053/3034 | 1.0051 | (b12 same round 3052/3055/3058, 1.0111)
    | vsi_pwl | 4702/4701/4702 | 4717/4716/4717 | 1.0032 | (b12 1.0039)
    | vsi_pwl_restart | 4868/4866/4867 | 4879/4881/4879 | 1.0027 | (b12 1.0037)
    | shockbuck_pwl | 18628/18635/18631 | 18648/18660/18650 | 1.0011 |
    | rlc152_pwl | 18446/18442/18473 | 18478/18440/18453 | 0.9999 |
    Skeleton vs legacy loop INSIDE the b5 build (same binary): buck
    3058/3049 vs 3028/3049, VSI 4714/4717 vs 4715/4712 — the skeleton
    driver now costs nothing over the legacy loop; what remains against
    OLD is shared by both drivers (the shared-phase hooks of 3c).
    Re-check, OLD vs b5: static_pwl 2300.2 vs 2299.2 (0.9996),
    buck_trbdf2 10855.8 vs 10857.8 (1.0002), bridge_trbdf2 12621.5 vs
    12631.0 (1.0008), shockbuck_trbdf2 72271.6 vs 72255.8 (0.9998).

    THE GATE, two independent builds per side (the second of each with a
    layout shake: a small used function prepended to python/bindings.cpp),
    interleaved, min of 3 runs each:
    | case | OLD | OLD shaken | NEW (b5) | NEW shaken | NEW/OLD | NEWs/OLDs |
    | buck_pwl | 3035.7 | 3036.1 | 3049.8 | 3050.6 | 1.0046 | 1.0048 |
    | vsi_pwl | 4701.7 | 4699.2 | 4716.5 | 4717.1 | 1.0031 | 1.0038 |
    | vsi_pwl_restart | 4864.0 | 4863.5 | 4882.0 | 4878.7 | 1.0037 | 1.0031 |
    | shockbuck_pwl | 18639.3 | 18639.7 | 18656.0 | 18654.0 | 1.0009 | 1.0008 |
    | rlc152_pwl | 18454.7 | 18441.7 | 18426.4 | 18405.4 | 0.9985 | 0.9980 |
    Gate met on every case on both pairs (buck the tightest, +0.46-0.49 %).

    What changed (b5, this tip): `step::IndexPhaseBody` / `index_phase_body`
    in stepper.hpp; `FixedClock::k0` (default 0, so every existing clock is
    unchanged; test_controllers checks the offset clock is the same
    expression on the same grid point); run_transient builds the body
    outside `run_twin` and runs the clock on the run's own k (dynamic and
    static path). The phases, the legacy loop and every number are
    unchanged. The hooks' no-listener cost is one test per step (`if
    (hook_mask)`, a run-invariant flag; the record phase's `if (hooks)` was
    there before), shared by both drivers; not specialised further (the
    whole remaining residual, buck +0.46 %, is within the gate).

    Reviewer (fresh, read-only, opus medium) on 67fef260..this tip: no
    divergence of results, records, stats, snapshots, exceptions,
    cancellation (`k_reached` / `grid_step`), resume across drivers,
    `store_every` (the phases get the run's own k as before), the static
    path (n_steps 0 / 1 / 2). Noted, accepted: TR-BDF2's R1 / #31 fire on
    the skeleton only (as 17 recorded), so its LEGACY switch drops them
    while pwl's does not; on a TR-BDF2 commit that also commits a post-edge
    probe, two rows / two `accepted` but one `step` / one `device_state`;
    hook order differs between the engines (pwl `device_state` before
    `accepted`; TR-BDF2 R2 before the landing row, pwl after); latent:
    `FixedController::advance` never advances k on a NaN grid point (the
    old `for` always ended) — unreachable today (`valid()` requires finite
    `t_start` / `dt`, the snapshot origin is checked); the zero-alloc test
    leaves `PULSIM_PWL_STEPPER_LEGACY=0` set (harmless).

    The committed tip itself (a third NEW build: b5 + comments), same
    harness, min of 3: buck_pwl 3036.3 → 3050.5 (1.0047), vsi_pwl 4696.1 →
    4710.5 (1.0031), vsi_pwl_restart 4860.3 → 4874.0 (1.0028), shockbuck_pwl
    1.0008, rlc152_pwl 1.0009, static_pwl 0.9996, buck_trbdf2 1.0000,
    bridge_trbdf2 1.0008, shockbuck_trbdf2 0.9998.

    Identity (this tip, macOS clang, LOCAL gate): default AND both-legacy
    byte-identical to 67fef260's default on the four compares (goldens,
    stateful, engines, step sequences; baselines recorded with the old
    binary); `pwl_twin_identity.py --goldens --reference` 379/379;
    `trbdf2_twin_identity.py --goldens --extra` 50/50 and `--reference
    --modes ',lcp,shadow'` 327/327; static path 11/11; the twin tests
    24 passed / 1 skipped.

    Lands on `feat/core-p3-stepper` with the legacy loops KEPT (behind the
    internal variables) until CI has compiled the skeleton's inline macros
    on GCC, Clang + libc++ and MSVC (empty on MSVC; placement on lambdas
    compiled by Apple clang only: Linux / Windows unverified). Their
    deletion is prepared on `wip/core-p3-stepper-delete-legacy` if it
    exists (not for integration until the orchestrator says so).

19. **Deletion of both legacy loops, prepared (2026-11-19; NOT for
    integration until CI has compiled the skeleton's inline macros on GCC,
    Clang + libc++ and MSVC).** Branch `wip/core-p3-stepper-delete-legacy`
    on top of the slice-3c tip (DECISIONS 18).

    Deleted: the fixed engine's `run_legacy` (dynamic) and `st_run_legacy`
    (static), the TR-BDF2 `run_legacy`, both environment reads
    (`PULSIM_PWL_STEPPER_LEGACY`, `PULSIM_TRBDF2_STEPPER_LEGACY`; a script
    that still sets them now runs the skeleton), and the two Python twin
    tests (`test_pwl_stepper_twin.py`, `test_trbdf2_stepper_twin.py`: with
    one driver they compared it with itself). Kept, owed to the integration
    of this branch: the identity tools `pwl_twin_identity.py` /
    `trbdf2_twin_identity.py` (they now compare the skeleton with itself;
    delete or repoint them), the zero-alloc case's `set_pwl_twin` (sets a
    variable nothing reads; the case still holds, both iterations run the
    skeleton; NOT recompiled on this branch).

    Identity, with no twin left: the default of this branch against the
    stored OLD baselines (67fef260's default): byte-identical on the four compares (goldens, stateful, engines, step sequences); `pytest -m reference` pytest -m reference 377 passed; full pytest on the branch tree 5535 passed / 108 skipped / 8 xfailed (the 25 twin tests gone; one skip more than expected, not yet identified). Cost
    (instructions per step, OLD vs this branch, min of 3): buck_pwl 3035.8 → 3050.5 (1.0048), vsi_pwl 4700.9 → 4714.3 (1.0029), vsi_pwl_restart 4865.2 → 4881.3 (1.0033), shockbuck_pwl 1.0009, rlc152_pwl 0.9996, static_pwl 1.0009, buck_trbdf2 0.9994, bridge_trbdf2 1.0000, shockbuck_trbdf2 0.9997 — the same as the 3c tip with the loops kept (one build; no shake build of this branch).
