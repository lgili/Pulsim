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
