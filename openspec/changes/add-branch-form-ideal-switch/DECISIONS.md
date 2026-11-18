# Decisions taken while implementing (feat/core-a-ideal-switch)

Each entry: question → option taken → evidence → how to reverse.

1. **`g_off = 0` in the conductance form.** Design "Decisions taken" #2
   says refuse it until the flip; tasks 2.1 / 1.1 say `g_off` finite and
   `>= 0`. → Kept legal (tasks 2.1). → Refusing it would remove the
   characterisation of task 1.1 (±396.8 V on `pwl`, 2.35e9 V on
   `trbdf2`), which is the measured reason the branch form exists, and
   would change no valid circuit's numbers either way; the branch form
   is the remedy the refusal would point to, and it is opt-in in this
   slice. → Reverse: one line in `CircuitBuilder::require_conductances_`
   (`g_off > 0`) and flip `test_goff_zero_*` to expect the refusal.

2. **Where the branch form lives in the capability table.** A row per
   `StoredKind` (D9) would need new kinds, and the branch form is a form
   of the Switch and Diode kinds (same mask bit, same parameters). →
   A separate table `pwl::kBranchFormCaps` ("Switch": pwl, trbdf2;
   "Diode": pwl), checked by `require_branch_form_supported` first in
   `require_supported`, by `run_transient_bdf1` and the kernel AC sweep;
   bound as `_pulsim._branch_form_capabilities()`, mirrored in a second
   docs block and checked by `test_engine_conformance.py` (one cell per
   form and engine; no silent cell). → The diode is pwl-only because its
   decisions are re-read through the solved current only in the fixed
   engine's sites (`DiodeEventState`, the split's crossings);
   `trbdf2_transient.hpp` is not ours to edit. → Reverse: add the engine
   bit to the row once its decision sites read `i_s`.

3. **Branch-form diode decisions.** The voltage law `decide(on, v)` reads
   `i = g_on·(v − V_f)` when ON. → An ON branch-form diode is entered in
   `DiodeEventState` with `g_on = 1` and reads `v = V_f + i_s`
   (`DiodeEntry::law_voltage`), so `decide` and `crossing_signal` get
   `i_s` (amperes) and every decision site stays the same code; OFF reads
   `v_a − v_k` as before. → Conductance-form entries are untouched
   (bit-identical: golden probe). → Reverse: a dedicated
   `decide_branch(i_s, w)` in `models::SwitchedDiode`.

4. **Islands bounded by OFF ideal diodes: refused, not a gauge (task
   4.2).** The design's gauge (complementarity D10) reports the potential
   at the centre of the interval the diodes allow. That is a per-mode
   replacement of the island's rows, which breaks D1's mask-invariant
   pattern (one symbolic analysis), and the centre equation couples the
   diode thresholds into the matrix. → `floating_node` with `kind =
   'Diode'` when any boundary element is a diode, naming every boundary
   element and the nodes; the message says the gauge is not implemented
   and gives the remedies (`g_off > 0`, a component to ground). With
   `g_off > 0` (the default 1e-9 in every test circuit) no island exists.
   → Reverse: implement D10 as a per-mode gauge row on a reserved
   structural slot (keeps the pattern), then drop the refusal for
   diode-bounded islands.

5. **TR-BDF2 dependent states (task 5.5 refused).** The per-mode pass
   needs a hook in the TR-BDF2 step loop (`trbdf2_transient.hpp`, not
   ours). → A static over-approximation at entry: any branch-form element
   with `g_off = 0`, or with `R_on = 0` closing a loop of voltage sources,
   capacitors or other `R_on = 0` elements, refuses with
   `dependent_states`. A circuit it refuses might never visit the
   dependent mode; none it accepts can. → Reverse: run
   `BranchStructureChecker` per mode inside the TR-BDF2 loop.

6. **One structural pass per mode, memoised.** → `BranchStructureChecker`
   keeps a map mask → result and a last-mask fast path; it is inactive
   (zero cost, zero checks) without an ideal branch element, so every
   conductance-form circuit skips it; `counters['mode_structure_checks']`
   counts the passes (4 for two ideal switches visiting their 4 masks,
   however often). → Reverse: drop the memo (the pass is O(branches)).

7. **Rank-one row update vs one factor per mask (task 3.2), by
   measurement.** Single-flip walk of 4000 masks on a 12-bit
   three-phase inverter in branch form (n = 24), best of 5, A/B/A/B
   (`[rank1_measure]` in `core/tests/layer4_v2/test_branch_form_switch.cpp`):
   the engines' per-mask factor cache 206/206 ns per solve (a hit after
   the first visit of each mask); `solve_rank1` 2113/2113 ns, and WRONG
   for the branch form: its path refactor keeps the previous pivot order
   and a flip swaps `(s, a)` between 1 and −g_off, so it lands on a
   g_off-sized pivot (`|x − x_ref| = 1.3e4` against 1.3e-8 allowed on the
   16-mask bridge; 5859 of 20000 path attempts fell back). → The per-mask
   cache stays the engines' path (it already holds one factor per visited
   mask on the one shared analysis); `solve_rank1` refactors in full when
   the circuit has a branch-form element (2195/2259 ns, equal to the
   per-mask factor within 1e-12). The conductance form keeps the path
   refactor. Dense partial-pivot growth max|U|/max|A| over the 16 masks:
   1. Same numbers with Eigen 3.4.0 (206/206 vs 2157/2209 ns, growth 1).
   → Reverse: a path refactor that re-pivots inside the path.

8. **Residual of task 3.3 on an all-OFF bridge.** With every switch open
   the bridge's inner nodes are tied by `g_off = 1e-9` alone; a 1–2 A
   injection into every row gives `‖Jx − b‖/‖b‖ = 8e-8` by conditioning,
   not by the factor. → The test checks the circuit's own right-hand side
   (`‖Jx − b‖/‖b‖ ≤ 1e-12`, measured ≤ 1.1e-15) and the normwise backward
   error under the injection (≤ 1e-15, measured ≤ 1.7e-18). Task 3.4's
   bridge rectifier is compared on its bus `v(p) − v(n)`: with every
   diode OFF its common mode is set through `Rg = 1 MΩ` by currents that
   cancel to the solver residual (7.6e-12 A in both forms; the gap scales
   with `Rg`: 7.6e-6 V at 1 MΩ, 7.8e-9 V at 1 kΩ), differential mode
   1.3e-11 V. → Reverse: none needed; the absolute node is ill-posed.

9. **Dependent modes on `pwl` need the restart rule.** The projection of
   D4 is realised by the SDIRK x3 restart, which the fixed engine runs
   only with `discontinuity_restart=True`. → With the rule a dependent
   mode runs and is recorded; without it the mode is refused
   (`dependent_states`, the message says to pass the flag), since the
   trapezoid rings on it (+-396.8 V, task 1.1). → Forcing the rule on for
   such runs was rejected: it changes every other event's integration
   too. → Reverse: arm the restart window on dependent-mode entry alone.

10. **The record is the explicit projection, `d` read after the restart.**
    One scalar constraint per dependent structure, `M` diagonal (`L` or
    `C` per state; coupled inductors are not in the structure). `d` is
    `K x(t+h)` (the restart leaves the constraint exact), so
    `lambda = (K x- - d) / sum(s_j^2 / M_j)` and `dW = lambda (K x- - d)/2`;
    `projection_gap` = max |x- - s_j lambda / M_j - x(t+h)| (0 on the
    cut-set; the step's own dynamics elsewhere). → Records are made after
    the step's commit, with x- the state at the step start: with the
    event split the split instant's state would be the right x- (noted as
    a shortcut in the code). → A mode entered only by diode changes is a
    consistent commutation and makes no record; neither does `K x- = d`
    exactly. → Reverse: compute `d` from the sources at t+.

11. **`on_impulse='error'` raises `UnsupportedError(feature='impulse')`.**
    It must pass through the step loop's `runtime_error` catches (dt
    halving would retry it); the typed refusal does. `'warn'` is one
    `UserWarning` per run from Python.

12. **An ON ideal diode that closes an ideal voltage loop is released.**
    In the ideal DCM buck the switch turns on while the freewheel diode is
    still ON, so the mode (source, closed switch, ON diode, all `R_on = 0`)
    is an ideal voltage loop and the run was refused at the first CCM
    turn-on. → The structural pass checks switches first (a loop of
    gated switches and sources is still refused) and returns the diodes
    of a remaining loop; the step loop turns them OFF and re-forms the
    mask (`counters['ideal_loop_releases']`). A second release of the same
    diode in one step is refused as `ideal_voltage_loop` (no cycling). →
    Evidence: the DCM buck then matches the prototype within 1.09× at
    N = 50 and 1.002× at N = 400 (task 6.2). → Reverse: drop the
    `loop_diodes` branch in `check_mode`; the refusal returns.

13. **Group-5 review fixes (fresh-context review, 2026-11-13).**
    - H1: a release that lands back on a mask already solved this step is
      the loop forward-biasing the diode → `ideal_voltage_loop` (was a
      silent breach with the diode stuck OFF).
    - H3: SaturableInductor, NonlinearCapacitor and PmsmMna with an ideal
      element → `dependent_states` naming the nonlinear device (the pass
      and the record model linear L and C only). Reverse: give them roles.
    - H4: `engine='auto'` routes a branch-form circuit trbdf2 refuses to
      `pwl` (or asks for `dt`) instead of raising.
    - M1: the record skips |r| <= 1e-12·Σ|x| (round-off at start/resume).
    - M2: ideal elements on a capacitor loop's path count as its elements.
    - M3: a step with a gated change ANYWHERE records (a gate can force a
      diode into the mode); shortcut noted in the code.
    - M4: the transformer primary is a connectivity tie in the pass.
    - M5: `device_loss_summary` warns for an ideal element it leaves out.
    - M6: the `dependent_states` refusal (and the record) are decided on
      the committed step and only when the committed state satisfies the
      cut-set (Σ s_j i_j = 0); the "flip at the step's end" rule keeps a
      solve whose mask is the cut-set while the diode already conducts.
    - L2: the DC structural check runs on the converged DC mode; the
      start mask is checked only when the DC solve fails.
    - Not changed: H2 (refuted: a resistive-only circuit is checked, the
      branch currents make the path dynamic); L1 (a duplicate cut-set needs
      two islands joined only by one inductor, which the floating pass
      refuses first); L3 (the extra `switch_fn` call exists only on
      opt-in split runs with an opaque `switch_fn`); L4 (commutation
      events of ideal diodes, C++ `run_trbdf2` callers, `res.i` hold —
      listed as remaining work).

## TR-BDF2 (feat/core-p2-trbdf2, task 5.5 and the trbdf2 half of group 4)

14. **One record, two engines.** → The detection of an entered dependent
    structure and the record (explicit projection, `on_impulse`) moved from
    `run_transient.hpp` into `pwl/reinit.hpp` (`in_dependent_mode`,
    `ReinitTracker::entering / record / note`), arithmetic unchanged; the
    fixed engine calls it from the same place. → Evidence: the golden probe
    and a byte compare of pwl's cut-set, capacitor-loop and ideal DCM buck
    runs (times, states, records) are identical before and after. On
    `trbdf2` the record is made after each committed step from the step's
    start state, as on `pwl`; `test_the_record_means_the_same_on_both_engines`
    pins Λ and ΔW of the same circuit to 1e-12. → Reverse: inline the
    tracker back into both engines.

15. **Which TR-BDF2 steps re-initialise.** Design D7: SDIRK x3 after an event
    with dependent states. → Every step whose mode has a dependent structure
    the last committed mode had not is an SDIRK(2,2) step, and so are the
    next two, whatever the circuit (the post-event SDIRK of
    `update-trbdf2-post-event-restart` stays Newton-only). Gated entries (and
    the start) are recorded when impulsive; diode-only entries (an ideal
    diode stopping at i = 0) are consistent and only take the steps. The
    gate-edge probe (a trapezoidal step of femtoseconds) is not committed
    when the post-edge mode is such an entry (it read 1e13 V on a 10 A
    cut-set). → Evidence: with gated entries only, the ideal DCM buck's TR
    stage after each diode turn-off read the pre-event derivative slot of the
    now-algebraic inductor current: 5 steps below 1e-12 s per period and 910
    rejects in 100 periods; with diode-only entries too: none, 210 rejects,
    orbit error 9.3e-8 V. A circuit with a Coss, Lauritzen, IGBT-tail, PMSM
    or saturable-inductor history that enters such a mode is refused
    (`dependent_states`): the SDIRK step has no stage law for them. →
    Reverse: `entering(..., gated_only=true)` in the step loop.

16. **The error estimate of the re-initialisation step.** The states of the
    entered structure jump by design (flux / charge conserved by the BE
    stages at any h), so their k1 − k2 is the jump over γ_s·h, not an error;
    left in, it does not shrink with h and rejects to h_min. → Their entries
    are zero in that one step's estimate; other states keep theirs. Task
    7.2 is measured away from the committed gate-edge probes (h_probe_floor,
    1e-14 s in both forms): 1.56e-10 s branch against 1.11e-14 s conductance.
    → Reverse: drop the exclusion loop in the SDIRK estimate.

17. **No crossing inside the re-initialisation step.** A landing there would
    be a trapezoidal sub-step across the impulse. → Crossing detection is
    skipped on that step (shortcut noted in the code); a diode left on its
    forbidden side is re-decided at the next step's start by the held-diode
    path. The `engine='auto'` route (review H4) is reversed by design D7: an
    ideal-switch circuit without `dt` now runs on `trbdf2`. → Reverse: land
    with an SDIRK sub-step.

18. **What the step-1 review changed.** The re-initialisation step's
    estimate skips the dependent states, so with every state dependent its
    proposal is 5x. → It does not set the post-edge h memory (it ratcheted
    the restart 5x per period); its growth is not capped (capping it left
    step 2 at 5e-12 s on the capacitor loop, and the switch current, a
    C*dv/h round-off, rose from 4.5e-10 to 1.14e-9 A). A resumed run
    continues its committed mode (no re-initialisation at the seam); the
    probe-commit gate counts diode-only entries; an ON branch diode's held
    tolerance scales with its current. → Reverse: each is one condition in
    trbdf2_transient.hpp marked "review M1/M3/L1/L5".

19. **Bridge on a 1 MOhm-tied floating bus: trbdf2 was right, the metric
    was wrong (step 2; corrected against the exact reference).** The exact
    30-40 ms bus average is 311.343881428885841697 V
    (benchmarks/tools/bridge_ground_truth.py, mpmath; unique mode sequence
    of 11 events). trbdf2 follows that sequence (D2's turn-on lands at
    13.234077 ms, exact 13.234043 ms). The "311.154 V, -0.19 V" read here
    before came from averaging the samples t >= 30 ms over 10 ms without
    clipping the window: trbdf2's first sample is at 30.006 ms, so 6 us of
    ~311 V were dropped (311.154 V branch, 311.155 or 311.341 V conductance
    by where that sample fell). Clipped: 311.341410 V, -2.47e-3 V in both
    forms, the integration error at the default ceiling (span/1000 =
    4e-5 s), O(h^2): -1.8e-4 V with a 1e-5 s ceiling, -7.3e-6 V with 2e-6 s,
    -1.91e-4 V at rtol 1e-7 (within rtol*|v| = 6.2e-3 V per step at
    rtol 2e-5). Over 20 runs with 1e-15 perturbations the average moves by
    <= 9e-6 V (rtol 2e-5) and <= 3e-6 V (rtol 1e-7) in both forms, while
    the diode-event count varies (6-13): leakage-scale decisions at the
    zero-current turn-offs (D3 at 1e-12 A), with no effect on the answer.
    The island KCL correction tried before (eps * 2C/h * v / G_tie on the
    common mode) is not needed and was not shipped. → Pinned:
    test_bridge_ground_truth.py (within 1e-3 V at rtol 1e-7, 3e-3 V at
    2e-5, rounding spread <= 1e-4 V). → Reverse: none (no engine change).

20. **trbdf2's periodic-source step ceiling stays at T/20 (step-ceiling
    study).** Question: would T/40 or T/80 buy accuracy? Evidence, the
    floating-bus bridge against its exact average (40 ms, both forms;
    error / accepted steps). The T/20, T/40, T/80 rows force `dt` = h_max;
    on this 40 ms run the default is min(span/1000, T/20) = 4e-5 s, the
    last row:

    | ceiling | rtol 1e-5 | rtol 2e-5 |
    |---|---|---|
    | T/20 (1 ms) | -3.08e-3 V / 632 | -4.59e-3 V / 512 |
    | T/40 | -2.98e-3 V / 631 | -4.72e-3 V / 522 |
    | T/80 | -3.75e-3 V / 687 | -5.95e-3 V / 581 |
    | span/1000 (4e-5 s) | -2.20e-3 V / 1306 | -2.47e-3 V / 1230 |

    The tolerance, not the ceiling, sets the error between T/20 and T/80
    (no monotone gain, +9 % steps at T/80). → Option kept: T/20, no
    breaking default change (no CHANGELOG entry, no re-pins). → Reverse:
    the cap in run_transient_trbdf2's periodic-source ceiling.

21. **Step-3 leftovers (order check, round-off floor, DCM xfail, ceiling).**
    (a) The C++ order check of the SDIRK(2,2) restart (half-wave RL
    rectifier, an SDIRK step after every event, controller off at rtol 1 so
    every step is the ceiling): error ratio per halving 4.09 (4.07, 4.04,
    4.0 down to 25 us), second order; plain TR-BDF2 on the same grid 2.28.
    A first attempt read 1.95 because tail_mean starts at the first sample
    past the window (an O(h) metric error); the check interpolates the
    window start. (b) spike_ok accepts a post-landing signal within the
    predicate noise: boost CCM pss gap 5.2e-9 -> 2.8e-9, everything else
    unchanged. (c) test_dcm_no_step_below_1e_11 stays strict-xfail: 2,606
    sub-1e-11 steps, 400 at each gate edge (probe commits) and 1,806 in
    the stiff L*g_off mode after a turn-off; fixing them needs an
    estimate that ignores the tau ~ 2e-13 s mode, not a landing change.
    (d) hard_switched_rc trbdf2 ceiling 4.45e-6 -> 2.5e-6 (measured
    2.2236e-6). (e) Not done: x_sig is still the pre-cascade solve after a
    second cascade at the same t. → Reverse: revert the step-3 commit.

## pwl losses and outputs of ideal elements (feat/core-a-ideal-switch)

22. **Losses, currents and commutations of ideal elements (tasks 6.4 /
    6.5, 2026-11-18).** Question: how are an R_on = 0 element's current,
    loss and turn-off reported, with no conductance law to apply? →
    Options taken: the current is the solved unknown i_s everywhere
    (`res.i` already read it; `_diode_state.DiodeStateTrace` now carries
    the column, so the state traces and every caller see a finite
    current instead of `inf·0`); `device_loss_summary` reports the
    element (`form='branch'`) instead of leaving it out (M5), integrating
    the power as the line between each step's end powers v·i (exact
    while ON: P = V_f·i is linear in i; the g_off·v² leak while OFF):
    the product of the v-line and the i-line across a commutation step
    invented 0.038 W on a 24 V, 0.5 A ideal buck, against 3.5e-10 W
    (the leak) now. Tabulated `E_on`/`E_off` go through the same
    `_switch_switching_loss` as the conductance switch. Ideal diodes sit
    exactly at V_on while ON, so the voltage-crossing commutation
    record never saw them turn off: their commutations are read from the
    stamped state at each step (time = the step end unless the event
    split located it). → Evidence (`python/tests/test_ideal_losses.py`):
    forced 10 A with V_f = 0.7 gives v_D = 0.7 V and P_avg = 7.0 W
    (window from 10 µs; the whole run reads 6.965 W, its first step
    ramps 0 → 10 A); `E_on = 10 µJ·(i/1 A)·(v/24 V)` books 5.0 µJ per
    turn-on within 4.8e-11 relative — not the 1e-12 of the task: the
    OFF leak g_off·24 V = 2.4e-11 A rides on the 0.5 A; 10 turn-ons and
    9 turn-offs of the ideal freewheeling diode recorded in 10 periods.
    Not done: the impulse energy category (6.4's last clause).
    (Numbered 14 on the ideal-switch branch; renumbered 22 at
    integration batch 11 behind the TR-BDF2 entries 14–21.)
    → Reverse: the `_ideal_entry` call sites in `losses.py`, `i_branch`
    in `_diode_state.py`, `has_ideal_diode` in `run_transient.hpp`.
    Review fixes (fresh-context review, 2026-11-18): `conducting` read
    `g_on·(v − V_f)` = `inf·0` = NaN for an ideal diode, so it was False
    on every ON step — it now reads `i_branch`; the static (dt = 0) path
    recorded no ideal-diode commutation (a sine source into an ideal
    diode and a resistor: 0 events, now 7, each within one dt after the
    zero); a result with no recorded mask, or a `conduction_specs` /
    `diode_specs` entry for an ideal element, now warns instead of
    dropping the duty, switching energy or spec silently.

23. **Group-7 measurement gate: the flip is allowed on pwl (2026-11-18).**
    (Numbered 15 on the ideal-switch branch; 23 at integration batch 11.)
    Question: does the branch form cost too much to become the default?
    → Measured (`probes/measurement_gate.py` → `measurement_gate.out`,
    median of 7 interleaved runs on a shared Mac): the 12-switch
    three-phase inverter at 1922 ns/step branch against 1791
    conductance (1.073×, 24 vs 12 unknowns, the same 251 event rounds
    and 492 segment searches); the buck at 680 vs 668 ns/step (1.018×).
    7.3's bound is 1.5×: the flip is allowed on pwl. Not done: the flip
    itself (out of this slice's scope). 7.2 was measured once the
    TR-BDF2 track landed the branch form there (#16: 1.56e-10 s branch
    against 1.11e-14 s conductance, so the branch form's smallest step
    is not the smaller one).
    → Reverse: none needed (measurement only).

## TR-BDF2 track 6 (feat/core-p2-trbdf2)

24. **The DCM grind: one filtered SDIRK step after a turn-off (linear
    circuits).** Question: 1,806 accepted steps below 1e-11 s on the DCM
    buck (400 periods) followed each diode turn-off. Cause, from the trace:
    the step after the turn-off was retried at the pre-event h (2.8e-8 s)
    and rejected down to ~8e-13 s, then grew back in four or five steps;
    the raw TR-BDF2 estimate read the L*g_off mode (tau ~ 2e-13 s, the
    residual 2.4e-8 A meeting g_off = 1e-9 S) as error at any h. Why the
    existing SDIRK restart did not help: on a piecewise-linear circuit it
    is off (item 9 of update-trbdf2-post-event-restart); forced on, it
    damps the mode in one step, but its embedded estimate h(1-gs)(k1-k2)
    reads the damped jump itself (k1 = jump/(gs h): constant in h) and
    rejects it the same way (DCM 7,608 -> 7,683 steps). Options measured
    (probes/trbdf2_baseline.py, accepted / rejects on the DCM buck):
    filter alone on every step 7,270 / 15,088; SDIRK + filter on every
    step and SDIRK after every diode event 3,301 / 862, but 18 tests moved
    (recovery energies -12 %, flyback 790 forced accepts, a fuzzed bridge
    run 0.095 V off); SDIRK + filter on the SDIRK step only 4,212 / 2,753.
    → Taken: after a turn-off of a diode with g_off > 0 on a linear circuit
    without stateful devices, one SDIRK(2,2) step whose estimate is
    filtered (W^-1 est on the stage factor), C++ handle
    `sdirk_after_turn_off` (default true). It is taken when ANY g_off
    diode of the event's cascade turned off (a bridge commutation, one on
    and one off, takes it); a cascade of turn-ons only, Newton circuits
    and ideal branch-form diodes are unchanged. Two consequences, fixed:
    (a) the post-event record (item 13 there) was the femtosecond probe,
    which has not moved the L*g_off mode: v_D = -0.52 V where the circuit
    is at -34.7 V a picosecond later, and the trace ramped across the
    0.87 us restart step (DCM boost recovery energy 9.9e-4 -> 2.4e-4 J);
    after such a turn-off the record is a backward-Euler solve over
    dt = max(h_min, 1e-4 h) from the committed state (L-stable: it lands
    on the post-jump state). The jump must be instant on that scale: if
    BE over dt and over 2 dt disagree on any capacitor voltage or inductor
    current beyond atol + rtol |x|, the mode is a real transient
    (half-wave RL: tau ~ 1e-9 s, 31 mA), and the step is the plain
    TR-BDF2 restart of before with no record (skipping that transient
    moved its reference cell 1.58e-5 -> 1.67e-4). (b) `device_loss_summary` flagged a turn-off
    unresolved from the step that carries it; a short record sample then
    hid a long step after it, so the flag now reads the step that holds
    t_off + t_rr, unless that step is flat (the plateau was already
    reached). A turn-off SDIRK step that is rejected falls back to
    TR-BDF2 steps (the flyback's coupled leakage at 1e-14 s: 100 forced
    accepts with the SDIRK step kept, 0 with the fallback; HEAD 0). The
    committed gate-edge probes stay (item 7 there): with `commit=false`
    10 loss / charge / recovery tests fail. The strict xfail
    test_dcm_no_step_below_1e_11 is now a plain test of the integration
    steps: every interval below 1e-11 s starts at an event instant (the
    edge probe, the post-event record) or in the first nanosecond.
    The records themselves are not short: on the DCM buck the interval
    starting at each of the 380 diode events is 9.6e-11 to 1.2e-10 s,
    except the first event's 9.2e-13 s in the first nanosecond. A record
    or SDIRK step that is skipped (probe failed, BE solve singular, jump
    not instant, step rejected for any reason) leaves the plain TR-BDF2
    restart; a rejected turn-off SDIRK step also removes its record (the
    jump it called instant is then resolved by TR-BDF2 steps). The
    undersampled-gate count now covers every closed pulse
    narrower than its sampling, not only a new minimum (25b lets the cap
    lapse on a quiet bit).
    Measured (rtol 1e-5 unless noted; error = max |iL|, |vC| at the end
    against expm):

    | run | before | after |
    |---|---|---|
    | DCM buck accepted / rejects / err | 7,608 / 4,306 / 1.17e-5 | 4,212 / 2,753 / 1.28e-5 |
    | DCM buck rtol 2e-5, 5e-6, 1e-6, 1e-7 err | 1.67e-5, 6.33e-6, 4.36e-7, 5.92e-7 | 1.52e-5, 6.48e-6, 4.52e-7, 5.92e-7 |
    | same, accepted steps | 6,913, 8,185, 10,857, 18,398 | 3,127, 5,292, 8,208, 18,398 |
    | DCM buck steps below 1e-11 s | 2,606 | 804 (800 edge probes, 4 in the first ns) |
    | DCM buck pss orbit v_C0 (exact 8.99624708729) | 8.9962470170 | 8.9962470170 |
    | boost accepted / rejects | 1,459 / 218 | 1,283 / 109 |
    | 325 V bridge accepted / rejects | 1,306 / 105 | 1,284 / 88 |
    | buck CCM rtol 1e-4 accepted | 1,620 | 1,617 |

    At equal rtol the DCM error moves by -9 % to +10 %; at 1e-7 the jump
    is not instant on the record scale and the run is the old one. At
    equal work it is lower (5,292 steps: 6.5e-6 against 1.67e-5 at
    6,913). Errors of every other baseline row, both pwl rows and dsed are
    identical. Rounding fuzz (20 runs, 1e-15 perturbations): DCM v_end
    spread 4.7e-8, 4,212 accepted and 800 sub-1e-11 intervals in every run.
    → Reverse: `sdirk_after_turn_off = false`.

25. **A tick at a gate edge fires before the edge probe is committed, and
    the pulse cap follows the recent pulses.** (a) `restart_after_gate_event`
    refused the commit when a tick was due within the probe, and a closed
    loop's period start is a tick: no post-edge sample at 0 of 199 rising
    edges (probes/trbdf2_tick_edge_sample.py of add-compiled-control-graph);
    V(sw) ramped across the first post-edge step (ll14 / ll15). → The due
    ticks fire at the edge, with x at t (what they read at the next step's
    start before), then the cascade decides under the post-tick gates and
    the probe is committed: 199 / 199. The same at a source breakpoint.
    (b) The step cap of an opaque switch_fn was twice the narrowest pulse
    of the run (a running minimum): cl_boost_pi's 0.28 ns start-up pulse
    capped every later step (519,783 accepted in 0.3 ms). → Per gate bit,
    the last high and low widths, counted while the bit keeps toggling;
    when one toggle direction is periodic (three toggles a constant period
    apart, 1e-6 relative), a step may run to just before its next
    predicted toggle (margin 2 w + 1e-3 period). That a step then holds no
    whole pulse assumes the gate does not get faster (every pulse bounded
    by a toggle at its predicted time); pulse skipping that resumes, or a
    rising frequency, breaks it (review: a pulse every 3T, then every T).
    So a lifted step calls switch_fn every w/2 inside it (up to 4,096
    calls, no solves): the pulse that comes where none is predicted is
    still seen and located (test: 74 of 78 such pulses without the dense
    samples, 78 of 78 with). A quiet bit keeps the
    run's narrowest pulse as its cap (a cap that lapsed missed 4 of 99
    10 ns pulses after a pause, and every 1 ns one). cl_boost_pi, 0.3 ms:
    28,049 accepted (26,947 in the first 0.1 ms, before the three periods
    prove the rises regular; 580 per 0.1 ms after, against 558 with the
    edges declared: 1,594 in all); 29 / 30 edges in both runs. The
    run-wide minimum is still reported (`gate_pulse_min`) and still drives
    the undersampling warning. → Reverse: the cap from gate_pulse_min.
    Reference cells after the merge of integration batch 11 (the
    reference is trbdf2 at rtol 1e-8): ll14 0.214 -> 0.00106, ll15
    0.00124 -> 0.00123, both still above their 1e-3 tolerance; edge
    timing 4.2e-6 -> 4.7e-11 s and 2e-7 -> 3.4e-12 s. What is left is not
    the sampling: ll14's is the default tolerance's duty error, 4.9e-6
    against the independent closed-loop reference (rtol 1e-8: 7.4e-8),
    which moves each edge 4.9e-11 s on a 24.9 V V(sw) jump; ll15's is one
    current zero 8 ns late at 1.1857e-4 s, because the inrush through the
    Shockley diode leaves I(L) at 27.394 A against 27.328 A converged
    (2.4e-3 at rtol 1e-5, 5.6e-4 at 1e-6): a global-error question of
    the Newton path, not of ticks or caps.

26. **Reviewer leftovers of step 3.** (a) The no-flip branch of the
    chatter-safe cascade reset x_sig to the committed x. When an earlier
    cascade at the same t (a gate edge) had decided the bits on its probe,
    x was solved under the bits before that cascade (the stale reference
    item 21(e)). Nothing flipped, so the state the bits were decided on
    did not change: x_sig is restored to its value before the probes. It
    equals x whenever no earlier cascade ran at t, so the runs of item
    17(b) are unchanged. (b) spike_ok accepted a post-landing signal inside
    its round-off bound whatever voltage it implied across g_off; the noise
    acceptance now also needs that voltage within the circuit's range
    max|x|. (c) The post-landing diode decision after an interpolated
    landing (no solve at dt_land) takes its noise bound on the bracket
    end's step if the dt_land factor is singular, instead of aborting.

27. **An ON diode's noise floor reads v_D, not its common mode** (batch 11
    review, finding 4). The leakage-scale floor of item 18 was
    g_off/g_on (1 + |V_th| + |V_f| + |v_a| + |v_k|): a diode floating at
    10 kV had a floor of 2e-8 V, i.e. a 2e-5 A reverse current taken for
    noise. → g_off/g_on (1 + |V_th| + |V_f| + |v_a - v_k|) plus the
    round-off of forming v_D, 64 eps (|v_a| + |v_k|): 2.8e-10 V at 10 kV
    (C++ test). Measured on the DCM buck floated at 10 kV (default and
    1e-10 tolerances) the worst turn-off current was -7.3e-9 A / -1.8e-9 A
    before the change, a few ulps of the node voltages through g_on, so no
    trace there was wrong yet; the floor was only waiting for a slower
    crossing. The common-mode-free runs are unchanged (golden, baseline).

<!-- The TR-BDF2 track and this branch both numbered #24-#27 in parallel; the
     sections name their branch, and this branch's references are #24-#34 below. -->

## Ideal switch 4 (feat/core-a-ideal-switch, 2026-11-18)

24. **Task 6.1: the LCP decides ideal diodes from their crossing signals
    and replaces the structural loop release under `'lcp'`.** Question:
    how does an ideal diode (R_on = 0) enter the port LCP, whose hybrid
    coordinates put an ON port's row at 1/R_on? → Option taken: an ideal
    port is always posed in OFF coordinates (`ModeSelector::coords_of_`:
    z = i_s, ŵ = w = V_f − v), whatever its previous mode, and the site's
    network is solved with it cut out (`PwlModePorts::ideal(j)`: the
    site's mask has it OFF). The LCP then reads exactly the two crossing
    signals of the element — the current it would carry and the gap it
    would block. A port shorted by a loop of sources and closed ideal
    elements has Z_jj = 0: OFF when the loop reverse-biases it (q_j > 0:
    DECISIONS #12's release, found by the LCP, `ideal_loop_releases`
    stays 0), no mode when it forward-biases it (refused as
    `ideal_voltage_loop`, the settle's name). A branch-form diode with
    R_on > 0 is now a port of its own R_on (was 0), kept in its mode;
    conductance ports (dsed included) are unchanged. A shorted ideal port
    whose gap is zero within tol_w (V_f = 0 across a closed ideal switch,
    every inverter leg) is held OFF before Lemke: the sign of the loop's
    round-off residual no longer decides it (a negative ulp was a ray;
    review finding). "Shorted" is |Z_jj|·iscale ≤ tol_w (scaled: an
    unscaled 1e-12·max|H| let one g_off port call a bulk-capacitor
    rectifier shorted). The default settle, the static path and the DC
    seed got a floor of the same kind
    (`DiodeEventState::update_from_state_floored`): an OFF ideal diode
    turns ON only past 1e-12·max(1, ‖x‖∞) (a positive ulp turned it ON
    and the leg was refused as `ideal_voltage_loop`); the half-bridge leg
    of `test_ideal_loop_release.py` runs under both selections, second
    order against the RL square-wave response (1.57e-5 A at T/100). Not
    shown by a test: that the leg's residual is nonzero on a given
    platform (the C++ case injects ±1e-13 into the selector; the Python
    leg no longer claims it). Held and released short ports are counted
    (`mode_diagnostics` `shorted_held` / `shorted_released`, #33). The
    judge reads the law voltage (`V_f + i_s` for an ON branch-form diode,
    so an ON ideal diode is judged on i_s; conductance entries: same
    bits). The default guard stays off on branch-form circuits (it would
    change the numbers of a run that asked for nothing); `'shadow'` and
    `'lcp'` run on them. Every R_on > 0 port, dsed included, is unchanged.
    → Evidence (`python/tests/test_ideal_loop_release.py`, pwl, restart
    + split, default and `'lcp'` the same switch states and waveforms
    within 1e-9 relative): a capacitor clamp released when its force i_s
    changes sign (sine current into C ‖ R ‖ ideal diode; 3.30e-4 →
    8.26e-5 → 2.07e-5 V at T/500, T/1000, T/2000 against the closed
    form); an inductor cut-set released when its force w changes sign
    (half-wave RL rectifier, 5.34e-6 → 1.34e-6 → 3.35e-7 A, extinction
    angle by root-finding); the CCM RL buck whose every turn-on closes
    source + switch + ON diode (6.64e-9 → 1.67e-9 → 4.17e-10 A; the
    settle releases 19 times in the measured run, the test asserts > 0;
    the LCP none). Oracles unchanged under
    `'lcp'`: ideal DCM buck −2.660e-6 / −6.591e-7 / −1.709e-7 /
    −4.354e-8 V at N = 50…400, order 1.972, no record. Bridge against
    the mpmath truth (311.343881428885841697 V, pwl + restart + split,
    dt = 1 µs, measured by a scratch probe, not a committed test):
    −3.733e-6 V for conductance and branch (R_on = 1e-3) forms, default
    and `'lcp'` alike, all four OFF after the D1/D4 tie.
    → Reverse: drop `coords_of_` (ideal ports refused again) and restore
    the `branch_form_diode` refusal in `run_transient`.

25. **Task 6.4 (last clause): the impulsive energy is a loss category
    booked to a switch.** Question: to which device is a re-initialisation
    dW booked, and where is it read? → Option taken: every `ReinitRecord`
    carries `cause` (`pwl/reinit.hpp`, shared by pwl and trbdf2): the
    gated element of the dependent structure whose bit changed on the
    step, else any gated element that changed (a gate that forced a diode
    into the mode), else the structure's first element. Python reads it in
    `pulsim.losses.impulse_losses(builder, result, t_from, t_to)` and
    `device_loss_summary` adds `E_impulse_total`, `n_impulses`,
    `P_impulse_avg` to the entry of that switch or diode — only when the
    run has records, so every other run keeps its keys. A record counts in
    the window its step starts in (`t_from <= rec.t < t_to`). dsed records
    (dicts) without `cause` fall back to the switch whose stored bit
    changed at `rec.sample`. `thermal/device_losses.hpp` is untouched; the
    reading rule there takes `E_impulse_total` from the summary. →
    Evidence (`python/tests/test_impulse_losses.py`): the capacitor loop
    books 24 µJ to `S` within 1e-9 relative on pwl and trbdf2, and
    impulse + conduction equals the stored-energy drop within 1e-12; the
    cut-set (10 V, 1 Ω, 1 mH) closes source = resistor + switch + impulse
    to −2.57e-5 at τ/20 and −6.48e-6 at τ/40 on pwl (second order, taken
    around the pulse step: across it the sample quadrature is first
    order), −3.2e-7 on trbdf2. The pulse step itself is partitioned
    exactly (#32): dW at its start, conduction at its end power. The cause is searched
    among the structure's own elements (a gate first, then a diode),
    never a switch toggling elsewhere on the step (review finding).
    → Reverse: drop the field and the keys.

26. **Shared island correction (`events/islands.hpp`).** Question: pwl and
    dsed each carried a floating-island KCL correction; which is shared?
    → Option taken: `find_islands(graph, pool, mask, weak_g)` and
    `IslandCorrection` (joint LU when there are 2+ islands and the
    estimate is regular, else sequential; σ = g·s − Σ_S f, optional
    `base` term in dsed's expression order) live in one header, written
    for dsed to share (dsed still runs its own `port_islands_of_`); pwl's
    `PwlModePorts` uses it (its numbers bit-identical: survey and
    `[pwl_ports]`). dsed (`dsed/*`) is not edited here: its switch to the
    header is the dsed owner's (weak_g 1e-6 there, 1e-3 on pwl, as today).

27. **Engine-neutral site driver for trbdf2 (complementarity 4.2).**
    `pwl::select_site(selector, ports, cache, diodes, mask_of, prev_on,
    x0, dt, h_site, b_extra, Z, u, xj, coords)` in `pwl/mode_ports.hpp`
    is the whole decision of one site: run_transient now calls it, and
    trbdf2's settle calls it with its own stage's (mask, dt, b_extra).
    `trbdf2_transient.hpp` is not edited here (its owner's branch has
    commits on it after this branch's base). The shared tests skip trbdf2
    while it refuses `mode_selection` by name and run once the hook lands
    (`test_lcp_agrees_with_the_default_settle_on_every_engine`); the cost
    gate (`probes/measurement_gate.py`) runs over pwl and trbdf2.

28. **Nits.** (a) A branch-form diode flip forced by a gate edge at the
    step start is decided in the restart's first stage; it was stamped at
    the step end (one dt late). It is now stamped at t_prev when the
    step's gate bits changed at its start and the flip is already in the
    first stage's bits of the attempt (reset per step and per halving retry)
    (test: the RL buck's diode turn-ons at the gate-off edges within
    1e-3·dt, with and without the split). Conductance diodes keep their
    interpolated stamp (defaults unchanged). (b) The default guard's own
    mask cycles and settle rounds now reach `mode_guard.lcp.mask_cycles`
    / `settle_rounds_total` (were always 0; no test reaches a guard mask
    cycle without a hook, so this is not tested).

29. **Default-flip evidence (not the flip).** `benchmarks/tools/
    ideal_switch_flip_evidence.py` writes `benchmarks/IDEAL_SWITCH_FLIP_
    EVIDENCE.md`: 109 reference circuits × 5 configurations × {default,
    branch, lcp, branch+lcp}, graded the same way in both directions
    (outcome rank, else err_rms beyond 1 %). Branch form here is finite
    R_on = 1/g_on (no R_on = 0 port: the ideal cases are the unit tests):
    same outcome on all 109 on
    pwl / pwl+restart / pwl+split; trbdf2 2 better, 106 same, 1 new error
    (`ll13_active_clamp_forward`, Newton fails at t = 0 with h at h_min:
    trbdf2-side); dsed refuses the branch form by name (54 newly refused,
    task 8). `'lcp'` (36 circuits with a PWL diode): same outcome on the
    34 it runs on pwl; 2 refused by design (a Newton device: the port model
    is linear); trbdf2 refused until hook 4.2; dsed 5 errors
    (`DiodeModeInconsistentError` on transformer / coupled-inductor
    circuits — dsed builds its ports with R_on = 1/g_on > 0, which this
    change does not touch, so dsed-side). Cost: branch / conductance
    median-of-7 ns/step 0.85 (inverter) / 1.46 (buck, loaded machine;
    1.02 earlier) on pwl, 1.006 / 1.027 on trbdf2; `'lcp'` costs 4.4–5.1×
    the default in median wall time on pwl (`branch+lcp` 5.0–6.9×; one
    run per cell, cells where both ran): two site solves per step, the
    cost item before `'lcp'` can be a default.

## Ideal switch 5 (feat/core-a-ideal-switch, 2026-11-18)

30. **`'lcp'` poses the LCP only where its answer can differ from a known
    one.** Question: `'lcp'` cost 4.4–5.6× the default on pwl (one LCP and
    two solves per site); which sites need Lemke? → Option taken
    (`run_transient` `mode_lcp_site`, `PwlModePorts`): (a) the start mode
    is solved first (the step's own factor, no second solve); when every
    diode's crossing signal on that solve (ON: its current; OFF: the gap to
    V_on), after the floating islands' KCL, is clear of zero by 1e-8 of the
    circuit scale (10× the selector's tie tolerance, with the selector's own
    σ·C_max/h term), the start mode is a strictly complementary solution of
    a passive LCP and the step is that solve (`quiet_sites`; Lemke returns
    it with no pivot, any circuit; with an ideal diode ON, posed open so
    Lemke pivots, only on a passive circuit). (b) Else the
    LCP's last answer from the same start mask (gates + diode bits; at most
    256 kept) is solved and kept when it is clear of the band the same way
    (`cached_sites`; passive circuits only — a unique solution — and not
    with ideal ports in the circuit, nor with a
    hysteresis port, V_th > V_f, whose turn-on reads V_th on the START
    solve: the cache broke `test_lcp_banded_dcm_buck_equals_the_default`
    by 1.1e-2 before this exclusion). (c) Else Lemke
    runs on the start solve's port voltages (u is not re-solved), and when
    it keeps the start mode the start solve is the step. Port matrices and
    island corrections are cached per (mask, dt) (`port_matrices`), the
    branch part of the circuit scale once per run. `pwl::select_site` got an
    additive trailing `x_site` argument (signature otherwise unchanged).
    → Evidence: a temporary check that ran the full LCP at every (a)/(b)
    site and threw on a different answer found 0 mismatches over
    3,348,409 sites (3,115,780 quiet, 102,760 cached; 34 circuits × pwl /
    restart / split × conductance / branch R_on = 1/g_on / ideal R_on = 0)
    and the full pytest (4587 passed with the check on). `'lcp'` outputs
    are byte-identical to the pre-change `'lcp'` on all 102 cells (times,
    states, switch states). Cost (`probes/lcp_cost_gate.py`, median of 7
    interleaved runs, wall of `simulate()` only): median ratio to the
    default 5.64 / 4.67 / 4.93 → 1.058 / 1.087 / 1.088 on pwl / restart /
    split (load average 12–15: a shared machine). Above 1.2: cells whose
    diodes sit inside the selector's own tie band for long stretches
    (npc_three_level 1519 of 3000 sites quiet: an ON diode carrying ~1e-7
    of the current scale, below σ·C_max/h; boost_pfc 3176 LCP sites), where
    the LCP is the decision, plus noise of the loaded machine on sub-ms
    runs. → Reverse: drop (a)/(b) (Lemke at every site).

31. **Smallest branch-form R_on.** Question: `R_on == 0` is compared
    exactly; what does 1e-300 do? → Option taken: the builder refuses
    0 < R_on < 1e-9 Ω by name (`CircuitBuilder::kBranchRonMin`,
    `ValueError` naming the device and field): below it the port's ON row
    (1/R_on) is past the round-off resolution of the network. R_on = 0 is
    the ideal switch, 1e-9 Ω and up a resistor. → Evidence
    (`test_ideal_loop_release.py`): 1e-300 and 1e-12 refused; 1e-6 runs as
    a resistor to 1e-12 relative. → Reverse: drop the bound.

32. **The pulse step is partitioned exactly.** Question: a
    re-initialisation's dW and the conduction loss both touched the pulse
    step (sample−1, sample]; conduction integrated the step's linear
    interpolation from the pre-jump value, an O(h) excess. → Option taken:
    `step_start_states` (`_result_views.py`) takes the right limit at the
    start of a pulse step: the row before the record's sample is replaced
    by the sample's own state and mask, so the step conducts its end power
    only and dW is the jump. dsed records (dicts) are untouched; a record
    whose stored interval is not the pulse step alone (store_every > 1:
    the interval's earlier steps conducted from the pre-jump state) moves
    nothing. Limit: under the diode split the pulse step's first sub-step
    ran in the old mode, so "end power over the whole step" is exact only
    without the split (review finding). →
    Evidence (`test_impulse_losses.py`): on the capacitor loop at
    h = 1e-8 … 1e-5, E_impulse + E_total(S) equals the stored-energy drop
    to 1e-12 relative; on the cut-set, R's E_total over the pulse step is
    its end power × h to 1e-12. → Reverse: drop `_reinit_step_starts`.

33. **Reviewer leftovers of slice 4.** (a) Every pwl `ideal_voltage_loop`
    message names the loop element by element (trbdf2's is the trbdf2
    owner's) in loop order
    (`ModeStructure::loop_texts`, "D3 -> D2 -> D1 -> V1"); under `'lcp'` a
    loop through several ideal ports (none shorted alone: the LCP ends on
    a secondary ray; only on that status) is found by closing every ideal port the posed network
    does not reverse-bias, and refused by the same name (was
    `consistent_mode`). (b) `shorted_held` / `shorted_released` count the
    short ports held OFF before Lemke and released by it. (c) The leg test
    asserts the hold is reached (`shorted_held > 0`, none released) and no
    longer claims a platform residual. (d) Bridge truth on pwl
    (`test_bridge_ground_truth.py`): restart + split, dt = 1 µs, both forms
    × default / `'lcp'`: −3.733e-6 V, and the 11 exact commutations with
    their change sets, each within 1.5 dt (the D1/D4 turn-on tie of event 6
    may land on two adjacent steps under the default, one breach logged).


34. **Ideal-twin flip evidence.** Question: what does the R_on = 0,
    V_f = 0 twin of every reference circuit do, graded against a reference
    regenerated for that circuit? → Option taken:
    `ideal_switch_flip_evidence.py` variants `ideal` / `ideal+lcp`
    (`ideal_form()`: every switch and diode branch form, R_on = 0, V_f = 0,
    the netlist's g_off and V_th kept), graded against
    `differential_runner.reference` run under the same substitution
    (`not comparable` with the reason when it has none), refusals counted
    by `UnsupportedError.feature`; section "The ideal switch" of
    `benchmarks/IDEAL_SWITCH_FLIP_EVIDENCE.md`. → Evidence (109 circuits;
    better / same / worse / refused / error / not comparable): pwl
    9/70/16/8/2/4, restart 9/70/22/2/2/4, split 3/79/19/2/2/4, trbdf2
    9/84/8/3/4/1; dsed refuses the branch form (task 8). Structural
    refusals by name: `ideal_voltage_loop` 2 per pwl configuration and 3 on
    trbdf2 (three-phase diode rectifier and boost PFC: two ideal diodes of
    a stiff-source bridge both ON at a commutation close source-to-source;
    full-bridge DC on trbdf2), `dependent_states` 6 on plain pwl only (a
    capacitor through an ideal diode at t = 0: plain pwl has no
    re-initialisation; restart runs them), `floating_node` and `impulse` 0.
    Errors: no DC operating point for the ideal twins of
    `ll15_boost_closed_loop` and `ll10_full_bridge_inverter_lc` (a source
    across an inductor and closed ideal switches at DC: the DC solve
    returns 1e13 A and refuses with a RuntimeError, not by feature name).
    `worse` is mostly err_rms beyond 1 % against two different references
    with the same outcome; outcome drops: `boost_switching_complex` on
    restart / split (agree → disagree, 0.021) and `ll07_boost_vcswitch`
    on trbdf2 (0.0068) — not traced here. No cell was traced to a
    branch-form defect, so nothing was fixed under this item. Cost: median
    wall ratio ideal / default 1.05–1.11 on pwl, 1.05 on trbdf2;
    ideal+lcp 1.40–1.66 (loaded machine, one run per cell).
    Migration audit for 9.1–9.3: `MIGRATION_AUDIT.md` (no default flipped).
