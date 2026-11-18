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

19. **Bridge on a 1 MOhm-tied floating bus: island rounding, not a resonance
    (step 2 verdict; fix not shipped).** The bus nodes have no capacitance
    to ground, so there is no L*g_off resonance: the line inductor and
    g_off are first order (tau = L*g_off = 1 ps). The common mode is set by
    uA of leakage, while the right-hand side carries 2C/h * v on both bus
    nodes; their rounding, eps * (2C/h) * v / G_tie, is 6e-5 V at h = 1 us
    and volts at a probe. Measured (20 runs, 1e-15 perturbations, rtol 2e-5,
    40 ms): branch form 8-12 diode events, 3-6 chatter breaks, bus average
    311.154723-311.154765 V; after the first turn-off the common mode reads
    -0.94 V in some runs where the island's KCL by hand gives -0.0305 V.
    Tried: imposing the island's KCL sum exactly after each linear solve
    (DSED #30's rule; islands per mask, ties <= 10 uS and inductors as
    boundaries; .scratch patch). Branch form: events + chatter = 13 in
    every run but split 8/9, bus average spread 1.5e-7 V; the turn-on at
    23.58 ms identical. What remains is D3's zero-current turn-off at
    1e-12 A: the OFF probe finds the island's common mode 1e-4 V forward of
    ac1, the cascade ping-pongs until the chatter guard's fallback takes a
    5 ns step with D4 ON. Conductance form: 7 of 20 runs still end at
    311.34 V. The same patch moved the pinned bridge loss test by 7e-5
    relative and turned trbdf2 on ll10_full_bridge_inverter_lc from agree
    to disagree (0.0051). → Not shipped: it does not reach an identical
    event count and moves references it does not explain. → Upgrade path:
    land the island correction together with a judged decision at
    zero-current turn-offs (the mode selector's complementarity), and
    re-pin the bridge tests from a reference, not from the engine.
    The reference is not settled either (unperturbed, 30-40 ms bus
    average): pwl 311.343881 V at dt 1e-6 and 2.5e-7 (with or without the
    restart rule; it chatters D1/D4 every dt at 4.66 ms with the source
    8 V below the bus), dsed / exact 311.316685 V, trbdf2 311.154 V
    (branch form, every run) or 311.154 / 311.341 V (conductance form).
    The bridge is not graded until a reference is settled.
