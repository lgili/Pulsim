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

14. **Losses, currents and commutations of ideal elements (tasks 6.4 /
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

15. **Group-7 measurement gate: the flip is allowed on pwl (2026-11-18).**
    Question: does the branch form cost too much to become the default?
    → Measured (`probes/measurement_gate.py` → `measurement_gate.out`,
    median of 7 interleaved runs on a shared Mac): the 12-switch
    three-phase inverter at 1922 ns/step branch against 1791
    conductance (1.073×, 24 vs 12 unknowns, the same 251 event rounds
    and 492 segment searches); the buck at 680 vs 668 ns/step (1.018×).
    7.3's bound is 1.5×: the flip is allowed on pwl. Not done: the flip
    itself (out of this slice's scope), and 7.2 — trbdf2 refuses the
    branch form by name, so its smallest committed step cannot be
    measured; the flip stays held on trbdf2 until it runs there.
    → Reverse: none needed (measurement only).
