# Decisions taken while implementing tasks 1–4

Each entry: question → option taken → why → how to reverse. The design
left no open question for the owner; these are the implementation
choices it did not pin.

1. **Coordinates of the LCP.** Previous-mode (hybrid) coordinates
   always: `z = w` for ports ON before the event, `z = i` for ports OFF
   before; no greedy basis search. → It is what D2 states, it makes the
   answer cost zero pivots when the mode does not change (tested over
   all 8 previous modes of a 3-port network), and it matches the
   prototype pivot for pivot on the 218 dumped LCPs. → Reverse: change
   `ModeSelector::hybrid_lcp`; nothing else depends on the basis.

2. **Diode law on the conductance form.** D1 with `R_on = 1/g_on` in
   series with `V_f` and the `g_off` leak `Γ` in parallel on every port
   (`i_term = i_d + g_off·v`). → The resistor-pair diode is exactly this
   element; the port LCP equals the prototype's on the same circuit with
   the resistors made explicit (1.2e-10 to 2.3e-8 relative on the
   buck/boost, 2.5e-7 on the three-phase overlap). → Reverse: the four
   numbers of `DiodePort`.

3. **Port model through the cache.** `(Z, u)` from `solve_at(mask, 2h)`
   (the trapezoidal split at `2h` is backward Euler at `h`), the BE
   history `−(C/h)·x(t⁻)` and the sources at `t + h`; `C` and the
   per-mask constant `b_c` from `assemble_segment_split`, cached. One
   solve for `u`, one per port column. → Reuses the factor the engine
   already owns; no new factorisation. → Reverse:
   `NativeCircuitBuilderAdapter::diode_port_problem_full`.

4. **Tolerance.** `tol = 1e-9` (the design's), `tol_w = tol·vscale`,
   `tol_i = tol·iscale + tol_w·C_max/h_e`, not the prototype's 1e-11.
   → At 1e-9 no tie in any family is decided differently at `tol/10`
   (task 3.3); 1e-11 sits under the rounding of the conductance form
   (1.49e-8 V gaps were seen on a 10 A ON port). → Reverse: `ModeSelector::tol`.

5. **Classification of free ports.** A port the selector solved is read
   from Lemke's own `w` (exactly complementary by construction), not
   from a recomputed `H·z + q`. → The recomputed residual carried a
   1.49e-8 V rounding gap that read a 10 A ON diode as OFF. → Reverse:
   `solve_with_hysteresis_`, one line.

6. **Default `h_e`.** `h_e = 1e-7·T_ref`, `T_ref` = the smallest of the
   run window, the shortest source period and the PWM period (from the
   switch function's edges). → Measured band (task 3.2,
   `probes/h_e_band.out`): every family returns the prototype's mode for
   `h_e/T_ref ∈ [1e-9, 1e-6]` (bridge3 C filter fails from 1e-5, the
   buck below 1e-10). 1e-7 is inside with a decade of margin both ways;
   at 1e-6 the three-phase C filter had 12 ties decided differently at
   `tol/10`, at 1e-7 none. → Reverse: `H_E_REL` in `_dsed_dispatch.py`.

7. **Boost at `t = 0` (one event).** With `R_on = 1 mΩ` on switch and
   diode, an inductor at rest and the capacitor at 0 V, the diode shares
   the switch current: the conductance circuit selects ON where the
   ideal prototype sees a tie and takes OFF. Accepted as a model
   difference, not a selector defect (pinned in
   `test_mode_selector.py::MODEL_DIFFERENCE`). → Reverse: none needed;
   it disappears with the ideal elements of task 6.5.

8. **Hysteresis (D8).** An OFF-before port with `V_th > V_f` is held OFF
   (its row removed from the LCP); after the solve, if its open-circuit
   `v ≥ V_th` it is released and the LCP re-solved once. ON ports follow
   the plain law. → The design's rule; the half-wave rectifier turns on
   at 15.9155 µs as specified. → Reverse: `solve_with_hysteresis_`.

9. **Who is right in the shadow report.** Neither decider judges
   itself: for each disagreement the kernel propagates both candidate
   modes exactly from the held state over `100·h_e` and `1e4·h_e` and
   checks the mode's own law (ON: `i ≥ 0`; OFF: `v ≤ max(V_th, V_f)`)
   at both points, within `1e-9·vscale`. → An independent reference
   that needs no analytic solution, so it applies to every circuit. It
   is a necessary condition only (a mode can be consistent over
   `1e4·h_e` and wrong later). → Reverse: drop `judge_`; the report
   keeps the modes.

10. **Shadow failures do not abort the run.** An exception inside the
    shadow (port model or LCP) is counted in `errors` with the last
    message; the settle's numbers are untouched. → Shadow must not
    change behaviour; the error is visible, not swallowed. → Reverse:
    rethrow in `shadow_compare_`.

11. **pwl / trbdf2 shadow refused, not added (task 4.2 open).**
    `mode_selection='shadow'` raises `UnsupportedError(engine,
    'mode_selection', …)` on `pwl`, `trbdf2` and `auto`; `'lcp'` on
    every engine. → Neither hook is a few lines at one call site: the
    fixed engine iterates the mask inside `run_transient.hpp`'s event
    loop and the TR-BDF2 settle sits in `trbdf2_transient.hpp`; both
    need a port problem from their own factor and history. → Reverse:
    add the hook and drop the refusal (see "What remains" in tasks.md).

12. **Task 4.3's `V_th = 1e-6` clause.** The shadow reports every
    disagreement with the settle's round count on that event and the
    run total `settle_rounds_total`, rather than listing every
    multi-round event. → A multi-round event where both agree is not a
    finding; the counter gives the cost comparison. → Reverse: record
    every event with `settle_rounds > 1`.

## Slice 2 (2026-11-13)

13. **Tie rule: fixed widening, unfix, leak pins judged (D5).** Decided
    ports stay fixed while the ties are widened; a fixed port the wider
    pass contradicts is released (counted `unfixed`); a final tie whose
    current pins a node (`i·R_jj > tol_w`) is ON unless the all-ties-OFF
    mode passes the exact judge (#9). → Measured 0 wrong modes on every
    family, prototype and engine, ideal and conductance, banded and not
    (`probes/tie_rule.out`, `probes/shadow_survey.out`); the three-phase
    C filter 30 → 0. The pin is real physics in the conductance form:
    the ~49 nA leak holds the floating DC bus. → Reverse: drop the pin
    step in `ModeSelector::select` (ties OFF, slice 1).

14. **Refusals reuse `UnsupportedError`.** No consistent mode on a
    passive circuit (secondary ray, pivot cap) raises
    `UnsupportedError(engine, 'consistent_mode', '<diodes>', 'Diode')`
    with `t`; a non-passive circuit (VCVS, negative R/C/L, found at build
    time by `CircuitBuilderAdapter::passivity()`) whose Lemke run ends on
    a ray is enumerated up to `k = 12` (`enumerations`,
    `nonunique_modes`), then `UnsupportedError(engine,
    'non_passive_ports', <device>, <kind>)`. → One exception type the
    callers already route; the feature string names the cause. →
    Reverse: a dedicated subclass in `mode_refusal.hpp`.

15. **Default-mode guard judges the projected settled state.** After
    each settle (t = 0 and every event) the kernel judges the settled
    mode with the exact judge (#9) from the projected state; an
    inconsistent mode is decided by the LCP from the held state instead
    (`diagnostics['mode_guard']`: `checks`, `repairs`, `confirmed`,
    `repair_times`, `repair_diodes`, `repair_violations`); an LCP refusal
    there is re-raised naming the violating diode. Consistent runs are
    bit-identical to the unguarded settle (24 of 24 surveyed correct
    circuits). → The settle stays the decider where it is right; only
    its wrong answers change. BREAKING for wrong numbers only. →
    Reverse: `pulsim._dsed_dispatch.MODE_GUARD = False` (A/B switch).

16. **Bridge repairs kept; double crossing is an event-location defect.**
    The guard repairs 19–20 events on the single-phase bridges and 9 on
    the three-phase rectifier example where the settle's mode fails the
    judge by a rounding-level margin; the waveform moves by ≤ 1.2e-12 V.
    → Keeping them costs nothing and the rule stays "every inconsistent
    mode is repaired"; exempting them would need a magnitude threshold.
    The V_th = 0 double-crossing tank stays wrong under the guard and
    'lcp' alike: after a turn-on at exactly v = 0 the armed turn-off
    predicate starts the next step at 0, which `locate_event_in_step_`
    skips as "just fired", so the return through zero 0.64 µs later
    inside that 1 µs step is never an event. → Not a mode decision;
    recorded for the event-location owner. → Reverse: n/a.

17. **'lcp' does not save events.** Settle / LCP event counts are equal
    on every correct circuit and LARGER where the settle was wrong (boost
    50: 969 → 978, boost 500: 1133 → 1157, single-phase bridge 78 → 80),
    because the correct mode has more commutations. → The claim in the
    proposal that the LCP removes settle rounds is about solves per
    event, not events. → Reverse: n/a (a measurement).

18. **'lcp' opened on dsed only.** `mode_selection='lcp'` runs on the dsed
    event integrator (diode circuits) and is refused by name on `pwl`,
    `trbdf2` and a diode-free dsed run (#11 still holds for the hooks).
    → The other engines' hooks sit in files this change does not own. →
    Reverse: refuse 'lcp' everywhere again in `pulsim.simulate`.

## Slice 3 (2026-11-13)

19. **Zero-start predicates: skip only the one that fired (fixes #16).**
    `locate_event_exact_` and `locate_event_in_step_` skipped every armed
    predicate with `g(t0) == 0`. Now: the predicate that fired the last
    event is skipped only on a step starting at that instant; a diode
    predicate at exactly zero is evaluated at `t0 + d` (`d = 100·h_e`,
    the judge's shortest propagation; `1e-6·h` without h_e; at most half
    a scan interval) and located from there when it moves into its
    admissible side (DiodeOn `v − V_on < 0`, DiodeOff `v − V_f > 0`).
    One that stays at zero (a diode at rest) or leaves on its event side
    (an inconsistent mode — the guard's case) is skipped; nothing fires
    at zero length, so no Zeno loop exists by construction
    (`test_diode_resting_at_exactly_zero_does_not_spin`). Non-diode
    predicates keep the old skip (no direction is known for them).
    → Measured: the V_th = 0 tank now switches at 1.2532358877,
    1.8877214715, 7.5370567364, 8.1702737953 µs (solve_ivp DOP853 rtol
    1e-13: within 3e-16 s; pwl at 0.1 ns: same edges to its grid); worst
    ON reverse current −1.83e-2 → 0 A. Every other survey circuit
    (shadow_survey + extra_circuits, 29 circuits × settle / guard /
    'lcp' = 87 runs) is bit-identical to 11ab8e95
    (`probes/survey_bit_identity.out`); `bit_identity.py` 36/36. → Reverse: restore `if (g == 0) continue;` in both locators.
    Review (fresh-context, read-only): no Zeno loop — a zero start is
    located only in `(t0 + d, t1]` and the predicate that fired at `t0`
    is skipped, so each zero start advances time by `d`. Residual risks,
    recorded in KNOWN_LIMITATIONS and not changed: `d = 1e-6·h` when
    `h_e = 0` (guard off), so a diode sliding on its threshold advances
    slowly until the chatter detector stops it; a zero start leaving
    toward its event side is skipped for the step (the decision at `t0`
    owns it; unjudged without the guard); current zero-crossing and
    custom predicates at zero are still skipped (as before);
    `last_fired_*` is not reset between two `simulate()` calls on one
    `PEDSimulator` (Python builds a new one per run).


20. **Cost gate on a 60-diode bank (task 8.3): the guard is affordable.**
    Ten three-phase bridges (60 diodes, phases 6° apart, each on its own
    200 µF ‖ 50 Ω, tied by 1 MΩ), `V_th = 0`, no line inductance,
    `t_end = 0.04 s` (`probes/bank60_cost.py`, `.out`). All three modes
    end at 173.198150009 V − 2.71e-11 V against trbdf2 rtol 1e-9.

    | mode | events | guard checks / repairs | LCP problems = port-model solves | pivots (max / event) | widenings | enumerations | masks = eigendecompositions | median wall (7 interleaved rounds) | per event |
    |---|---|---|---|---|---|---|---|---|---|
    | settle (guard off) | 435 | — | 0 | 0 | 0 | 0 | 221 | 21.2 s | 48.6 ms |
    | default (settle + guard) | 373 | 374 / 98 | 292 | 699 (30) | 188 | 0 | 204 | 21.4 s | 57.5 ms |
    | 'lcp' | 373 | — | 1102 | 1177 (30) | 728 | 0 | 126 | 21.7 s | 58.2 ms |

    Gate "no event over `4·k` pivots": max 30 against 240, passes. The
    enumeration limit (`k ≤ 12`) is never reached: the bank is passive,
    Lemke always terminates. The machine was shared; the min/max spread
    was 13–37 s per mode, so the medians differ by noise, and the
    counters decide: the guard adds 292 port-model solves and 699 pivots
    to a run whose cost is one eigendecomposition per new mask (221 /
    204 / 126). The 62 fewer events under the guard are settle detours
    the repaired mode no longer takes. The guard is affordable; nothing
    was made cheaper. Per-event time is dominated by mask extraction
    (add-exact-propagator-strategy owns the mask cache).

    Variants that do not run on dsed in any mode, identical on the HEAD
    11ab8e95 extension (not caused by #19): 0.2 mH line inductors
    chatter ("commutated 32 times") at `V_th = 0` and `1e-2`; the same
    bank paralleled onto one 2 mF ‖ 5 Ω bus does not settle at `t = 0`
    (settle, guard) or makes no progress ('lcp'); trbdf2 aborts on both
    ("non-finite step … h_min"); pwl at `dt = 1 µs` runs both (181.81 V /
    200.01 V, no reference to check against). Recorded in
    KNOWN_LIMITATIONS. → Reverse: nothing to reverse (a measurement); a
    cheaper guard would cache the judge's two propagations per mask.

21. **Passivity classes at build time (task 5.1).**
    `CircuitBuilderAdapter::passivity()` now reports three classes,
    exposed as `diagnostics['mode_guard'|'mode_selection']['passivity']
    = {'class', 'device', 'kind'}`: `non_passive` (a VCVS; a negative
    R, C or L; a group of coupled inductors — connected by couplings —
    whose inductance matrix has an eigenvalue below `−1e-12·max L_ii`,
    named `L1+L2+…`, kind `InductorCoupling`), `passive_nonreciprocal`
    (an ideal transformer, named; Lemke keeps its PSD guarantee, the
    class is reported, nothing refused), `passive_reciprocal`.
    Pairwise couplings with `0 ≤ k ≤ 1` do not make three windings PSD:
    `k12 = k13 = 1, k23 = 0` has eigenvalues `1 ± √2, 1` (Catch2
    "Passivity: coupled inductors, ideal transformer, VCVS");
    `add_multi_winding_transformer` already refuses such a matrix at
    build time, `add_inductor_coupling` calls do not, so the check
    belongs here. No gyrator device exists in the builder. → Measured on
    real circuits: a four-diode bridge behind an ideal transformer
    (n = 0.5) on dsed default: 31.876241227679 V against pwl at 0.1 µs
    31.876241231725 V, 2 guard repairs, no diode conducting in reverse
    or blocking forward voltage; a rectifier fed by a coupled-inductor
    transformer is refused before any mode decision by the extractor
    (`UnsupportedError('dsed', 'extraction')`; the inductance-matrix
    block is add-exact-propagator-strategy task 7), so the PSD class is
    exercised by the Catch2 test only. → Reverse: `passivity()` back to
    the slice-2 loop.

    **Finding (closed by #22, not a passivity matter):** `mode_selection='lcp'`
    is wrong on that bridge — 11.726610 V, D2 and D3 never conduct and
    D2 blocks +49.86 V — and identically with the transformer replaced
    by a floating sine source between the secondary nodes (n = 0.5 /
    1 / 2: 11.73 / 23.45 / 46.91 V against 31.88 / 63.75 / 127.50 V).
    The result flips with `H_E_REL` (wrong at 1e-10, 1e-9, 1e-7; right
    at 1e-8, 1e-6…1e-4). The shadow shows the LCP's open-circuit gap of
    D2 at its turn-on event as +2.3e-4 V (reverse) where the exact
    trajectory rises 0.31 V in 1e4·h_e: the port model's `q` at a
    turn-on event on a bus whose common mode is set by 1 MΩ and g_off
    leaks. The default (settle + guard) is right on all six.

22. **Port model solved for the increment (fixes the #21 finding).**
    `diode_port_problem_full` solved the backward-Euler companion in
    history form, `J·s = (C/h)·x(t⁻) − b(t + h)`. The history term puts
    `C·v_C/h_e` on both nodes of a capacitor (100 µF, 31.9 V, 2 ns:
    1.6e6 A); the common mode of an island held by weak ties is their
    sum, where that term cancels, and its rounding (~2e-10 A) over the
    island's pivot (1 µS) is ~2e-4 V. At D2's turn-on root the gap moves
    `h_e·dv/dt` = 3.1e-5 V in `h_e`, so rounding chose the sign: +2.3e-4 V
    read as OFF with zero pivots, decided, never widened. The error grows
    as `1/h_e` and its sign is arbitrary — the flips with `H_E_REL`. Now
    `J·δ = −(G·x(t⁻) + b(t + h))`, `u = v(x) + v(δ)`: the same equation,
    no `1/h` on the right-hand side; `G` per mask is cached next to
    `b_c` (the cached `C` is gone). A numpy replica of the all-OFF
    companion gives, for `H_E_REL` = 1e-10 … 1e-4, history-form gaps of
    +2.8e-1, +6.9e-3, −1.9e-3, −3.3e-5, −3.4e-4, −3.1e-3, −3.1e-2 V
    against the algebraic −3.1e-8 … −3.1e-2 V; the increment form
    matches to four digits at every one. → Measured: the bridge under
    'lcp' ends at 31.876241228 V for `H_E_REL` = 1e-10 … 1e-4, floating
    sine and ideal transformer alike, bit-identical to the default at
    1e-7 (pwl at 0.1 µs: 31.876241232 V); D2's turn-on is a tie, widened
    (and a judged leak pin at 1e-9, 1e-10), as D5 intends. Shadow
    disagreements on it 6 → 2, both guard repairs where the settle fails
    the judge. `survey_bit_identity.py`: 90 runs bit-identical, 0
    changed. `test_mode_selector.py::LEAK_PINS['bridge1_vf0']` 3 → 2:
    that count is which series-pair turn-offs carry a pA residual on the
    judge-less probe surface, rounding by its own description. Not
    removed: the factor `G + C/h` still cancels on the island's pivot
    (relative error ~eps·(C/h)/G_tie = 5e-6 at the default `h_e`), which
    now scales the increment, not the state. Checked on a scratch build
    with the history form back (everything else as committed): the new
    test fails in all six cases at 11.726609512 V and
    `test_mode_selector.py[bridge1_vf0]` reads 3 pins. → Reverse: the
    history right-hand side in `diode_port_problem_full`.
