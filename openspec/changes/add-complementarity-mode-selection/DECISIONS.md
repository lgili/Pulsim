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

