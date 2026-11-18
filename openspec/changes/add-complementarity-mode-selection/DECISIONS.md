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

    **Finding (open, not a passivity matter):** `mode_selection='lcp'`
    is wrong on that bridge — 11.726610 V, D2 and D3 never conduct and
    D2 blocks +49.86 V — and identically with the transformer replaced
    by a floating sine source between the secondary nodes (n = 0.5 /
    1 / 2: 11.73 / 23.45 / 46.91 V against 31.88 / 63.75 / 127.50 V).
    The result flips with `H_E_REL` (wrong at 1e-10, 1e-9, 1e-7; right
    at 1e-8, 1e-6…1e-4). The shadow shows the LCP's open-circuit gap of
    D2 at its turn-on event as +2.3e-4 V (reverse) where the exact
    trajectory rises 0.31 V in 1e4·h_e: the port model's `q` at a
    turn-on event on a bus whose common mode is set by 1 MΩ and g_off
    leaks. The default (settle + guard) is right on all six. Recorded in
    KNOWN_LIMITATIONS; 'lcp' stays opt-in.

## Slice 4: the fixed engine (2026-11-13 … 2026-11-17)

Numbered F1, F2, … so they do not collide with #22–#31 of
`feat/core-a-complementarity` (not yet integrated here). The 2026-11-13
work in progress recorded its port model as "22."; F1 replaces it.

F1. **The pwl port model is the site's own system, solved directly, with
    every floating island's KCL imposed exactly (revises the WIP's
    "22." premise; the idea of #30, re-implemented for pwl).** Question: at which
    discretisation, and in which form, does a `pwl` decision site pose
    its LCP? → Option taken: pass 0 is the site's own linear system
    `J(mask₀, dt)·s = −(b_c + b_extra)` — the trapezoidal step at `dt`,
    a restart stage on the factor of `2γh` with its backward-Euler
    history, a split sub-step at `dt₂`, the static step at `dt = 0` —
    solved through the cache's factor of the settle's first mask
    (`cache.solve_at(mask₀, dt, b_extra)`: the very solve the settle's
    first round makes; on the static path the engine solves through its
    segment, the port model through the dt = 0 entry of the same
    matrix), so the LCP and the settle answer the same discrete
    question. After each solve, every floating island (nodes
    joined by capacitors, inductors, sources, ON switches / diodes and
    resistors > 1 mS, tied to the rest only through OFF switches /
    diodes and resistors ≤ 1 mS) gets its KCL sum imposed exactly:
    `σ = g·s − Σ_S f`, `s −= w·σ/(g·w)`, `w = J⁻¹e_ref`, with `g` the
    island's summed row from its boundary branches alone and `Σ_S f` the
    exact (Neumaier) sum of the site's right-hand side over the island's
    rows (so a user `b_extra_fn` injection is honoured, unlike #30's
    analytic zero, which needed #30's follow-up refusal). One more
    back-substitution per island and site; several islands are corrected
    jointly (`K·α = σ`, `K(m, n) = g_m·w_n`: a weak edge between two
    buses moves both sums); disabled for a circuit with a
    device kind the rule does not classify (controlled sources,
    transformers, Newton devices). The WIP's increment form
    (`J·δ = −(b_c + b_extra + J·x₀)`, `u = v(x₀) + v(δ)`) is dropped: it
    is no more accurate than the direct solve (table).
    → Evidence (Catch2 `[pwl_ports]`, floating-bus rectifier 100 V,
    1 mF ‖ 50 Ω, both diodes OFF, tie 1e4 … 1e9 Ω, sub-step `f·dt`,
    dt = 10 µs; reference: the same system assembled and solved in
    double-double arithmetic (~32 digits; `long double` is `double` on
    arm64, so the previous "long-double" reference — and the f = 1e-3
    numbers quoted in the wind-down — were double-precision LU against
    double-precision LU); worst |u − u_ref| over both ports):

    | tie Ω | f | direct cache solve | increment form (WIP) | direct + island KCL |
    |---|---|---|---|---|
    | 1e4 | 1 … 1e-3 | 1.2e-9 … 3.1e-6 | 8.0e-9 … 2.1e-5 | ≤ 6.0e-15 |
    | 1e6 | 1 … 1e-3 | 6.7e-7 … 5.2e-4 | 2.8e-6 … 8.0e-4 | ≤ 7.9e-15 |
    | 1e7 | 1 … 1e-3 | 1.8e-6 … 4.0e-3 | 1.1e-5 … 1.5e-2 | ≤ 1.1e-14 |
    | 1e9 | 1 … 1e-3 | 3.4e-4 … 3.2e-1 | 1.5e-4 … 4.3e-2 | ≤ 9.9e-15 |

    Neither form is accurate on an island; the form matters less than
    the island's common mode (the dsed finding of #30 holds on pwl). The
    test asserts ≤ 1e-10 V on u and ≤ 1e-12 relative on Z at every tie,
    f ∈ {1, 0.5, 0.1, 0.01, 0.001} and both an island and a grounded
    mask, and on two buses joined by 1 MΩ (ties 1e5 … 1e9 Ω). The engine's own settle solve is NOT corrected (it is the
    trajectory; changing it changes every floating-bus default run):
    the first column is the rounding the pwl settle itself decides on
    (KNOWN_LIMITATIONS). → Reverse: drop the `fix_` calls in
    `PwlModePorts::site`. Duplicated from the dsed adapter
    (`NativeCircuitBuilderAdapter::port_islands_of_`, c4255226): the
    island classification and the correction formula; to unify in a
    shared `pwl/port_islands.hpp` once both branches are integrated.

F2. **On pwl every selector pass, the tie widenings included, is posed
    on the site's own step.** Question: what does the port callback
    return for the selector's 100·h and 1e4·h tie passes on the fixed
    engine? → The WIP posed them as backward Euler over 100·dt and
    1e4·dt from the step start (0.1 s = five 50 Hz cycles at dt = 10 µs)
    with the sources frozen. Measured on the single-phase bridge
    (100 V, 50 Hz, 1 mF ‖ 100 Ω, V_f = 0, bus on a 1 GΩ tie, dt = 10 µs,
    40 ms, standalone C++ driver on run_transient): 'lcp' ended at
    p − n = 0.848 V (settle 95.171 V; trbdf2 95.1711166 V) with 990
    sites whose selected mode its own solve violated, and 'shadow'
    reported 846 disagreements. At the first wrong site (t = 12.14 ms,
    all OFF, bus at 93.2 V, source at −62 V) D2 leaks 5e-10 A forward —
    a tie at the site's step — and the widened passes (bus discharged
    over 0.1 s) turned D2 AND D3 on: the port model itself predicted
    −4410 A for that mode, matching the solve (−4410.07 A), so the model
    was right and the look-ahead wrong. The dsed band (#3.2) is
    h_e = 1e-7·T_ref, where 1e4·h_e is still a sliver of the period; on
    pwl the site's step IS the decision horizon (the next site is one
    step away). → Option taken: the callback returns the site's own
    (Z, u) at every pass (computed once per site); a tie left after the
    passes goes OFF unless it pins an island and the engine's judge
    rejects the all-OFF mode (the selector's own rule). The widened
    backward-Euler model (`PwlModePorts::widened`) is deleted. The
    selector-side remedy on feat/core-a-complementarity (#24: a widened
    pass stops short of the decided mode's next event) is not on this
    branch; it would not help here, the pass ends five cycles out.
    → Evidence: see the survey (F3). → Reverse: restore `widened` and
    the h ≠ h_site branch of the callback.

F3. **What the shadow found on pwl, and what the guard repairs.**
    Question: where do the settle and the LCP disagree on the fixed
    engine, and who is right? → Measured over 15 circuits × 3 configs
    (`probes/pwl_shadow_survey.py` / `.out`: buck at rest, DCM buck
    V_th = 0 and 1e-2, boost 50 / 500 Ω V_th = 0, half-wave RL, the
    single-phase bridges V_f = 0 / 0.7, the three-phase overlap and
    C-filter bridges, L·(1 ± δ) × 6, the 5- and 10-bridge banks with
    0.2 mH lines); each disagreement judged by the engine's law on the
    site's solve under both modes. No disagreement at all on the
    converters, the half-wave rectifier, the overlap bridge and the
    banks; 'lcp' is bit-identical to the default there. On the
    single-phase bridges 505–567 (V_f = 0.7) and 21–47 (V_f = 0)
    disagreements, nearly all "both" (a diode at the leak current, both
    labels consistent; the waveforms agree to 3.1e-11 V); the LCP alone
    right at 2 sites per config of the V_f = 0 bridge (pwl, restart),
    the settle alone right at 1–27 sites (the LCP's choice violates the
    raw, uncorrected solve by ≤ 3e-4 V — the island rounding of F1, the
    same reason 'lcp' counts 3–87 `inconsistent` sites there); the
    three-phase C-filter bridge 1 "both". The settle ended without a
    fixed point 25–43 times on the V_f = 0 bridge, nowhere else; the
    guard repaired 2 of them in the pwl and restart configs (D1 at
    105.12 / 125.12 ms, 9.9e-7 V; D2 at 115.12 / 175.12 ms, 9.1e-4 V);
    old → new: pwl, max |Δ(p − n)| 4.5e-13 V, end 95.171220828299 V
    unchanged; restart, max |Δ(p − n)| 7.07e-3 V near 185 ms, end
    95.177877798615 V unchanged, distance to trbdf2 over 175–190 ms
    7.099e-3 → 7.098e-3 V. Every other default run is bit-identical.
    The restart and split runs of the V_f = 0 bridge settle 6.8e-3 /
    4.5e-3 V off trbdf2 (95.1711166 V) where 'lcp' lands within
    3.1e-5 / 4.6e-7 V: a converged settle decided on the island's
    rounding, which the guard (it judges breaches only) cannot see. →
    Option taken: the guard stays breach-only (a converged settle is
    consistent with its own solve by construction; judging it on the
    corrected model would move every floating-bus default run); 'lcp'
    stays opt-in on pwl. → Rounding fuzz (20 last-bit perturbations of
    C and R): one waveform within 2.7e-11 V on every bridge, one mode
    sequence on the three-phase bridge; on the single-phase bridges ≤ 6
    of 4001 samples label a diode differently while every diode carries
    < 1e-7 A (ties). → Found on the way (engine, not changed here): a
    restart stage or split sub-step entered with the previous stage's
    mask cycle never solves — `mask_cycle` is kept per outer step on
    purpose (breach reporting), and `run_event_iteration` breaks on it
    before its first round. The shadow and the guard skip such a site
    (they read an empty settled mode before; the review found it in two
    survey rows). 'shadow' leaves the trajectory bit-identical but not
    the cache counters: its judge solves candidate masks through the
    cache (event entries, restart factorizations, segment searches
    move). → Reverse: `_PWL_MODE_GUARD = False`.
