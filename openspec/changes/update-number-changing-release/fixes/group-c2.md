# Group C2 — the mode of an edge under the restart rule (F-3-7) and the restart rule's open items

Branch `fix/core-release-c2-edge-mode`: `fix/core-release-a4-jump-test` d83d9c80 (code 27d83718) +
merge of `fix/core-release-c-mode-selection` d014f049. Tests: `python/tests/test_release_findings_c2.py`,
`test_release_findings_a.py::test_f37_*`. Marks: **[M]** measured, **[R]** read in the code.

The merge alone: every byte compare identical to the branch base (goldens, stateful, engines, steps,
pwl_identity 379/379 incl. the restart / split entries, trbdf2_identity 371/371) [M].

## Task 1 — F-3-7: saturable-primary PSFB aborts under the restart rule

**Cause** (group A's trace, 27d83718; confirmed and extended here) [M]. Restart stage 1 at a gate edge
is solved by Newton in the STEP-START diode bits (D_LS_A OFF at the HS_A turn-off): the saturable L_leak
is cut into g_off, Newton's first iterate puts mid_a at −1.26e4 V, and at that |x| the FLOATING secondary
(rect_neg / vout tied to the rest only by the diodes' 1e-9 S leaks) cannot converge: its common mode moves
volts per 1e-9 A of round-off residual. The event iteration never reaches a re-decision. Two facts
separate the two factors:
- The same circuit with the secondary grounded (rect_neg → gnd by 1 kΩ or 1 mΩ) ran under the rule
  BEFORE the fix (Newton converges in the stale mode, the re-decision flips D_LS_A) — but the floating
  and the 1 kΩ variant then gave different outputs (6.866415 vs 6.865542 V split): the trajectory
  depended on what the stale-mode Newton converged to.
- After the fix all three (floating, 1 kΩ, short) give 6.8655416568 V (split) / 6.9016607565 V (no
  split), identical to 10 digits.
Linear primary: its stage is one LU solve per mask, so the event iteration's own solves are the
decision (none kept from a stale mask) — it passed by construction of the linear path, not by luck;
'lcp' / 'auto' decide at the site (`mode_lcp_site`) before any solve.

**Fix** (`run_transient.hpp`, one place: `decide_edge_mode`, called by `run_restart` before stage 1 —
rule-on only; a first version inside `run_event_iteration` cost +0.6 % on the default buck_pwl). Restart stage 1
at a LOCATED event (`loc < γh`: gate edge / unknown start / source jump at the step start, a scheduled
split event, a split-located crossing) first decides the diode set consistent with the pre-event state x0
in the new switch configuration: one Newton iteration from x0 (the stage's linearization at x0, written
to a scratch vector, never kept), `events::redecide` on it, repeated until no diode flips (bounded by
`max_event_iterations`). Newton then runs from x0 in that mode. The flips count as the stage's own
(`iters`), so the impulse probe's widening and the restart window behave as when the settle flipped them
— unless the settle ends on the stage's entry bits (a linearization's flip the converged state undid:
iters 1, no widening; reviewer M2). Only Newton stages run it (a linear stage IS its linearization); default path untouched (no restart).

**Selectors.** Newton circuit: `'legacy'` → the settle + this decision; `'auto'` resolves to the settle
for a Newton device on pwl → the same; `'lcp'` refuses a Newton device by name (unchanged, tested).
Linear circuit: all three agree on the output to 10 digits (the absolute v(vout) differed by the floating
common mode, mV — the old test compared absolute node voltages; it now compares v(vout) − v(rect_neg)).

**Numbers** (v(vout) − v(rect_neg) at 200 µs) [M]:
- Truth, LINEAR primary: mpmath `exact_truth.py` (fa4b5c71, 30 digits), the switch_fn's exact edges
  (leg B at T/2π off the grid), secondary tied to ground by 1 GΩ: **6.856714473154 V**.
  pwl+restart+split at dt 50 / 25 / 12.5 / 6.25 ns: 6.8567143992 / ...4495 / ...4634 / ...4711
  (errors −7.4e-8, −2.4e-8, −9.7e-9, −2.0e-9). trbdf2 rtol 1e-8: 6.856716543 (+2.1e-6);
  esdirk43 rtol 1e-7 / 1e-9: 6.856717566 both (+3.1e-6, not controlled by rtol).
- Without the split the gate edges act at the grid (leg B's edges move by up to 50 ns): another circuit,
  6.8926854 V (no truth for it); the old "restart-less truth 6.911 V" is that quantized circuit's
  restart-less run at 50 ns, not a truth.
- SATURABLE primary, split: 6.8655416568 / ...6940 / ...7045 / ...7064 at dt 50 / 25 / 12.5 / 6.25 ns:
  own limit 6.86554171 ± 1e-9 (the linear twin converges the same way to its truth). Before the fix:
  abort at 61.55 µs (split) / 21.55 µs (no split).
- Second, independent reference for the saturable value: NONE at the 1e-7 level. trbdf2 (rtol 1e-8) and
  esdirk43 (rtol 1e-7, 1e-9) on `fix/core-release-d2-trbdf2-stateful-restart` (cad6ab6b, built in
  `.scratch`, deleted) abort on the saturable variant at t = 0 (floating) / 5 µs (secondary grounded),
  Newton at h_min — the same edge class on trbdf2 (not this slice). Their linear twins: esdirk43 grounded
  6.85671455 (+7.6e-8 vs truth), floating 6.85671757 (+3.1e-6, not controlled by rtol: the floating
  island). Cross-check by a different integrator: leg B moved onto the grid (phase 2π·0.16, edges at
  1.5 / 1.6 µs), secondary grounded, the DEFAULT trapezoidal path (no restart) at dt 10 / 5 / 2.5 /
  1.25 ns: saturable − linear = 0.0096424, 0.0089608, 0.0089035, 0.0089680 V against the restart+split's
  0.0089752 V (dt 50 ns) / 0.0089751 (12.5 ns); the trapezoid scatters ±6e-5 on this circuit (its
  ringing at commutations), so it confirms the saturable increment to ~1e-5 only. The test pins the
  saturable value to its own limit (stated as such in the test), and the linear one to the truth.
- Not fixed (default path, not the rule): restart-less pwl at dt 5 ns on the floating saturable variant
  aborts at 1.495 µs; trbdf2 / esdirk43 abort at t = 0 (floating) / 5 µs (grounded) on the saturable
  variant on this tree. Both are the floating-island / saturable Newton class, outside the restart rule.

## Task 2 — synchronous_flyback, err_rms 1.287e-6 → 1.618e-6 (pwl+restart / pwl+split)

Compared event by event against the mpmath truth (`benchmarks/truth/synchronous_flyback.json`'s circuit,
`exact_truth.py` fa4b5c71, conductance form, 30 digits), 1cec4e8b's extension (`.scratch/so_1cec.so`) vs
27d83718 [M]. Every gate edge (199) is an impulse on the grid's scale: S_pri's g_off cuts the primary
leakage (τ ≈ 10 ps against dt 100 ns), the magnetizing current transfers to the secondary.

| quantity (first edge, 8 µs) | 1cec4e8b | 27d83718 | truth |
|---|---|---|---|
| record dW at dt 100 / 50 / 25 ns (J) | 1.79401e-5 / 1.79162e-5 / 1.78686e-5 | 1.7963969e-5 (all three) | ½·Lp(1−k²)·i_p(8µs⁻)² = 1.796442e-5 |
| i_p after the edge (sample 81) | −6.0e-7 A off (the partial jump's residual a(2−a)J) | +7.6e-14 A | 2.40087e-5 A |
| i_s after the edge (sample 81 / 82) | +9.0e-7 / +3.3e-7 A | +1.0e-6 / +1.0e-6 A | |
| V(out) err_rms / span over 2 ms (dt 100 ns) | 7.71e-7 | 1.22e-6 | |
| V(out) error at 2 ms, dt 100 / 50 / 25 ns | −1.12e-5 / −2.80e-6 / −7.0e-7 V | −1.62e-5 / −4.05e-6 / −1.02e-6 V | |

Cause: the change is the WHOLE JUMP (J = d/(1−a)²); the probe-on-every-step and the stage-1 allowance
do not enter (the record times are the same 198 edges, one more record at 52 µs; no diode in the
circuit). The +1.0e-6 A on i_s after each edge is common to both builds and scales as h² (9.99e-7,
2.50e-7, 6.26e-8 at dt 100 / 50 / 25 ns): the SDIRK restart's own second-order error at the edge
(x0+ = 2x1 − x2 carries O(γ²h²ẍ) for the states that keep moving). In 1cec4e8b the partial jump left
−6e-7 A in i_p, which the next step relaxed into i_s (×0.95: −5.7e-7 A), cancelling part of that
error — and booking an energy that is 0.14 % low at dt and drifts away from the truth as dt shrinks
(0.27 %, 0.53 %). 27d83718's record is dt-independent and within 2.5e-5 of ½·Lp(1−k²)·i_p². Both
converge at order 2 in V(out); 27d83718 is ~1.45× the error constant of 1cec4e8b. **Not a defect: the
price of a correct record; listed.** What would remove it: a quadratic probe extrapolation for the
states that keep moving (x0+ from x1, x2, x3 cancelling the h²ẍ term) — not done (it moves every
restart step of every circuit; its own reference_status pass).

## Task 3 — the jump the rule still misses

- **Zero-rise gate pulse at 5.6 / 5.95 ns** (a built-in source jump in (t_from + γh, t]). A time event
  inside a step must be a landing. With the split it is one (a split event at the jump; the restart starts
  AT it): one record 5 J at t_off, i_L → 0. The release resolves the split ON with the rule
  (`_plan.DEFAULT_SUBSTEP_STATE_CORRECTION`): **closed by construction in the release profile** — test
  `test_a_gate_jump_inside_a_restart_step_is_landed_by_the_release_profile` (monkeypatches the release
  value, passes [M]). With the split explicitly OFF it stays a miss (strict xfail with the cause): that
  configuration opts out of landings.
- **Continuous 1 ps fall at 1.30–1.90 ns** (split on or off): a smooth Newton device (IGBT level 1)
  changes region inside a step; the ramp's corners are not events (only zero-rise jumps are split
  events), so nothing is located. What the engine needs: the source ramps' CORNERS as split landings
  (a breakpoint class kept out of the restart trigger), which locates the fall to its 1 ps sub-interval
  for any device it drives; or a state event on the device's region change. Not local: corners as
  landings move every ramped source of the reference set (MOSFET gate drives, `mos_l1` golden), and
  need their own reference_status pass. **Open**, strict xfail
  `test_a_continuous_gate_fall_inside_a_restart_step_is_an_impulse[split]`.

## Verification
(in progress)
