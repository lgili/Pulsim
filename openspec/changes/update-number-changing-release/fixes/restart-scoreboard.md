# Restart-rule scoreboard columns after group B: metric or defect

Branch `docs/core-restart-scoreboard-investigation` (from the PR head be5454f3).
Extension `so_be5454f3` (head) and `so_b9e46778` with the b9e46778 Python package
(base). Probes in `.scratch/sbinv/` (not committed): `decomp.py` (per-cell
decomposition), `dump.py` (rows around an instant), `caploop*.py`, `minimal.py`.
The referee is the mpmath truth (`exact_truth.Solver`, the circuit's stored dps),
read through `at_float` at every graded row, exactly as `exact_scoreboard.grade`.

## Verdict: (c), no engine integration error; one record defect and two measure artefacts

1. **Every move is in the right-limit rows.** The stored samples are byte-identical;
   the only new rows in the graded trace are `with_right_limits`' rows one ulp after
   each `right_limit_index`. Replacing each right-limit row by the engine's OWN left
   value at the same instant reproduces the b9e46778 score in all 47 cells (column
   "right := own left" below, equal to "before" to 3 digits). Replacing it by the
   truth's own value gives a score <= before in all 47. Nothing the engine integrates
   got worse or better.
2. **Record defect F-SB-1 (real, the release publishes it).** On the fixed engine
   under the restart rule, `right_limit_states` is not the right limit: it is the
   start of the restart step's quadrature line, `2·w1·X1 + (1 − 2·w1)·X2`
   (`run_transient.hpp` `restart_lined`), recorded at EVERY restart step (the event's
   and the restart window's next two, and t = 0). It integrates right (group B's
   charge / loss identities) but as a value it is O(h²·x'') off on a smooth stretch
   and O(1) off after a stiff transient, and it puts a "jump" on continuous states
   at instants with no event. `with_event_limits` / `with_right_limits` document
   it as "the exact trajectory one ulp after the event", so every reader (plots,
   max|v|, settle, the scoreboard, the strict gate) takes it as a value. This is
   25 of the 25 worse cells (alone or with item 3). Parked:
   `python/tests/test_restart_scoreboard_findings.py` (strict xfail).
3. **Measure artefact: the 0+ row.** `metrics.err_exact(exclude_t0=True)` drops
   rows with t == t[0] only; the right limit at t = 0 (t = 5e-324) stays graded.
   The strict gate's `_window` drops it since 14q (STATE "new reference-gate
   rule"); the exact measure did not follow. `three_phase_diode_rectifier`: one row,
   V(out) 17.22 V (line start of the first step) against the truth's 0 V, carries
   99 % of the score: 0.0127 with it, 0.000998 without (= before, exactly).
4. **Measure artefact: event timing graded by row weights.** A sample whose step
   holds a truth event displaced by δ > DELTA = 1e-3·h is compared with the wrong
   branch and charged jump² × its trapezoid weight, not jump² × δ (the snap /
   gap charge only covers δ <= DELTA). Inserting a right row halves the left row's
   weight, so the same timing error now reads 1/√2 of before (the "better" cells:
   `diode_half_wave_rl_load` 0.000949 -> 0.000671 = 0.707, `ll07_boost_vcswitch`
   0.0183 -> 0.0130, `vcswitch_resistor` pwl+restart 0.0592 -> 0.0447), and a tiny
   δ is charged a half step (`vcswitch_resistor` pwl+split 0 -> 0.0316: the split
   puts the gate event at the 1 ns ramp's start, 0.5 ns = 0.005 h before the 5 V
   threshold crossing; true L2 cost jump²·δ, charged jump²·h/2, 100× in the square).
   Before and after are both wrong here; neither is an engine change.

The seven disagree -> agree outcomes are the strict `interp` gate, which reads the
same right-limit rows (fewer ramps interpolated across jumps); they inherit item 2.

## The 47 moved cells (render's list), decomposed

Scores of the same cell re-graded with the right-limit rows modified (head
extension, same truth, same `grade` arithmetic; `decomp.py` asserts its own
score equals `metrics.err_exact` and the json value). "drop 0+": the t = 0 right
row removed. "right := own left": additionally every right row holds the engine's
left value at that instant. "right := truth": additionally every right row holds
the truth candidate nearest to it. "right rows (mode chg)": number of right-limit
rows, those whose mask differs from the stored row's in brackets (the rest are
restart-window rows with no event).

| circuit | config | before | after | ratio | right rows (mode chg) | drop 0+ | right := own left | right := truth | class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| back_to_back_diodes_sine | pwl+restart | 1.93e-16 | 1.43e-09 | 7.41e+06 | 16 (10) | 1.42e-09 | 1.96e-16 | 1.92e-16 | line-start values |
| back_to_back_diodes_sine | pwl+split | 1.94e-16 | 1.64e-09 | 8.46e+06 | 18 (12) | 1.64e-09 | 1.98e-16 | 1.93e-16 | line-start values |
| back_to_back_diodes_sine | pwl+next | 1.96e-16 | 1.64e-09 | 8.36e+06 | 18 (12) | 1.64e-09 | 2e-16 | 1.95e-16 | line-start values |
| diode_bridge_rectifier | pwl+restart | 2.21e-05 | 4.43e-05 | 2 | 51 (31) | 1.89e-05 | 2.21e-05 | 1.53e-05 | 0+ row + line-start values |
| half_wave_rectifier_sine | pwl+split | 2.56e-06 | 1.58e-05 | 6.16 | 21 (11) | 8.39e-06 | 2.56e-06 | 2e-06 | 0+ row + line-start values |
| half_wave_rectifier_sine | pwl+next | 2.56e-06 | 1.58e-05 | 6.16 | 21 (11) | 8.39e-06 | 2.56e-06 | 2e-06 | 0+ row + line-start values |
| voltage_doubler | pwl+split | 2.19e-07 | 2.19e-06 | 10 | 45 (25) | 2.02e-06 | 2.19e-07 | 1.58e-07 | line-start values |
| voltage_doubler | pwl+next | 2.01e-07 | 1.16e-06 | 5.8 | 41 (21) | 7.95e-07 | 2.01e-07 | 1.46e-07 | 0+ row + line-start values |
| transformer_step_up | pwl+restart | 4.03e-16 | 8.19e-10 | 2.03e+06 | 3 (0) | 8.1e-10 | 4.03e-16 | 4.03e-16 | line-start values |
| transformer_step_up | pwl+split | 4.03e-16 | 8.19e-10 | 2.03e+06 | 3 (0) | 8.1e-10 | 4.03e-16 | 4.03e-16 | line-start values |
| transformer_step_up | pwl+next | 4.03e-16 | 8.19e-10 | 2.03e+06 | 3 (0) | 8.1e-10 | 4.03e-16 | 4.03e-16 | line-start values |
| chopper_with_rc_snubber | pwl+restart | 2.5e-08 | 2.88e-07 | 11.5 | 59 (35) | 2.88e-07 | 2.5e-08 | 2.44e-08 | line-start values |
| chopper_with_rc_snubber | pwl+split | 3.11e-08 | 2.79e-07 | 8.98 | 59 (35) | 2.79e-07 | 3.11e-08 | 2.98e-08 | line-start values |
| chopper_with_rc_snubber | pwl+next | 3.11e-08 | 2.79e-07 | 8.98 | 59 (35) | 2.79e-07 | 3.11e-08 | 2.98e-08 | line-start values |
| pwm_voltage_rc | pwl+restart | 1.38e-08 | 4.86e-08 | 3.52 | 1500 (0) | 4.86e-08 | 1.38e-08 | 1.27e-08 | line-start values |
| pwm_voltage_rc | pwl+split | 1.38e-08 | 4.86e-08 | 3.52 | 1500 (0) | 4.86e-08 | 1.38e-08 | 1.27e-08 | line-start values |
| pwm_voltage_rc | pwl+next | 1.38e-08 | 4.86e-08 | 3.52 | 1500 (0) | 4.86e-08 | 1.38e-08 | 1.27e-08 | line-start values |
| cascaded_boost_buck | pwl+split | 2.37e-07 | 6.02e-07 | 2.54 | 1334 (855) | 6.02e-07 | 2.37e-07 | 2.31e-07 | line-start values |
| cascaded_boost_buck | pwl+next | 2.37e-07 | 6.02e-07 | 2.54 | 1333 (854) | 6.02e-07 | 2.37e-07 | 2.31e-07 | line-start values |
| flyback_converter | pwl+restart | 2.55e-06 | 6.47e-06 | 2.54 | 600 (400) | 6.47e-06 | 2.55e-06 | 2.53e-06 | line-start values |
| flyback_converter | pwl+split | 2.55e-06 | 6.47e-06 | 2.54 | 600 (400) | 6.47e-06 | 2.55e-06 | 2.53e-06 | line-start values |
| flyback_converter | pwl+next | 2.55e-06 | 6.47e-06 | 2.54 | 600 (400) | 6.47e-06 | 2.55e-06 | 2.53e-06 | line-start values |
| three_phase_diode_rectifier | pwl+restart | 0.000998 | 0.0127 | 12.8 | 229 (125) | 0.000998 | 0.000998 | 0.000987 | 0+ row |
| three_phase_diode_rectifier | pwl+split | 0.000996 | 0.0127 | 12.8 | 206 (111) | 0.000996 | 0.000996 | 0.000985 | 0+ row |
| three_phase_diode_rectifier | pwl+next | 0.000996 | 0.0127 | 12.8 | 206 (111) | 0.000996 | 0.000996 | 0.000985 | 0+ row |
| vcswitch_resistor | pwl+split | 0 | 0.0316 | inf | 11 (6) | 0.0316 | 0 | 0 | timing: event 0.005 h early charged as a half step |
| ll09_half_bridge_inverter_leg | pwl+restart | 0.00981 | 0.0414 | 4.22 | 720 (478) | 0.0414 | 0.00981 | 0.00695 | line-start values |
| ll09_half_bridge_inverter_leg | pwl+split | 0.00981 | 0.0414 | 4.22 | 720 (478) | 0.0414 | 0.00981 | 0.00695 | line-start values |
| ll09_half_bridge_inverter_leg | pwl+next | 0.00981 | 0.0414 | 4.22 | 720 (478) | 0.0414 | 0.00981 | 0.00695 | line-start values |
| ll10_full_bridge_inverter_lc | pwl+restart | 1.64e-07 | 1.71e-07 | 1.04 | 360 (238) | 1.71e-07 | 1.64e-07 | 1.62e-07 | line-start values |
| ll10_full_bridge_inverter_lc | pwl+split | 1.64e-07 | 1.71e-07 | 1.04 | 360 (238) | 1.71e-07 | 1.64e-07 | 1.62e-07 | line-start values |
| ll10_full_bridge_inverter_lc | pwl+next | 1.64e-07 | 1.71e-07 | 1.04 | 360 (238) | 1.71e-07 | 1.64e-07 | 1.62e-07 | line-start values |
| full_bridge_dc | pwl+restart | 3.76e-09 | 8.02e-09 | 2.13 | 250 (149) | 8.02e-09 | 3.76e-09 | 3.68e-09 | line-start values |
| full_bridge_dc | pwl+split | 3.91e-09 | 4.87e-09 | 1.24 | 250 (149) | 4.87e-09 | 3.91e-09 | 3.78e-09 | line-start values |
| full_bridge_dc | pwl+next | 3.81e-09 | 4.82e-09 | 1.27 | 248 (147) | 4.82e-09 | 3.81e-09 | 3.73e-09 | line-start values |
| cascaded_h_bridge_5level | pwl+restart | 6e-16 | 6e-16 | 1 | 60 (38) | 6e-16 | 6e-16 | 5.99e-16 | unchanged (<2 %) |
| cascaded_h_bridge_5level | pwl+split | 6e-16 | 6e-16 | 1 | 60 (38) | 6e-16 | 6e-16 | 5.99e-16 | unchanged (<2 %) |
| cascaded_h_bridge_5level | pwl+next | 1.85e-16 | 1.85e-16 | 1 | 60 (38) | 1.85e-16 | 1.85e-16 | 1.85e-16 | unchanged (<2 %) |
| flying_capacitor_3level | pwl+restart | 1.53e-06 | 1.5e-06 | 0.976 | 240 (158) | 1.5e-06 | 1.53e-06 | 1.47e-06 | row weight split at a displaced event |
| flying_capacitor_3level | pwl+split | 1.53e-06 | 1.5e-06 | 0.976 | 240 (158) | 1.5e-06 | 1.53e-06 | 1.47e-06 | row weight split at a displaced event |
| flying_capacitor_3level | pwl+next | 1.53e-06 | 1.5e-06 | 0.976 | 240 (158) | 1.5e-06 | 1.53e-06 | 1.47e-06 | row weight split at a displaced event |
| mmc_4cell_chain | pwl+restart | 1.18e-13 | 1.18e-13 | 1 | 60 (38) | 1.18e-13 | 1.18e-13 | 1.17e-13 | unchanged (<2 %) |
| mmc_4cell_chain | pwl+split | 1.18e-13 | 1.18e-13 | 1 | 60 (38) | 1.18e-13 | 1.18e-13 | 1.17e-13 | unchanged (<2 %) |
| mmc_4cell_chain | pwl+next | 2.23e-13 | 2.23e-13 | 1 | 60 (38) | 2.23e-13 | 2.23e-13 | 2.22e-13 | unchanged (<2 %) |
| t_type_3level_halfbridge | pwl+restart | 2.1e-07 | 3.26e-07 | 1.55 | 180 (118) | 3.26e-07 | 2.1e-07 | 2.02e-07 | line-start values |
| t_type_3level_halfbridge | pwl+split | 2.1e-07 | 3.26e-07 | 1.55 | 180 (118) | 3.26e-07 | 2.1e-07 | 2.02e-07 | line-start values |
| t_type_3level_halfbridge | pwl+next | 2.1e-07 | 3.26e-07 | 1.55 | 180 (118) | 3.26e-07 | 2.1e-07 | 2.02e-07 | line-start values |


Class "line-start values": the movement goes away when the right rows hold the
engine's own left value (item 2); "0+ row": it goes away when the 0+ row is dropped
(item 3); "0+ row + line-start values": both.

## The nine cells, decomposition (i)-(iv)

(i) smooth error = the engine's left-limit rows (identical old/new, column "right :=
own left"); (ii) event instants = first stored row in the new mask minus the last
truth event, in steps (upper bound of the engine's instant error on pwl+restart);
(iii) one-sided limits at the engine's instant vs the truth's limits (RMS / span,
rows excluding 0+); (iv) what the pointwise grade adds for a jumping column when
the instants differ by δ.

| cell | (i) left rows rms/span | (ii) instant offset, steps (mean / max) | (iii) left at events / right rows | (iv) | old measure saw | new measure sees |
|---|---|---|---|---|---|---|
| three_phase_diode_rectifier restart | 0.00114 | 0.61 / 1.37 | 5.4e-4 / 5.6e-4 (V(out) continuous; right = left to 6e-4) | none (no jump in V(out)) | (i) | (i) + 0+ row 17.22 V vs 0 V (99 % of 0.0127) |
| chopper_with_rc_snubber restart | 1.3e-7 | 0.74 / 1.0 | 1.1e-7 / 6.8e-6; no-event rows 1.5e-7 / 6.2e-6 | none | (i) | (i) + line starts: worst t = 1.3 ms, no event, 2.475242 vs left = truth 2.475477 |
| voltage_doubler split | 1.0e-6 | 0.74 / 1.01 | 1.4e-6 / 1.8e-5; no-event 1.3e-7 / 4.7e-6 | none | (i) | (i) + line starts (10×) |
| half_wave_rectifier_sine split | 1.6e-5 | 0.70 / 0.94 | 2.2e-5 / 8.0e-5 | none | (i) | (i) + 0+ row (71 % of the square) + line starts |
| back_to_back_diodes_sine restart | 2.7e-16 | event rows at the grid | 2.4e-16 / 1.8e-8; no-event 3.1e-16 / 3.9e-8 | none | round-off | line starts of a sine: 3e-6 vs truth 3.14e-6 at 0.02004 s (2 steps after the event, no event there) |
| transformer_step_up restart | 1.1e-18 | no events | — / 3.6e-8 (3 rows: t = 0, h, 2h) | none | round-off | the three start-up window rows (8.2e-10) |
| ll09_half_bridge_inverter_leg restart | 0.040 | 1.0 / 1.0 | 0.048 / 0.29 | V(mid) continuous with τ ≈ ns after the edge | (i) | (i) + line start 48.98 V vs truth 99.99 V both sides at 0.924 ms (stiff transient; L-stable steps right, the line start is a step average) |
| diode_half_wave_rl_load restart (better) | 9.5e-4 | 0.40 / 1.0 | 0.0184 / 6e-6 | event 0.0041 h before the grid row; left row −0.309 V vs the right branch | (iv) at full row weight | (iv) at half weight (0.707×); true L2 cost jump²·0.0041h |
| ll07_boost_vcswitch restart (better) | 0.0183 | 1.25 / 1.5 | 0.072 / 0.0022 | gate sampled half a step late | (iv) at full weight | (iv) at half weight |
| vcswitch_resistor split (0 -> 0.0316) | 0 | 0.99 (record) ; true instant 0.005 h early | 0 / 0.58 | right row ON at 20 µs, truth ON at 20.0005 µs | nothing (left rows exact) | jump² × h/2 for δ = 0.005 h |

## Control: engines that did not move (step 4)

Same probe, same circuit, same measure:

| cell | right rows (mode chg) | old / new | right := own left | right rows rms/span | left rows at events |
|---|---|---|---|---|---|
| three_phase_diode_rectifier dsed | 96 (96) | 1.9377e-14 / 1.9377e-14 | 1.9377e-14 | 2.34e-14 (V(out): right − left = 0 exactly) | 2.34e-14 |
| three_phase_diode_rectifier trbdf2 | 0 | 2.979e-6 / 2.979e-6 | — | — (no record) | — |
| ll09_half_bridge_inverter_leg dsed (a column that jumps: right − left up to 0.55 span) | 239 (239) | 3.65e-13 / 4.02e-13 | 3.65e-13 | 7.7e-11 | 3.5e-15 |

The measure grades the three engines' right rows the same way. The exact engine's
are true right limits (one per event, none at t = 0+, a continuous column has
right = left exactly), and they grade at the truth's evaluation error. The trbdf2
column records none. Only the fixed engine's restart rows hold values that are not
values.

## F-SB-1, the record defect (minimal case, parked)

10 V -> ideal switch -> 1 Ω -> 1 µF from 0 V, gate closing at 2.25 µs, pwl dt = 0.5 µs
= τ/2, `discontinuity_restart=True` (split on or off gives the same rows):

| t (µs) | left V(c) | right row V(c) | truth (continuous) |
|---|---|---|---|
| 0, 0.5, 1.0 (window at start) | 0 | 0 | 0 |
| 2.5 (closing seen; truth 2.25) | 0.000000 | 0.163174 | 2.2120 |
| 3.0 (no event) | 3.967365 | 4.065802 | 5.2763 |
| 3.5 (no event) | 6.360732 | 6.420115 | 7.1350 |

A capacitor voltage reads a 0.163 V jump at the closing and jumps of 0.098 V and
0.059 V where nothing happens. Test: `test_a_continuous_state_has_equal_left_and_right_limits`
(strict xfail, atol 1e-9 V; fails today by 0.163 V). The 0.5-step late closing is the
sampled `switch_fn` (a separate, known timing property), not this finding.

What the engine should record (engine track's decision, not done here):
- a right limit only where something jumps (a mask change or a source
  discontinuity at the step start), not on the window's next two steps nor at t = 0
  unless the initial state is inconsistent;
- as the value, the state at t_e+ (x_n for every differential state; the new mode's
  algebraic solve at x_n for the rest), and the quadrature line start in a separate
  field (`StepLines.x_start` already exists) for the loss port and
  `pl_integral(y_start=...)`, which need it. The comment in `restart_lined`
  ("record the extrapolation separately if a reader needs the value there") names
  exactly this; the reader now exists (with_event_limits, the scoreboard, the gate).

## Proposed measure (not applied: the scoreboard's definition changes by decision only)

**Applied (branch `feat/core-scoreboard-measure`, by decision; DECISIONS section 6).** As
implemented the rules differ from the proposal below in three places: the smooth part
leaves out only the rows inside a matched event's displacement (the fresh-context review
showed a one-step exclusion hides the row closing an off-grid event's step); a missing
truth event is charged jump² × (time to that device's next event, at most the 2 dt
window); `max δ/h` uses the netlist dt for every engine. Definition: `metrics.exact_parts`.

Grade three things separately and show all three in the cell:

1. **Smooth error E_s**: `err_exact` over the stored (left-limit) rows only, with
   (a) every row in (t_e − h, t_e + h] of any truth event or any engine event
   left out, and (b) every row with t <= nextafter(t0) left out (the 0+ row, as
   `_window` does). Time weights from the remaining rows' own trapezoid.
2. **Event instants E_t**: match each truth event to the engine's nearest event of
   the same flip (engine event = mask change in the stored + right-limit record, or
   an inner event); δ_e = t_eng − t_truth. Report max |δ_e|/h and the true L2 cost
   `sqrt(Σ_e jump_e² · |δ_e| / T) / span` (jump_e from the truth's two limits). An
   unmatched event is reported as such, never folded in.
3. **One-sided limits E_lim**: at each matched pair, |y_left(t_eng) − y_truth_left(t_truth)|
   and, for an engine whose record claims right limits,
   |y_right(t_eng) − y_truth_right(t_truth)|, divided by span; max and RMS.

Cell summary = max(E_s, E_t, E_lim), with the argmax named; the three are printed
beside it. Under this measure the 47 cells would read: E_s = the "right := own
left" column restricted away from events (unchanged by group B by construction);
E_lim on pwl+restart would expose F-SB-1 directly (0.163 V / 10 V on the minimal
case, 0.51 span on ll09) instead of mixing it into a time-weighted RMS; E_t would
grade the 0.005 h early split event at its real cost. The snap window DELTA and the
gap charge are then not needed. Until F-SB-1 is fixed, the fixed-engine right rows
should be graded under E_lim and flagged, not inserted into E_s.

## The caploop ledger (`_caploop_closing`, test_impulse_losses.py)

`C1 = 1 µF @ 10 V`, `C2 = 3 µF @ 2 V`, ideal switch g_off = 1e-9 S, gate edge at
10.5 h, pwl + restart. The engine sees the edge at 11 h (sampled switch_fn); the
pulse step is (11h, 12h]; `rec.dW = W(11h) − W(12h)`.

| h | E_total head | E_total base | leak (h, 11h] of the engine's trajectory | head: E_imp + E_total − (W(t_window_start) − W_end) | base: same | the test's form (dW from t = 0), head |
|---|---|---|---|---|---|---|
| 1e-8 | 6.400069e-15 | 6.720064e-15 | 10 steps | 1.4e-19 | +3.2e-16 | −6.4e-16 |
| 1e-7 | 6.399992e-14 | 6.719995e-14 | 10 steps | 4.1e-19 | +3.2e-15 | −6.4e-15 |
| 3e-7 | 1.920000e-13 | 2.016000e-13 | 10 steps | 1.0e-20 | +9.6e-15 | −1.9e-14 |
| 1e-6 | 6.400000e-13 | 6.720000e-13 | 10 steps | −3.0e-20 | +3.2e-14 | −6.4e-14 |
| 1e-5 | 6.399999e-12 | 6.719999e-12 | 10 steps | 3.4e-19 | +3.2e-13 | −6.4e-13 |

- 7.04e-12 (h = 1e-5) is the engine trajectory's leak over (0, 11h], 11 steps:
  W(0) − W(11h) = 7.039999e-12, and with the impulse record it closes exactly
  (W(0) − W_end = rec.dW + 7.04e-12, W(12h) = W_end). But `device_loss_summary`'s
  window starts at the first SOLVED sample, t = h (`t_window_start`), so its leak
  is (h, 11h] = 10 steps = 6.40e-12. **Head is right**: with the stored energy taken
  over the same window the ledger ΔE_stored + ∫p + E_impulse closes to <= 4e-19 J
  at every h (round-off of W ≈ 5.6e-5 J). Base (6.72e-12) booked half of the pulse
  step's pre-jump power on top of the impulse: +0.5 step, the artefact DECISIONS #42
  / group B removed.
- The test asserts the right identity over mismatched windows (dW from t = 0, loss
  from t = h) and passes only through pytest's default abs = 1e-12, which is larger
  than the whole leak at h <= 1e-6 and 1.6× the one-step error at h = 1e-5: it would
  pass base and head alike, and any error below 1e-12 J. Proposed assertion (not
  changed here):
  `k0 = searchsorted(t, s["S"]["t_window_start"]); dW = W[k0] - W[-1]`;
  `assert E_imp + E_total == pytest.approx(dW, rel=1e-13, abs=0)` (tolerance
  2.4e-18 J; head 4.1e-19 worst, base 3.2e-16 best: it can fail), plus
  `E_total == pytest.approx(10 * g * 64 * h, rel=1e-6)` for the regular part.
- Physical truth (edge at 10.5 h, closed form): leak (0, 10.5h] = 6.72e-12, impulse
  2.399999328e-05; the engine's impulse is 3.2e-13 smaller because its edge is 0.5 h
  late (sampled switch_fn), a timing property of the fixed engine, not of the ledger.

## What was run / not run

Run (head extension unless "base"): `decomp.py` on the 16 circuits of the 47 cells
× pwl+restart / pwl+split / pwl+next (each reproduces the committed json value to the
last digit), plus dsed and trbdf2 on three_phase_diode_rectifier, dsed on ll09, and
diode_half_wave_rl_load / ll07 / ll06 / vcswitch_resistor for the "better" side;
`dump.py` on vcswitch_resistor and ll09; caploop probes on head and base; the new
test (xfails; fails by 0.163 V with --runxfail); ruff on the test.
Not run: the scoreboard itself (no regeneration), the "before" scoreboard grades on
the base extension (the old trace is the head result without right rows: samples
byte-identical per the batch notes, and the reconstructed old score equals the
b9e46778 json value in every cell), ctest, the full pytest, any C++ change.

## Measure v2 — state (branch `feat/core-scoreboard-measure`, 2026-11-20)

### Definition as implemented (`benchmarks/tools/metrics.py`: `match_events`, `exact_parts`)

Per graded column, three parts, each a fraction of `span` (the truth's peak-to-peak as before);
cell = the largest part of the worst column (`exact`, `exact_part`, `exact_col`).

- **smooth**: `sqrt(Σ w_i e_i² / Σ w_i)` over the engine's STORED rows (`res.times`, no
  right-limit rows, so no 0+ row), e_i = distance to the truth at t_i (`at_float(..., 0.0)`, no
  snap window; two candidates only where t_i is a truth event instant, nearest taken). Left out:
  rows at t == t[0], and rows inside a MATCHED event's displacement `[min(t_eng, t_truth),
  max(...)]`. Every other row is graded (an earlier one-step exclusion hid the row closing an
  off-grid event's step: reviewer case).
- **instants**: `sqrt((Σ_matched jump²·|δ| + Σ_missing jump²·d) / T) / span`; δ per truth instant =
  largest over its devices, 0 within 4 ulp; a missing truth event charged with d = time to that
  device's next truth event, capped at the window. `max_dt_h` = largest matched δ / h.
  **h = the netlist dt for every engine** (pwl's step; the output grid / dt_max of trbdf2 and dsed).
- **limits**: max |engine − truth| over every one-sided value the engine records after t0: at a
  matched event its left value (row before the record change) and right value (the right-limit
  row one ulp later, if any) against the truth's limits at the TRUTH's instant; spurious engine
  events and every other right-limit row (F-SB-1's restart-window rows) against the truth at the
  engine's instant. trbdf2 records no right limits: graded on left values only.
- **Events**: truth = `sol.events` with a device flip (a source break without a flip is not an
  event; rows around it stay in smooth). Engine = changes of the switch record of
  `pulsim.with_event_limits(res)` (stored rows + split's inner events + right limits), instant =
  the earlier row's time, columns named by `builder.switch_index_of` (verified for all 81
  circuits, default and branch form). Key = device + new state. **Matching**: per key, in time
  order, nearest within 2 h after the key's previous match (chattering -> extras spurious; a
  cluster closer than h matched one by one). Unmatched engine events at t0 = initial mode
  selection (`n_initial`), not spurious. Events at/after the last stored row are not graded.
- **Not available**: `instants = None` (`n/a` in the table) when the run has no switch record but
  the truth has device events; summary = max(smooth, limits); with no exclusions every row is graded.
- Non-finite value anywhere (engine or truth) -> inf, part `non-finite`.
- `exact_old` = previous definition (`metrics.err_exact`, unchanged code, on the two-sided trace);
  reproduces the ad9ee191 json to ~1e-14 on the circuits probed.

Scoreboard side: `exact_scoreboard.grade` (reads `tr.result`; `differential_runner.Trace` keeps
the run), `event_record`, `switch_names`, `_old_exact`; render: `## The measure` (what changed
and why), summary with old counts and `set by`, `## Cells whose class changed with the
definition`, long-form full table; `--previous` compares `exact_old` with an old-definition json.
`accuracy_summary.py` updated (part, set-by, old counts).

### Tests (all pass on 39e68edf + head extension)

- `python/tests/test_exact_measure.py` (18): each part on constructed traces, the old measure's
  mispricing pinned, 0+ row, chattering, missing / spurious, no record, NaN truth, reviewer cases.
- `python/tests/test_impulse_losses.py` caploop: matched-window identity rel 1e-13 abs 0 plus the
  regular part; passes on head, fails at all five h on b9e46778 (package from `git archive` +
  `so_b9e46778`, run without the repo conftest).
- NOT yet re-run: `test_exact_scoreboard.py` (bounds need the new numbers),
  `test_exact_output_grid.py`, `test_exact_truth_tool.py`, `test_accuracy_summary.py`, full pytest.

### Running

`exact_scoreboard.py run --force` then `twins` (head extension, `nice -n 15`), records in
`/Users/lgili/pulsim_wt/p2-trbdf2/.scratch/exact_scoreboard/*.json` (+ `_twins.json`), log
`.scratch/sbm/run.log` (ends with a `done <rc>` line), twins log `.scratch/sbm/twins.log`.

### Remaining steps

1. When `done 0`: `exact_scoreboard.py render --previous ad9ee191` (expect 0 moved: same
   definition, same extension), then `accuracy_summary.py`; `--check` both; write `## Notes`.
2. Old-vs-new class counts per configuration (old at b9e46778: pwl 38/11/8, restart+split
   76/41/10, trbdf2 80/30/9, esdirk43 81/65/13, dsed 81/77/70) from the json.
3. Every cell that changes class with the part (render section lists them); worse cells as
   candidate findings. Already seen in probes: `vcswitch_resistor` pwl+split instants
   δ = −0.005 h (gate at ramp start); plain pwl δ = −0.015 h at t ≈ 0; `diode_half_wave_rl_load`
   dsed / pwl+split / pwl+next limits 0.030: the recorded right limit of V(out) at D1 turn-off is
   the value after the 1 ns g_off/L transient (−3.1e-6 V), the truth's right limit is −0.314 V
   (continuous); F-SB-1 in limits on pwl+restart / pwl+split / pwl+next (ll09 0.51,
   three_phase 0.032).
4. Re-pin `test_exact_scoreboard.py` bounds (each explained by its part); run
   `test_exact_output_grid.py`, `test_exact_truth_tool.py`, `test_accuracy_summary.py`,
   `test_benchmark_harness.py`.
5. Docs: DECISIONS section 6 (this definition), tasks 4.3 note; CHANGELOG entry is written;
   AI guide does not describe the scoreboard (nothing to change). Rehearsal cells stored on disk
   are the old definition until it is re-run.
6. Full pytest on the final tip (background, no commits while it runs), ruff,
   `openspec validate update-number-changing-release --strict`.
7. Reviewer pass done once (findings 1–4 fixed in 39e68edf; finding 5, the 2 h window cliff —
   an event 2.1 h off becomes missing + spurious and its left value reads a full jump in
   limits — documented, not changed). A second pass on the regenerated data is optional.

### Update (13375eb0): first full run, one more artefact fixed, re-run in progress

- First full run on 39e68edf: `exact_old` reproduced the ad9ee191 json in all 729 cells
  (`render --previous ad9ee191`: 0 moved). Old counts at the head extension: pwl 38/11/8,
  pwl+restart 70/31/8, pwl+split 74/38/8, trbdf2 80/30/9, dsed 81/77/70, esdirk43 81/65/13,
  pwl+next 75/39/9, trbdf2+next 80/30/9, dsed+next 81/77/70.
- It exposed an artefact: `cascaded_h_bridge_5level` / `mmc_4cell_chain` read limits ~1.0 on
  all nine configurations (dsed included). The truth places a complementary pair one ulp apart
  (S2_BH- at 0.00035, S2_BL+ one ulp later, an intermediate state between); engines flip both
  at once. Fixed in 13375eb0: truth events within 4 ulp share their outer limits. Still open
  (minor): the truth's values AT an engine instant (`_one_sided`, used for spurious events and
  unpaired right rows) take the last candidate at that float time, not the cluster's.
- The regenerated md/json/summary of that run are in the working tree, NOT committed (they carry
  the artefact). Re-run on 13375eb0: `run --force && twins && render --previous ad9ee191 &&
  accuracy_summary.py`, log `.scratch/sbm/run2.log` (ends `done <rc>`), twins
  `.scratch/sbm/twins2.log`. When `done 0`: check `accuracy_summary.py --check`, commit the
  three files, then the remaining steps above from step 2.
