# F-SB-1 — an event's right limit is a state value (restart rule, fixed engine)

Branch `fix/core-restart-right-limit` from `origin/feat/core-integration-14r`
(6027f4fd). Finding and evidence: `restart-scoreboard.md` (branch
`docs/core-restart-scoreboard-investigation`, ad9ee191). Tests:
`python/tests/test_restart_right_limit.py`, the investigation's
`test_restart_scoreboard_findings.py` (its strict xfail removed: it passes).

## The decision implemented (the orchestrator's)

Two things were stored in one field. Now:

1. **The right limit is a state value** x(t_e+), recorded only where
   something happens. `StepLines::rl_*` (loss_tracker.hpp) → `record_lines`
   → `result.right_limit_*` / `inner_event_right*`.
2. **The quadrature start stays the lines' own** (`StepLines::x_start`,
   `right[]`, the held-pieces rows): the loss tracker and the in-loop
   recorder read them as before; they never reach the result.

## What x(t_e+) is, and how it is computed (`event_right_limit`, run_transient.hpp)

Each capacitor voltage / inductor current at its value at t_e+, every other
unknown solved in the event's mode there:

- continuous states: x0 exactly (the histories at the step start);
- a state an IDEAL element's constraint moves (branch form, the structural
  record): x0 + J, J the probe's jump (consistent_start's `ip_d`, flagged
  jumping) — right − left = the record's jump;
- a CONDUCTANCE-form element's unresolved fast mode that the impulse rule
  records (τ = L·g_off = 1 ps, a vcswitch cutting an inductor): the states are
  NOT relaxed. Its right limit is the model's own, i_L continuous and
  v = i_L/g_off — the exact engine's (F-5-10) — and the record reports the
  collapse that follows. This is the one reading of the decision's
  parenthetical "x0+ = consistent_start's" I did not take literally: with
  x0 + J there the row equals no engine's right limit, and the group-A
  hand-off (the impulse's peak visible in the right-limit row) cannot hold.
  For ideal elements both readings coincide.

The algebraic solve: the ε → 0 limit of a backward-Euler solve of ε from those
states in stage 1's mode (sources at t_e + ε read with no left-bias window, gates the step's own sample,
i.e. a source edge's right value even when t/h > 2e4), as two SINGLE solves
(`run_event_iteration(..., single = true)`: the mode selector's caches and
counters — mode_guard, lcp `mode_answers`, shadow — are the run's and are not
touched) of ε, 2ε, 4ε, 8ε and Richardson (below). The ε solve is re-decided
(≤ max_event_iterations rounds; the others keep its bits) — a located
DCM turn-off's residual current otherwise spikes through g_off (1.6 kV in the
boost diode-audit test). Event-factor thrash is avoided by an aux event table
(`PwlCache::swap_aux_events`), and `n_restart_factorizations` is put back
(the one-factor-per-mask test holds). A solve that throws (runtime_error, or
an UnsupportedError refusal) records no right limit; the run goes on.
Richardson: the motion cancels to O(ε^4); ε = 5e-4·γh (1e-9·γh at a conductance-form impulse). A mode with τ ≫ ε keeps
64·(ε/τ)^4; rounding grows as 1/ε (the L/ε, C/ε entries). Everything the solves touch is put back (x,
b_extra, histories, stateful histories, diode bits, iteration flags, the Newton
round-off budget and its counters); the step's trajectory does not change.

Where: `restart_lined(a, h, loc, ev)` — `ev` true at a trigger at the step
start that is an event (`rs_trig_ev`: a gate-mask change, a source / b_extra
edge; NOT an unknown start's stamp and NOT t = t_start), at a scheduled split
event (i > 0) and at a located split crossing; `run_restart` computes the
limit when `ev` or the probe found a jump (`ip_jump`: a recorded impulse at
any restart step, t = 0 included). Never at a window's later steps.

The held-pieces rows of a mixed-mode restart step (F-4-4: X1 held over w1·h,
then X2) are quadrature, not events: they stay in the lines and no longer
appear in `inner_events`.

## Numbers

| case | truth | before | after |
|---|---|---|---|
| 10 V → ideal S → 1 Ω → 1 µF, dt 0.5 µs: right rows | one, at the closing | 6 (t = 0, 0.5, 1.0, 2.5, 3.0, 3.5 µs) | 1 (2.5 µs; the sampled gate) |
| same, v_C right − left at the closing | 0 (continuous) | 0.163174 V | 4e-19 V |
| same, v_a / i right | 10 V / 10 A | 10 V / 9.84 A | 10 V / 10 A |
| C 1 µF (2 V) onto 10 V via ideal S | v_C+ = 10 V, J = −λ/C | line start | 10 V; J = record to 1e-9 (pwl and dsed) |
| L 10 µH cut by ideal S | i_L+ = 0, J = λ/L | line start | 0; J = record to 1e-9 (pwl and dsed) |
| 48 V, L 1 mH, switch g_off 1 nS (F-5-10 on pwl) | v_sw+ = i_L/g_off, i_L continuous | no right-limit row voltage | 2.4e9-class row, warning raised |
| ll09, every right row: i(Lload), V(mid) | continuous (V(mid) = f(i_L)) | V(mid) 48.98 V vs 99.99 V | right − left ≤ 1.2e-17 A / 7.7e-13 V |

(scoreboard cells: see Scoreboard)

## Voltage guard on the fixed engine (group A hand-off)

`solver::find_implausible_voltage(graph, pool, result)` now also scans the
result's right-limit rows and inner-event right rows (NaN rows skipped). The
fixed engine's finalize path already calls it (`_auto_check_voltage_sanity`
and `sim::Simulation::check_voltage_`), so the check costs no Python bytecode:
the overhead gate does not move. On a result without event limits the scan is
one bool test.

## Review findings (fresh reviewer on 452c2556) and their state

| id | finding | state |
|---|---|---|
| D1 | ε solves read a source edge on its pre side when t/h > 2e4 (64-ulp left bias) | fixed: sources read at t_e + ε with lb = 0 |
| D2 | the non-single event iteration ran the mode selector (mode_guard counters, lcp `mode_answers`, shadow): an lcp bridge trajectory moved | fixed: single solves + the settle's own `events::redecide` (no mode machinery) |
| D3 | UnsupportedError (invalid_argument) escaped the ε solves | fixed: caught, no row |
| D4 | constraint currents of ideal dependent states amplified ~1/ε | reduced: ε = 5e-4·γh (1e-9·γh only for a conductance-form impulse); see ll10 below. Not separately tested |
| D5 | `rl_want` stays set if `run_restart` throws | fixed: read and cleared at the top of `run_restart` |
| D6 | a non-split source-edge trigger inside the step (rs_trig_loc > 0) records a row at t_prev; split clusters read gates at ev_s, sources at ev_t + ε | open |
| D7 | a gate edge exactly on a resume seam (t_prev == t_start) records no row | open |

ll10 (found by the scoreboard re-grade, after D2): with every switch OFF in
the dead time the two legs float on 1 MΩ bleeds; at ε = 1e-9·γh the L/ε entry
(1.4e13) took the common mode (row v_a = −73326 V, v_b = 2 V; truth −36664 /
36668 V). Now ε = 5e-4·γh and a fourth-order Richardson
(64x(ε) − 56x(2ε) + 14x(4ε) − x(8ε))/21: conditioning 5e5 better, truncation
64·(ε/τ)^4 of a mode τ (a third-order one left 2.5e-10 V on the 1 µs RC of the
closing test: error ∝ (h/τ)³, measured). A conductance-form impulse keeps ε = 1e-9·γh (its
ps mode must not relax inside ε: F-5-10's micro right limit).

## Scoreboard (exact measure; `.scratch/sb2.py`, 20 circuits, pwl+restart / pwl+split)

28 cells were worse at the head than at b9e46778. Now (0dacc666 code; the
gate_t revert after it does not change values):

| circuit | cell | b9e46778 | head | now | |
|---|---|---|---|---|---|
| back_to_back_diodes_sine | restart / split | 1.93e-16 / 1.94e-16 | 1.43e-9 / 1.64e-9 | 1.93e-16 / 1.97e-16 | ok |
| diode_bridge_rectifier | restart / split | 2.21e-5 / 2.38e-5 | 4.43e-5 / 4.38e-5 | = b9e | ok |
| half_wave_rectifier_sine | restart / split | 1.01e-4 / 2.56e-6 | 1.08e-4 / 1.58e-5 | = b9e | ok |
| voltage_doubler | restart / split | 1.67e-6 / 2.19e-7 | 3.02e-6 / 2.19e-6 | = b9e | ok |
| transformer_step_up | both | 4.03e-16 | 8.19e-10 | 4.03e-16 | ok |
| chopper_with_rc_snubber | split | 3.11e-8 | 2.79e-7 | 3.11e-8 | ok |
| pwm_voltage_rc | both | 1.38e-8 | 4.86e-8 | 1.38e-8 | ok |
| flyback_converter | both | 2.55e-6 | 6.47e-6 | 2.56e-6 | ok (+0.5 %) |
| three_phase_diode_rectifier | both | 9.98e-4 / 9.96e-4 | 1.27e-2 | = b9e | ok |
| ll09_half_bridge_inverter_leg | both | 9.81e-3 | 4.14e-2 | 1.85e-7 | ok (better) |
| full_bridge_dc | both | 3.76e-9 / 3.91e-9 | 8.02e-9 / 4.87e-9 | 3.72e-9 / 3.82e-9 | ok |
| t_type_3level_halfbridge | both | 2.10e-7 | 3.26e-7 | 2.10e-7 | ok |
| vcswitch_resistor | split | 0 | 3.16e-2 | 6.7e-25 | ok |
| chopper_with_rc_snubber | restart | 2.50e-8 | 2.88e-7 | 2.57e-8 | worse by 2.8 % |
| cascaded_boost_buck | split | 2.37e-7 | 6.02e-7 | 3.16e-7 | worse (half the head's excess left) |
| ll10_full_bridge_inverter_lc | both | 1.64e-7 | 1.71e-7 | 2.46e-6 | worse: the dead-time right row's common mode (rounding through L/ε; 0.4 V of 36664 V) |

New, not worse at the head:

| circuit | cell | b9e46778 | head | now | |
|---|---|---|---|---|---|
| diode_half_wave_rl_load | split | 9.64e-4 | 6.82e-4 | 5.55e-2 | worse: the located D1 turn-off (10.1 ms) right row reads the RL residual current through g_off (a reverse spike, 2.48 spans); the truth's root has i = 0. Open: at a located turn-off the ON mode is the well-conditioned representative |

Mispriced by the measure: none separated; every remaining worse cell above
is a real right-row value.

## Gates (on 8d7b2a44's code, vs the branch base .so)

- Default options byte-identical: goldens, stateful (sbit), engines, steps,
  trbdf2_identity (371/371) — rc 0. pwl_identity default entries identical.
- Rule-on (pwl+restart, pwl+split): stored times and states identical to the
  base. Differences are only the right-limit / inner-event rows,
  `_implausible_voltage` (the new scan), and on 8 Newton circuits the work
  counters of the right-limit solves (`newton.*`, `stats.n_*`, total_bytes).
  The mode-selector counters (mode_guard, lcp) no longer move.
- Cost, default path: not certifiable to 0.5 % on this machine (load 10; buck
  pwl min-of-six 0.38–0.74 ms across alternating runs of the SAME binaries).
  ll09: pwl 2.317 vs 2.356 ms, trbdf2 8.6 vs 9.6 ms (new vs base). The default
  path's new work is one bool test in the voltage scan and the record
  condition. Rule-on: ll09 pwl+restart 3.73 vs 3.82 ms; buck 0.56–0.70 vs
  0.55–1.40 ms (noise).
- Overhead gate `test_simulate_overhead.py`: 2 passed.

## Must-hold items

1. Minimal case: one row (2.5 µs), v_C right − left 1.8e-14 V at dt 0.5 µs,
   none at 3.0 / 3.5 µs, none at t = 0 (split and not). ll09: continuous on
   all 239 rows (scoreboard cell 9.8e-3 → 1.85e-7).
2. Jumping states: C onto source and L cut — right − left = record's jump to
   1e-9, equal to dsed's (tests).
3. Recorder untouched: lines keep the quadrature start; buck window / energy /
   window-impulse-convention tests pass. test_f44 still XFAIL (window END
   inside a restart step reads the line start: a recorder item, not a right
   limit); kept.
4. Group A: the F-5-10 row shows i_L/g_off and the guard warns on pwl; the
   scan runs inside the existing finalize call (overhead gate passes).
5. F-W-1: does not fall out (both xfails stand: the pulse step's END state,
   not the right limit).

## Session 2 (7739d64a, 62b86ff1): located crossings, the ε-limit check, D4/D6/D7

### 1. A located crossing (the new regression, diode_half_wave_rl_load split)

Rule: at a LOCATED crossing the state is continuous (the predicate passes
through zero) and the crossing variable is at its crossing value. For a diode
with V_f = 0 the ON and OFF laws meet there (i = 0, v_D = 0), so the row is
solved with the crossing diodes ON and no re-decision: the locator's residual
(i_L = −2.6e-5 A, the location is 83 ns late) is read through R_on (0.13 µV),
not through 1/g_off (25.6 V). Sites: split_step's three branches (crossing
within snap of a, at b — the next step's start, only when b = t —, inside);
`rl_loc_t` / `rl_loc_on` name the time and the crossing diodes.
shortcut (in the code): a V_f > 0 turn-off keeps the OFF reading — its laws do
not meet at i = 0 (v_D jumps by V_f); the ON reading would need the drop
removed from the stamp.

| D1 turn-off, pwl+split, V(out) | left | right | truth (at 10.0999172 ms) |
|---|---|---|---|
| before (c43be8b4) | −0.314107461 | +25.5709346 | −0.313847579 / −0.313847579 |
| now | −0.314107461 | −0.314107461 | same |

The engine's row is at 10.1 ms (the crossing lands on the grid point b); the
truth continued to 10.1 ms is v_in(10.1 ms) = −0.31410759: left and right agree
with it to 1.3e-7 V (6e-9 of the 20 V span). Cell 5.55e-2 → 9.637e-4 (b9e
9.639e-4). Test: `test_a_located_turn_off_is_continuous_at_its_right_limit`
(dt 20 µs: on b; 30 µs: inside the step).

### 3. The ε-limit validity check

Richardson is valid where x(ε) is smooth on (0, 16ε]: each mode of the ε mode
either ε ≪ τ (polynomial motion) or ε ≫ τ (relaxed: a c·τ/ε tail). Not valid:
τ inside [ε, 16ε], rounding through L/ε and C/ε. Five solves ε…16ε; per
unknown P = (64x1 − 56x2 + 14x4 − x8)/21 is accepted when it agrees with the
same combination on 2ε…16ε within 1e-6 of the unknown's size; else
Q = (−8x1 + 22x2 − 13x4 + 2x8)/3 (it also annihilates a 1/ε term: a relaxed
mode; an ideal constraint the moved states meet only to the probe's
tolerance — D4; systematic rounding) against its shifted twin. An unknown
passing neither takes the next rung: 1e3·ε0 = γh/2 (the stage's own
conditioning), then 1e-3·ε0 (last: where rounding saturates a lost unknown is
constant in ε and passes either test — ll10 at 1e-3 showed it, 51 V). After the
ladder: the fallback, counted in `res.n_right_limit_fallbacks` — the impulse
rule's x0+ after a probed jump, else ε0's P (the states are x0 or x0 + J in
every solve: only algebraic unknowns fail; the left value is no fallback for
them — ll10's leg jumps 36 kV at the dead time).

Fallback on the 81 solved scoreboard circuits (rows with a fallback / right
rows incl. inner): pwl+restart 65 / 32648, pwl+split 120 / 29102, in
active_clamp_forward (6 / 45), half_bridge_transformer_dc (59 / 57),
diode_bridge_rectifier split (7), three_phase_diode_rectifier split (1),
voltage_doubler split (10 of 10). Their values are ε0's P, i.e. the session-1
values; every cell of these circuits in the re-grade is at or under b9e.

D4 (tested): C1 (10 V) — ideal S — C2 (0 V), R 1 kΩ / 500 Ω: the right-limit
switch current was 68.1 A (C·δv/ε of the probe's J tolerance, then the
fallback's impulse line); now 2.49376e-3 A = C2·dv/dt + v+/R2 at v+ to 1e-6, and
dsed's 2.49438e-3 A at its own v+ (`test_a_constraint_current_after_a_charge_sharing_jump_is_the_regular_one`).

D6: a split cluster's right limit reads the sources at ev_s + ε (the gates'
side). Without the split loc = 0: a source edge inside a non-split step is not
landed; the row at t_prev holds the t_prev+ values, which are the truth's
there (reading at t_prev + loc would put the post-edge source at t_prev).
D7: `rs_trig_ev` counts a resume seam (t_prev = t_start with `resume_from`):
`test_a_gate_edge_on_a_resume_seam_records_its_right_limit` (resumed row =
uninterrupted row to 1e-9).

### 2. The other cells (exact measure; b9e46778 / head = stored scoreboard / now at 62b86ff1)

| circuit | cell | b9e46778 | head | now | |
|---|---|---|---|---|---|
| back_to_back_diodes_sine | restart / split | 1.93e-16 / 1.94e-16 | 1.43e-9 / 1.64e-9 | 1.93e-16 / 1.95e-16 | ok (+0.9 % at 1e-16) |
| diode_bridge_rectifier | restart / split | 2.21e-5 / 2.38e-5 | 4.43e-5 / 4.38e-5 | = b9e | ok |
| half_wave_rectifier_sine | restart / split | 1.01e-4 / 2.56e-6 | 1.08e-4 / 1.58e-5 | = b9e | ok |
| voltage_doubler | restart / split | 1.67e-6 / 2.19e-7 | 3.02e-6 / 2.19e-6 | = b9e | ok |
| transformer_step_up | both | 4.03e-16 | 8.19e-10 | 4.03e-16 | ok |
| chopper_with_rc_snubber | restart | 2.50e-8 | 2.88e-7 | 2.57e-8 | NOT MET (+2.8 %) |
| chopper_with_rc_snubber | split | 3.11e-8 | 2.79e-7 | 3.11e-8 | ok |
| pwm_voltage_rc | both | 1.38e-8 | 4.86e-8 | 1.38e-8 | ok |
| cascaded_boost_buck | restart | 3.28e-6 | 3.32e-6 | 3.28e-6 | ok |
| cascaded_boost_buck | split | 2.37e-7 | 6.02e-7 | 3.16e-7 | metric, listed |
| flyback_converter | both | 2.55e-6 | 6.47e-6 | 2.56e-6 | ok (+0.6 %) |
| three_phase_diode_rectifier | restart / split | 9.98e-4 / 9.96e-4 | 1.27e-2 | = b9e | ok |
| vcswitch_resistor | restart | 5.92e-2 | 4.47e-2 | 3.87e-2 | ok |
| vcswitch_resistor | split | 0 | 3.16e-2 | 3.16e-2 | not this branch (below) |
| ll09_half_bridge_inverter_leg | both | 9.81e-3 | 4.14e-2 | 1.85e-7 | ok |
| ll10_full_bridge_inverter_lc | both | 1.64e-7 | 1.71e-7 | 1.64e-7 | ok (was 2.46e-6) |
| full_bridge_dc | restart / split | 3.76e-9 / 3.91e-9 | 8.02e-9 / 4.87e-9 | 3.72e-9 / 3.88e-9 | ok |
| t_type_3level_halfbridge | both | 2.10e-7 | 3.26e-7 | 2.10e-7 | ok |
| diode_half_wave_rl_load | restart / split | 9.49e-4 / 9.64e-4 | 6.71e-4 / 6.82e-4 | 9.49e-4 / 9.64e-4 | ok (split was 5.55e-2) |
| ll07_boost_vcswitch | restart / split | 1.83e-2 / 2.96e-5 | 1.30e-2 / 2.96e-5 | 1.30e-2 / 2.96e-5 | ok |

cascaded_h_bridge_5level, flying_capacitor_3level, mmc_4cell_chain (both
columns) are within +0.3 % of b9e (mantissas 6.0034/6.0031, 1.5294/1.5333,
1.1789/1.1758; same exponent).

- ll10: the right row of every dead-time edge now agrees with the truth to
  8.7e-6 of the span at worst (it was 1.8e-4: v_a −32248.82 vs −32242.25). The
  leg potentials are ALGEBRAIC (no capacitance on leg_a / leg_b): at the gate
  edge they jump from ~V_dc to ∓36 kV, the common mode (+2.16 V) set by the
  1 MΩ bleeds against the switches' g_off. So the row must NOT hold the left
  value: a leakage-only potential without charge storage is not a state. The
  stored LEFT samples inside the dead time have a common-mode error of
  ~4e-4 V (1e-8 of the value: conditioning 2L/dt against 1e-6 S); the ε rows
  at ε0 had 6.6 V (L/ε 1e3× larger). The ladder's γh/2 rung brings the row to
  the samples' conditioning.
- cascaded_boost_buck split: every right row equals its left to 1.4e-15 of the
  span (V(out) is a capacitor voltage, continuous); the largest row errors
  (1.5e-6 of span) are the stored samples' own (eR = eL). The cell rises
  because the two-sided reading counts each event row's sample twice. Listed.
- vcswitch_resistor split: the rows are the session-1 rows (identical to
  8d7b2a44's extension to 1e-22). The cell is set by an inner event at 1.5 ns
  whose LEFT (the engine's start, out = 9.9999 V with the switch ON from the
  pulse's previous period; the truth starts OFF) is a stored-sample matter,
  identical at the head (3.16e-2). Session 1's 6.7e-25 for this cell does not
  reproduce with its own extension.
- chopper_with_rc_snubber restart: NOT MET. The right row of each switch
  turn-off reads V(sw) 1e-5 V low (5e-7 of the 20 V span) with the states
  exact: rounding of the ε0 solve (L/ε against the 1e-6 S off-conductances),
  which passes the 1e-6 agreement test. A 1e-7 test would send it up the
  ladder, where the γh/2 rung's truncation on the 1 µs snubber is 3e-6 — not
  tried. Next: an ε chosen from the conditioning estimate, not a fixed ladder.

### 4. Gates (62b86ff1)

- Default path byte-identical vs the base extension (6027f4fd = cedac36b's C++,
  built with this tree's flags, LTO ON): goldens, stateful, engines, steps,
  trbdf2_identity rc 0; pwl_identity: the 218 differing entries are all
  pwl+restart / pwl+split; the default entries are identical.
- Rule-on: stored times and states identical to the base (`rlcmp.py`: 0
  entries with a non-allowed difference); differences are the right-limit /
  inner-event rows, `n_right_limit_fallbacks`, the voltage-scan peak and the
  work counters of the ε solves on 16 Newton entries.
- C++: pulsim_core_step_tests (32 cases) and pulsim_core_record_tests (18) pass
  on Eigen 5. Full ctest not run.
- Cost, instructions per accepted step (`/usr/bin/time -l`, min of six of the
  K-run total minus min of six 0-run totals, base = 6027f4fd built with this
  tree's flags; this tip includes session 1's changes): buck_pwl −0.06 %,
  vsi_pwl +0.63 % (second run +0.60 %: GATE NOT MET), buck_trbdf2 +0.35 %,
  vsi_trbdf2 −0.27 %. Rule on: buck_pwl_restart +10.5 %, vsi_pwl_restart
  +11.4 % (five ε solves per rung, up to three rungs, at each event).
  Not separated between session 1 and 2. Candidates (default path does no new
  work): new by-reference captures in per-step closures — `split` in
  run_restart, `rl_loc_t` / `rl_loc_on` in split_step, session 1's `rl_want`,
  `L` uses — each a load + store per step if those closures are built per step
  (STATE: the closure-construction work is another agent's).
- test_simulate_overhead.py 2 passed; stubs --check rc 0; ruff clean; full
  pytest 6420 passed / 114 skipped / 17 xfailed / 0 failed (at 62b86ff1).

### Fresh review (opus, on 62b86ff1) — open, NOT fixed

| id | finding | severity |
|---|---|---|
| R1 | `located` turns off re-decision for EVERY diode and forces the crossing diodes ON even when a gate edge coincides (crossing in the last snap = 1 % of the step, recorded at b, next step's mask trigger): e.g. a boundary-mode buck, D forced ON beside S ON, a short of Vin accepted by P. Also a second diode that stage 1 turned ON keeps ON with reverse current. Fix: force only `rl_loc_on`, keep re-deciding the others; do not mark located when the b-path's next step also has a gate trigger | high |
| R2 | a V_f > 0 turn-off is still `located` (vector size matches) with nothing forced: the residual through 1/g_off is constant in ε, P accepts it, the counter stays 0. At least count it | medium-high |
| R3 | ladder order 1, 1e3, 1e-3: a fast mode with τ ≲ ε0 that the probe misses is recorded relaxed (Q accepts at ε/τ ≥ 1e3) where the truth's x(t_e+) keeps its amplitude | medium |
| R4 | rung 1e-3 still ACCEPTS saturated rounding (constant in ε) when rung 1e3 fails (ll10 with a coarser dt) | medium |
| R5 | per-unknown rung mixing: a diode that commutes inside rung 1e3's range (16ε = 8γh) gives some unknowns from another mode; a capacitor between two large potentials resolved on different rungs is not exactly x0 | medium |
| R6 | stale `rl_loc_t` on a retried step; fallbacks of discarded attempts are counted | low |
| R7 | two modes of opposite sign can cancel in Q's test | remark |

## State and next step

Tip 62b86ff1 (+ this notes commit). Not mergeable as is: vsi_pwl default-path
cost +0.6 % over the 0.5 % gate (cause not separated), the reviewer's R1/R2
(unflagged wrong right limits at a located crossing that coincides with a gate
edge or a second diode; V_f > 0), R3–R5 (validity check accepts relaxed /
saturated values), and chopper_with_rc_snubber restart still +2.8 %.
Next, in order: (1) R1 — in event_right_limit force only `rl_loc_on` diodes
ON and keep `events::redecide` for the others (a redecide that skips the
forced indices), and clear `rl_loc_t` when the b-path's next step has a gate
trigger at b; (2) R2 — `located` only if some `rl_loc_on` bit is set, and
count a V_f > 0 located turn-off as a fallback; (3) R3/R4 — rung order
1, 1e-3 (only accept P there, never Q: Q's job is the relaxed regime), 1e3,
and refuse a rung whose x1..x16 are all equal to the last bit while rung 0
moved (saturation); (4) cost — build c43be8b4 with these flags and measure it
against this tip to separate session 1 from 2, then move `split` /
`rl_loc_*` out of the per-step closures (read `split` inside event_right_limit,
which is built once per run). Then re-run sb2.py, fb_count.py, ident2.sh
compare, rlcmp.py, the restart tests, instr.sh.
