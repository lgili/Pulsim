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

## State and next step

Open: D4 (no test), D6, D7; diode_half_wave_rl_load split (located turn-off
read through g_off), ll10 (2.46e-6, conditioning), cascaded_boost_buck split,
chopper restart (+2.8 %). Next step for the turn-off: at a located diode
crossing, take the ε solves in the mode where the commutating diode is ON
(i = 0 there; the OFF mode amplifies the location residual by 1/g_off).
