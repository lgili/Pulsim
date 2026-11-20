# Split event location: F-SB-2 and F-SB-4

Branch `fix/core-split-event-location` from `origin/feat/core-integration-14t` (d9d32b88).
Findings: `origin/feat/core-scoreboard-measure` (8b12ff35),
`python/tests/test_restart_scoreboard_findings.py` and the end of `fixes/restart-scoreboard.md`.
Tests of this slice: `python/tests/test_split_event_location.py`.

Functions touched in `core/include/pulsim/solver/run_transient.hpp` (LOCATION only):
`locate_islands` (now takes `run_event_iteration`; the crossing refinement at its end),
`step_begin` (the gate sample instant `gate_t`; the split's event collection: the
opaque-gate bisection; the old `unlocated_gate_steps` block removed), the static path's
`st_begin` (`st_gate_t`). Not touched: `consistent_start`, `restart_lined`, `run_restart`,
`split_step`'s body (one call argument), the recorder, `event_right_limit` / `record_lines`.

(sections below filled in as the work proceeds)

## F-SB-4 — cause (trace on the base d9d32b88)

The split locates a diode crossing from ONE chord of its crossing signal
(`crossing_signal`, `v - V_on` while OFF): `interp_commutation_time(a, b, s(x_prev),
s(x_first))`, `x_first` = the step's first solve under the start bits (design D1). The signal
is the diode voltage `in - out`; `in` is a sine source, so `s(t)` is curved over the step,
most near the sine's peak, where turn-on happens. The chord's root is off by
`~ s''/(2 s') (t - a)(t - b)`, a fraction of h. It is the right signal, located by the wrong
rule (a single secant), at the right instant handed to the split: the split solves sub-step 1
to exactly the chord's root.

- diode_rectifier, step [24, 25] us: s(24 us) = 10 sin(0.4 pi) - out = 9.511 - 9.81 = -0.30 V,
  s(25 us) = 10 - 9.80 = +0.20 V; chord root 24.60 us (engine 24.6028 us), truth 24.3754 us.
- minimal case (`_onset` in the test file): sine 10 V 50 kHz -> D1 (1e3 / 1e-9 S) -> 100 ohm ||
  10 uF precharged to 9.8 V; truth `10 sin(w t) = 9.8 exp(-t/RC)`, t = 4.298251016 us (mpmath).
  Step [4, 5] us: s = -0.250 V / +0.249 V, chord root 4.5014 us. delta/h at dt, dt/2, dt/4 =
  +0.203, +0.078, +0.030 (the chord converges, at order ~1.4 in delta/h, with a large constant).

## F-SB-2 — cause (trace on the base)

`make_pwm_switch_fn` (the builder case and vcswitch_resistor, whose `v1_reader` mapping is a
schedule with the threshold crossing 20.0005 us as its edge) HAS an edge oracle. The split
reads it and places every edge more than `split_snap*dt` from a step end exactly; the default
`split_snap` was 0.01 (design D4), and 0.5 ns < 1 ns: the edge was snapped to the step start
(delta = -0.005 h). `control_tick='state'` sets `split_snap` to 1e-6 (G30): that is why it was
exact. Engine at h = 100 ns (delta/h, edge 0.005 h / 0.3 h / 0.995 h into a step / on a grid
point):

| config | oracle | opaque (`lambda t: fn(t)`) |
|---|---|---|
| pwl | -0.005 / -0.3 / -0.995 / -1.0 | same |
| pwl+restart | +0.995 / +0.7 / +0.005 / 0 | same |
| pwl+split | -0.005 / 0 / +0.005 / 0 | +0.995 / +0.7 / +0.005 / 0 (`unlocated_gate_steps` 2) |
| pwl+split, control_tick='state' | 0 / 0 / 0 / 0 | as pwl+split |

So two defects: the 1 % snap (oracle), and no location at all for an opaque callable (taken at
the next grid point, DECISIONS 4 of the split change). Without the split the restart rule
samples the gate at the step START (restart design D4), so an edge inside a step is taken at
the NEXT grid point, while plain pwl (sampling at the step end) takes it at the one before.

## The fixes

1. **Diode crossing (F-SB-4)**, `locate_islands`: after the chord (and the island grouping),
   the earliest crossing is refined with `events::illinois` (the one locator; TR-BDF2 uses it
   the same way) on trapezoidal TRIAL solves from the step start under the start bits:
   restore the snapshot, `accumulate_b_extra(tau, tau - a)`, `run_event_iteration(tau, tau -
   a, false, true)` (one solve, no re-decision), signal = the same `crossing_signal` (summed
   over the crossers sharing the estimate: one diode, or a floating island's group; mixed
   directions keep the chord). Stop: bracket or two successive trial instants within
   `snap`; the instant is the final bracket's unweighted chord root. The split then solves
   sub-step 1 to that instant, as before. Counter `split_locator_trials`. A failed trial
   (singular / non-finite) keeps the bracket (the chord, if it is the first).
   What is located is the crossing of the trapezoidal sub-step trajectory from x(t_prev):
   its distance to the true instant is the state's global error over the signal's slope.
2. **Oracle gate edges (F-SB-2)**: `SimulationOptions.split_snap` default 0.01 -> 1e-6
   (G30's value, which `control_tick='state'` already set; the two modes now agree). It is
   also the locator's and the bisection's resolution.
3. **Opaque switch_fn with the split (F-SB-2)**: the event collection reads the gate at t_prev
   and t (two calls per step, as before: the old block read t_prev and t_prev - snap for the
   `unlocated_gate_steps` count) and bisects each change between them to `snap`, then
   splits at the post side like an oracle edge (up to 8 per step; more -> `unlocated_gate_steps`
   and the overflow path). Counter `n_gate_events_bisected`. A pulse that opens and closes
   between two grid samples is not seen (no interior samples: they would cost a switch_fn
   call each on every step). This reverses DECISIONS 4 of add-fixed-step-event-splitting:
   a grid-indexed closure (`round(t/dt) % N`) changes value at the HALF step, and is now split
   there (it was taken at the next grid point); such a closure should be written with
   `floor` or as a schedule with `next_edge_after`.
4. **Restart rule without the split**: the gate of step (t_{k-1}, t_k] is read just before its
   end, `switch_fn(t_k - dt/100)` (was `switch_fn(t_{k-1})`). An edge inside a step now takes
   effect from t_{k-1}, the side plain pwl takes (plain pwl reads `switch_fn(t_k)`); an edge on
   a grid point t_{k-1} is still exact (design D4's order-2 case is kept); an edge in
   (t_k - dt/100, t_k) is taken at t_k (error < dt/100). Why dt/100 and not an ulp or `snap`:
   a closure with its own tolerance band (the discontinuity-order test's buck, `floor(ph + 1e-9)`)
   reads `t_k - 1e-6 dt` as "at the edge" once 1e-6 dt < 1e-9 T (N = 1600 per period): the
   on-grid edge then fired one step early (7.3e-4 A). A grid-indexed `round(t/dt)` closure
   reads the same index as plain pwl (it read one step earlier). Same on the static path
   (`st_gate_t`).

## Convergence against truth (tip)

F-SB-4, delta/h of the located turn-on (`.scratch/sel/table.py`; trials = `split_locator_trials`
of the whole run):

| case | truth | dt | dt/2 | dt/4 | before (dt, dt/2, dt/4) |
|---|---|---|---|---|---|
| minimal onset (closed form, mpmath) | 4.298251016 us | -5.4e-9 (10 trials) | -6.9e-9 (9) | -1.1e-8 (9) | +0.203, +0.078, +0.030 |
| diode_rectifier D1+ (exact_truth, mpmath) | 24.3754231 us | -1.39e-4 (17) | +1.7e-6 (18) | +4.8e-8 (16) | +0.227 at dt |

The minimal case sits at the locator's resolution (its state error at the onset is 4e-9 V: the
OFF interval is one exponential the trapezoid follows closely); diode_rectifier converges at
order ~6 in delta/h from dt to dt/4 (the state error at the onset, over the signal's slope, is
what is left). Bound justified by the locator: the located instant is within `snap` = 1e-6 h
of the crossing of the trapezoidal sub-step trajectory; its distance to the truth is the
state's global error over the signal's slope, O(h^2) / s'.

F-SB-2, delta/h at h = 100 ns (edge 0.005 h / 0.3 h / 0.995 h into a step), tip:

| config | oracle | opaque |
|---|---|---|
| pwl | -0.005 / -0.3 / -0.995 | same |
| pwl+restart | -0.005 / -0.3 / +0.005 (within dt/100 of t_k: taken there) | same |
| pwl+split | 0 / 0 / 0 | 1.1e-7 / 1.9e-7 / 8.4e-7 (bisected; `n_gate_events_bisected` 2) |

## Left and right values at the located event (what is NOT fixed here)

Minimal onset, V(out) at the located instant minus the truth 9.757967538 V (continuous there):
left -3.7e-9 / -2.4e-9 / -1.9e-9 V at dt / dt/2 / dt/4; RIGHT +0.167 / +4.9e-5 / +4.9e-5 V.
diode_rectifier D1+ at dt: left 9.808027 V (truth 9.808 V at its instant), right 9.9397 V; at
dt/2 and dt/4 right 9.80612 V. The left values are at the engine's order; the recorded RIGHT
limit after a split diode event is not the state at the event: at dt it matches the start of
the restart sub-step's quadrature line, 2 w1 X1 + (1 - 2 w1) X2 (with X1 the stage value at
t_e + gamma dt2 and X2 at t, w1 ~ 0.31 fits 9.9397), i.e. F-SB-1's mechanism (the record of a
restart step; owned by the record track, `event_right_limit` / `record_lines`). So the
limits part of diode_rectifier pwl+split stays at the right-limit record's error, not at the
smooth level, until F-SB-1 is fixed.

Also left open (design, add-fixed-step-event-splitting DECISIONS "Diode location only in a
trapezoidal interval"): a diode crossing inside an SDIRK interval (the split's remainder or the
next two window steps) is resolved per stage, on the grid. diode_rectifier's D1 turn-off
(~25.02 us, 0.65 us after the turn-on) falls in that window at every dt tried and is recorded at
25.000 us (-0.02 h at dt, -0.04 h at dt/2, -0.08 h at dt/4). Fix needed: the same locator
(`locate_islands`' refinement) in `restart_lined`'s stages, with trial solves of the SDIRK
stage (as TR-BDF2's ESDIRK trial is a re-step of the method itself), then a split of that
interval.

## Gates (tip b0523a49 + notes; default options)

- Byte compares vs the base build (`.scratch/bl.sh`): goldens, stateful 41, engines (incl. dsed),
  step sequences, trbdf2_identity 377/377, pwl_identity default entries: IDENTICAL; the only
  default-path difference is two NEW counter keys (`n_gate_events_bisected`,
  `split_locator_trials`, 0) in the counters dict.
- Moved, option-on only: golden `vsi_pwl_restart` (restart side), steps `vsi_pwl_split`,
  pwl_identity 47 pwl+restart / pwl+split entries + static `pwm_src_restart` (list in
  `.scratch/logs/bl_compare/pwlid.log`).
- reference_status (strict 'interp' gate), cells that moved: pwl+split: 30 cells, err within a
  few % except diode_half_wave_rl_load 6.2e-5 -> 5.8e-4 (agree) and vcswitch_resistor
  0.0193 -> 0.0332, agree -> DISAGREE. The latter is the gate's trace, not the engine: the
  strict gate reads stored rows + right limits only (`with_right_limits`), so an edge the split
  now places INSIDE a step (20.0005 us, exact) is a linear ramp across the step there (it was
  snapped onto the grid point 20.0, a sharp edge 0.5 ns early). The three-part measure (inner
  events included) prices it 0. pwl+restart (side change, floor instead of ceil, first order
  either way): better vcswitch_resistor 0.0546 -> 0.0050 (disagree -> agree), boost_pfc_open_loop
  0.0304 -> 0.0191 (disagree -> agree), cl_buck_pi, cl_buck_aggressive_pi, cl_boost_pi,
  ll06_buck_vcswitch; worse ll15_boost_closed_loop 0.0485 -> 0.216 (disagree both; record now
  'chatter'), ll14_buck_closed_loop 0.139 -> 0.194 (disagree both), interleaved_buck_3ph
  0.0025 -> 0.0097 (agree both). An off-grid edge's grid error is its fraction f into the step
  on plain pwl's side and 1 - f on the old side: which cells win depends on where their edges
  fall; ll15's chatter was not investigated.

## Review (fresh-context, opus) and state

Found, fixed: the opaque-gate bisection had no iteration cap; with snap < ulp(t) (t_start = 1000
s at dt = 1e-7) it never ended. Now capped at 64 halvings and stops when the midpoint stops
moving.

Found, OPEN (next step): the refinement refines only the EARLIEST chord's crossers. Two
independent diodes in one step, one convex and one concave signal (sine sources with phases,
closed-form onsets `t = (asin(E/10) - phi)/w`, dt = 1 us): D1 truth 4.75 us, D2 truth 4.20 us;
the chords say D1 first, D1 is refined to 4.75 us, and D2 is taken at its unrefined chord (+0.53
h) or, when D2's true crossing is inside sub-step 1 (solved with D2 held OFF), at D1's instant
(+0.40 h). No counter. Fix: after refining, evaluate every other crosser's signal on the trial
solve at the refined instant; any that changed sign crossed earlier: refine it on [a, t_new]
with that trial as its upper end, and repeat (at most the number of diodes). Also silent: the
locator's give-ups (failed trial, 40-iteration cap, non-bracketing summed signal, mixed
directions) keep the chord without a counter; a stateful switch_fn (one that assumes time only
moves forward) is now called at bisection midpoints that go backwards.

Also open: the diode crossing inside an SDIRK interval (above), the right-limit record (F-SB-1).

## Cost (instructions per accepted step, min over 6 interleaved rounds x 3 in-process, same flags)

Default path: buck_pwl 3133.0 -> 3131.3 (-0.05 %), vsi_pwl 4789.8 -> 4811.7 (+0.46 %, inside the
0.5 % gate, close to it). Split on (one round x 2): buck_pwl+split 3962.4 -> 4020.8 (+1.5 %),
vsi_pwl+split 5650.6 -> 5683.2 (+0.6 %).

## Hand-off state (context budget reached)

- Committed and pushed up to 124a43c8. The bisection cap (review finding B) is committed on top
  WITHOUT its verification: at hand-off `.scratch/final.sh` was rebuilding the extension, then
  running the area tests (`.scratch/logs/area.log`) and ONE full pytest (`.scratch/logs/full.log`).
  The previous build (b0523a49) passed the area set: 14 files, 213 passed, 1 xfailed.
- Scoreboard three-part grading, pwl+restart and pwl+split on the 81 circuits: base in
  `.scratch/sbr/base.json` (vcswitch_resistor pwl+split instants 0.00316, max_dt_h -0.005;
  pwl+restart instants 0.0629, +0.995; diode_rectifier pwl+split limits 0.0161, max_dt_h 0.227),
  tip running into `.scratch/sbr/tip.json` (frozen package `.scratch/tippkg` = b0523a49); compare
  with `python3 .scratch/sbr/cmp.py .scratch/sbr/base.json .scratch/sbr/tip.json`.
- Next steps, in order: (1) read full.log / area.log; (2) cmp.py and list every moved cell;
  (3) the multi-diode refinement (review finding A, above); (4) C++ targets
  `.scratch/ctest_sel.sh` (layer5_v3, step, perf, layer5) - not run; (5) stubs --check (the
  `split_snap` docstring changed in bindings.cpp) - not run.

## Three-part scoreboard, pwl+restart / pwl+split, base vs tip (b0523a49), 81 circuits

`python3 .scratch/sbr/cmp.py .scratch/sbr/base.json .scratch/sbr/tip.json`: 131 cells unchanged,
31 moved. Targets: vcswitch_resistor pwl+split instants 0.00316 -> 2e-9 (smooth, limits 0);
pwl+restart 0.0629 -> 0.00632. diode_rectifier pwl+split limits 0.0161 -> 0.0132 (NOT at the
smooth level: the right-limit record, F-SB-1, above), smooth 1.84e-4 -> 2.32e-4.
WORSE (open, not investigated): diode_half_wave_rl_load pwl+split smooth 1.35e-9 -> 5.05e-5,
limits 2.52e-5 -> 0.0281 (instants 4.8e-6 -> 5e-15) - suspect F-SB-3's collapsed 1 ns L*g_off
transient at the now exactly-located turn-off; half_wave_rectifier_sine pwl+split smooth
2.56e-6 -> 5.37e-6; interleaved_buck_3ph pwl+restart smooth 0.00249 -> 0.00966, limits
0.00528 -> 0.0198 (side change). Better: boost_pfc_open_loop pwl+restart limits 0.0613 ->
0.0387, voltage_doubler, chopper_with_rc_snubber, full_bridge_dc limits, ll06 restart instants.
