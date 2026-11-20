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
