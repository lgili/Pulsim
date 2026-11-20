# `t_start` on the exact engine read the drive on another time base

Branch `fix/core-tstart-time-base`, from `fix/core-exact-stateless-loss-port`
(3b6f2a34). Python only. Found by the exact-stateless reviewer (side finding 1
of `exact-stateless.md`).

## Defect

`simulate(engine='dsed' | 'exact', t_start=t0)` runs the engine on its own
clock from 0 and relabels its outputs by `t0`, but handed the drive the
engine's clock: the gate and its edge oracle ran `t0` early, and the start
mask was read at absolute `t0` (`sf(opts.t_start)`), so a schedule with an
oracle ran on a MIX of the two bases. pwl / trbdf2 read every input at
absolute `t`. The leg of `exact-stateless.md` with `t_start = 5e-5`
(HS scheduled off there): dsed booked HS conducting (E_cond 1.7996e-4 J,
v(a) 599.94 V), trbdf2 HS off (1.8e-8 J). After: dsed 1.7999999999964e-8 J,
trbdf2 1.79999e-8 J, v(a) 6e-10 V on both.

## The rule (docs/internals/run-planning.md, "Time")

`t_start` is an ABSOLUTE label: every time-dependent object is evaluated at
absolute `t` on every engine, so a run on `[t_start, t_end]` from the state at
`t_start` equals the tail of the run on `[0, t_end]`. An engine whose clock
runs from 0 shifts every input it hands a time to in one place — where the
plan hands it the drive (`_run_dsed` → `_dsed_dispatch.on_absolute_clock`) —
and every output back (`_result_from_native`; the loss events in the binding).

## Table, measured (`.scratch/tstart_probe.py`, now `test_tstart_time_base.py`)

Each input on a stateless circuit (the output at t is the input at t), run on
[3e-5, 2e-4]: ABS = reads f(t), SHIFTED = reads f(t − t_start), MIXED = neither
(the start mask at absolute t, the edges on the engine's clock).

| item | pwl | trbdf2 / esdirk43 / auto | dsed, exact (rk45, auto integrator) | dsed bdf2 | after (exact engine) |
|---|---|---|---|---|---|
| SwitchSchedule (native edge oracle) | ABS | ABS | MIXED | SHIFTED | ABS |
| Python gate, no oracle (poll + bisection) | ABS | ABS | SHIFTED | SHIFTED | ABS |
| Python gate with `next_edge_after` | ABS | ABS | MIXED | SHIFTED | ABS |
| `b_extra_fn` | ABS | ABS | refused at any t_start (no exact step) | SHIFTED | ABS (bdf2); refusal unchanged |
| sine / pulse / PWM / PWL sources (the circuit's own) | ABS | ABS | SHIFTED | SHIFTED | REFUSED by name (`feature='t_start'`, device named) |
| control ticks (controller_period, closed loops, graphs, monitors) | callable gets absolute t; grid origin t_start (documented contract) | same | refused by name | refused by name | unchanged (refused) |
| sensitivities run (exact path) | — | — | gate SHIFTED (65 % of samples wrong) | — | ABS |
| `periodic_steady_state` | has no t_start | | | | n/a |
| `resume_from` | starts at snap.t; an explicit t_start wins (the caller's relabel) | same | absolute clock; another t_start refused | no snapshot | unchanged |

Outputs of a fresh exact-engine `t_start` run:

| output | before | after |
|---|---|---|
| `times`, `event_times`, `output_grid.output_budget_t` | relabelled | relabelled |
| loss events (`loss_events`, `t_offset` in the binding) | relabelled | relabelled |
| `reinit_records[i].t` | ENGINE clock (7e-5 edge read 4e-5) | absolute |
| mode diagnostics (`mode_shadow` / `mode_selection` / `mode_guard` disagreement `t`, `repair_times`) | engine clock | absolute |
| recorders (stored-array pass on the relabelled times; the in-loop feed is off for a fresh t_start run) | absolute | absolute (windows and periods checked) |
| `final_snapshot` | invalid, by name | unchanged |
| kernel error messages ("at t = … s") | engine clock | engine clock (C++ text; not fixed) |

## What needs C++

The circuit's own time-varying sources are evaluated by the native adapter
(`build_time_varying_b_extra_(t)`, the source corners, the closed forms of
the exact segments) on the engine's clock: no Python wrapper can move them.
A fresh `t_start` run with one is refused by name. The fix: the scheduler's
clock starts at `t0` (`simulate(x0, t_end, t0)` in the three schedulers, as a
resume already does), after which the Python shift and the relabelling are
deleted. Controllers with `t_start` on the exact engine stay refused for the
same reason (a control graph's `ChainTick` is a C++ callable).

## Property test (`test_tstart_time_base.py`)

Three circuits (600 V leg, R-L load, loss port on HS; 10 kHz sine-fed diode
rectifier; PWM buck whose gate a P control tick sets each period) × pwl
(dt 1e-7), trbdf2 / esdirk43 / auto (rtol 1e-8), dsed: run [0, T] uninterrupted,
[0, T_MID] for the snapshot, then (a) resume and (b) a fresh run at
t_start = T_MID from the snapshot's state. Checked: the end state, the loss
energies and counts (resume = whole run, fresh = whole − first), the loss
events' instants (1e-12 s) and energies, the recorder's window and period
statistics over absolute windows.

- dsed: resume bit-identical; fresh to 1e-10 of scale (measured 5e-13 A on
  60 A); rectifier fresh refused by name (Vs), buck fresh refused by name
  (controllers).
- pwl: resume bit-identical (leg, rectifier); fresh within 2e-3 of scale
  (measured 1e-5 state, 3.1e-4 on the first loss event: the fresh start has
  no trapezoid history).
- trbdf2 / esdirk43 / auto: within 1e-5 (measured ≤ 3e-7); recorder
  statistics within 1e-3 (quadrature over each run's own samples: esdirk43's
  rectifier rms 2.6e-5).
- dsed `integrator='auto'` / `'bdf2'` (a leg with no port and no diode
  reaches them): fresh = tail to 1e-10 / 1e-6.

Without the fix 40 cells of the file fail (patched back in a plugin).

## Side findings (not fixed here)

1. pwl resumed EXACTLY on a control-tick instant is not bit-identical (the
   buck: 7e-3 V of 24 V at the end): the uninterrupted run's tick at label t
   reads x(t − dt) (v(out) 17.9559 at t = 4e-4 against x(4e-4) = 17.9639),
   the resumed run's tick reads the snapshot's x(t). Same under
   `control_tick='state'`. Off the tick grid the resume is bit-identical.
2. (from exact-stateless) edge counting at the run's first instant still
   differs: the leg from t_start = 5e-5 (an HS edge): trbdf2 n_on = 1, dsed 0.
