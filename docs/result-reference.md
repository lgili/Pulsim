# Result reference — `SimulationResult`

Every engine returns the same class: `pulsim.simulate(...)` on `pwl`,
`trbdf2` and `exact` (dsed) gives a `pulsim.SimulationResult`
(`isinstance(res, pulsim.SimulationResult)` holds on all three). The exact
engine's former `_PEDSimulationResult` is gone (**breaking** for code that
tested the class name; see the CHANGELOG).

## Rows and the grid they sit on

| Member | What it is |
|---|---|
| `times` | sample times, 1-D float array |
| `states` | read-only `(num_steps, state_size)` view, zero-copy |
| `num_steps()`, `empty()` | row count; `True` when nothing was recorded |
| `grid` | `GridKind.Uniform` (pwl), `Accepted` (trbdf2, exact: the accepted steps, packed at edges) or `Requested` |
| `switch_states`, `switch_states_unsolved` | per-row switch / diode mask the solve stamped; rows not produced by a solve under it |
| `engine_used`, `engine_route_reason` | the engine that ran and why `engine='auto'` chose it |

On an `Accepted` grid a plain `.mean()` weights samples, not seconds. Use
the time-weighted readers below or [`time_average`](api-reference.md#statistics-over-a-result-time-weighted).

## Reading a probe

A probe is a text such as `"v(out)"`, `"v(a,b)"`, `"i(L1)"`, `"p(R1)"`
or an expression of them (`"v(out)*i(L1)"`).

| Method | Returns |
|---|---|
| `v(name, t=None)`, `i(name, t=None, port=None)` | node voltage / branch current per row (or at row index / slice `t`) |
| `at(probe, t)` | value at time(s) `t`: linear on the step that holds it, the right limit at an event where the engine stored one |
| `avg(probe, t0=None, t1=None)` | time average over `[t0, t1]`, window clipped exactly |
| `rms(probe, t0, t1)`, `integral(probe, t0, t1)`, `minmax(probe, t0, t1)` | the same window, exact for the linear dense output |
| `currents()`, `power(device)`, `signal(name)`, `signals()` | all branch currents; average device power; observer traces |
| `loss_events(device)`, `device_energy(device)` | per-event loss records; energy split per device |
| `flux(name)`, `b_field(name)`, `h_field(name)` | magnetic network quantities |

## Run statistics

`res.stats` is a `RunStats`: engine-independent counters, zero where an
engine does not report one — `engine`, `n_steps`, `n_accept`, `n_reject`,
`n_solves`, `n_factorizations`, `n_newton_iterations`,
`n_newton_retries`, `n_forced_accepts`, `n_mode_events`,
`n_state_events`, `n_gate_events`, `n_ctrl_ticks`, `n_chatter_breaks`,
`n_exact_steps`, `n_rk45_steps`, `n_bdf2_steps`, `n_gate_undersampled`,
`gate_pulse_min`, `gate_sample_spacing_max`, `cpu_time_seconds`.

The exact engine's old attributes stay as read-only aliases of
`res.stats`: `n_accept`, `n_reject`, `n_events` (= `n_mode_events`),
`n_rk45_steps`, `n_bdf2_steps`, `n_exact_steps`, `cpu_time_seconds`.

Engine-specific diagnostics: `newton` (`NewtonCounters`), `counters`
(pwl loop counters), `commutation_events`, `dt_retries`,
`event_iteration_breaches`, `reinit_records`, `resolution` (pwl with
`resolution_diagnostics`), `mode_diagnostics` / `diagnostics`,
`n_restart_steps`, `n_restart_factorizations`, `newton_roundoff_*`,
`control_tick_counts`, `control_tick_jitter_max`,
`step_loop_python_callbacks`, `final_snapshot` (for
`simulate(resume_from=...)`), `states_bytes` / `total_bytes`.

## Snapshots: `final_snapshot` and `resume_from`

`result.final_snapshot` is the solver's state at the run's last time.
Passing it as `simulate(..., resume_from=snap)` continues from there, and
it can be pickled. Besides the circuit state it can carry the step
controller's memory, which `snap.controller_state` shows as
`(kind, values)` or `None`:

| kind | written by | what a resume does with it |
|---|---|---|
| 2, LTE | `engine='trbdf2'` | the next step is the step the stopped run would have taken |
| 3, exact | `engine='exact'` | `engine='dsed'` resumes from it; the other engines refuse it |
| `None` | `engine='pwl'`, or a snapshot without the block | `trbdf2` restarts its controller; `pwl` steps on its `dt` |

What to expect from a split run (tests in
`python/tests/test_trbdf2_resume_controller.py`):

- **Bit-identical** on `trbdf2` when the controller block is the run's
  whole loop state, for example an RLC with no switches or diodes, split
  at one of the continuous run's step times.
- **Not bit-identical** on a switched circuit. The edge-learning and
  hold state of the TR-BDF2 loop is not in the snapshot, so the next gate
  edge can land in a different 14th digit. The resumed run still agrees
  to tolerance.
- **Across engines.** A `pwl` snapshot resumes on `trbdf2` and the
  reverse, to the engines' tolerance.
- A run with a control graph refuses to resume between two of its ticks.

## Recorders: `recorded` and `recorder_feed`

With `simulate(recorders=[pulsim.Recorder(...)])`:

- `res.recorded[name]` is a `pulsim.Recording`: `probes`, `times`,
  `values[probe]` (or `rec[probe]`), `interpolated`, `n_samples`,
  `windows[probe][field]` and `periods[probe][field]` (fields `integral`,
  `avg`, `rms`, `min`, `t_min`, `max`, `t_max`, `t0`, `t1`), the event rows
  `event_t`, `event_left[probe]`, `event_right[probe]`, `event_t_right`,
  `event_resolution` (`'exact'`, `'settled'`, `'within_dt'`), and
  `feeding`.
- `res.recorder_feed` says which feed ran: `mode` (`'in-loop'` or
  `'stored-arrays'`); in the loop also `n_steps`, `n_events`, `n_jumps`,
  `n_dropped`; on the stored-array pass `why`; after
  `simulate(resume_from=...)` a `resume` note when the record did not
  continue from the snapshot.

The arithmetic, window rules and feeding table are in
[Kernel recorders](api-reference.md#kernel-recorders-simulaterecorders).

`Recorder(stream=pulsim.NativeLiveStream(...))` pushes kept rows into
the stream's ring as `(t, one value per probe)` — every `decimate`-th one
(the stream's own setting, default 100; pass `decimate=1` for all); in the
loop (pwl, trbdf2) while the engine runs, on the stored-array pass after
it. Each run re-attaches the stream (a new ring, read from its start); a
run that falls back from the loop to the stored pass keeps the rows the
loop pushed. Give each recorder its own stream.

**A run that throws.** The recorder's feed is detached from the
Simulation object when `simulate()` returns or raises. A later run on the
same object starts with no feed attached and records nothing from the
failed run: call `simulate(..., recorders=[...])` again with fresh
`Recorder` objects (or the same ones: a `Recorder` holds no run state;
its stream, if any, is re-attached).
