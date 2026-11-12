## Why

A run records one thing, the full MNA state at every step, and
everything a user actually reads is rebuilt afterwards, per engine and
per device kind, in Python. Counted on this branch
(PY = `python/pulsim/__init__.py`, DD = `python/pulsim/_dsed_dispatch.py`,
`core/…` = `core/include/pulsim/…`):

- **Device currents are reconstructed post hoc**, by a kind ladder in
  `_result_i` (PY:861-1188, 328 lines). It restates the Shockley law
  from a Python side table, differentiates stored samples for
  capacitors (`C·np.gradient(v_C)`, PY:1087) and raises
  `NotImplementedError` for the kinds it does not know (PY:1129, 1158,
  1177, 1219), telling the user to add a 1 mΩ series resistor.
- **Two result types.** `SimulationResult` (C++, read-only properties,
  not constructible from Python) and `_PEDSimulationResult` (DD:330), a
  duck-typed mimic that exists because the first cannot be filled. Only
  the mimic carries right-limit samples at events (DD:357-368); the
  fixed and variable-step results do not.
- **Nineteen assignments of ad-hoc attributes onto results** in PY
  (`_builder`, `_preflight`, `_switch_fn`, `engine_used`,
  `engine_route_reason`, `_implausible_voltage`, `_motor_traces`,
  `_trbdf2_stats`), repeated per engine branch
  (`grep -cE "^\s+(res|result|_partial)\._?[a-z_]+ = "`).
- **The result does not know its grid.** `times` is a uniform grid on
  the fixed engine and the accepted (irregular) grid on the other two;
  nothing in the type says which. Time-weighted measures exist
  (`python/pulsim/stats.py:140, 175`, `_result_views.py:163`) but are
  not methods of the result, so the first thing a user types is
  `.mean()`.
- **Recording is all-or-nothing on a uniform stride.** `store_every` is
  refused on the variable-step engine (PY:3177-3289); there is no
  `t_eval`, no probe selection, no dense output. The drivers push
  samples at four sites in `solver/run_transient.hpp`, three in
  `solver/trbdf2_transient.hpp` and two in `solver/bdf1.hpp`.

Measured consequences:

| Symptom | Cause | Source |
|---|---|---|
| A whole "TR-BDF2 turn-off defect" was invented by a sample mean: 3.04 V against 6.33 V; time-weighted, six runs agree to 0.3 % | no grid kind, no measures on the result | owner note *time-weight-averages-on-adaptive-grids*, PR #140 |
| A DCM boost diode's average current read 0.0234 A against 0.0535 A | trapezoid from the left limit at an event; right limits exist on one engine only | DD:362-367 |
| `res.i()` raises for `mosfet_level1`, `igbt_level1`, `lauritzen_diode`, `nonlinear_diode`, `nonlinear_capacitor`; the loss summary lists none | post-hoc ladder | audit `raw/models.md` finding 5, probe `models/p6_misc.py` |
| `res.i('D1')` costs 43 ms on first call at 100k samples; `res.times` 7.5 ms per access | Python reconstruction; list conversion | audit `raw/perf.md` row 27 |
| A 1 s run of a 12-node inverter at 100 ns stores 1.2·10⁸ reals to read three currents | no probe selection | inferred from the record layout |

## What Changes

- **Probes.** A probe names one scalar quantity: a node voltage, a
  voltage difference, a device pin current, a device power, a device
  output channel, or a raw unknown. Probes are declared on the circuit
  (builder `add_probe`, YAML `probes:`) or on the run, compiled once
  into a flat evaluation plan, and evaluated in the kernel at commit
  time from the committed step (`x`, device state, mode bits, step
  context). A device current is the current the solve used
  (`f + d/dt q`, the mode law, the companion law), never a derivative
  of stored samples.
- **Recorders.** `RecordOptions{what, when, events}`:
  `what` = all unknowns (today's behaviour, the default) or a probe
  selection; `when` = every step, every n-th step, a list of times
  (`t_eval`), or a period; `events` = keep left- and right-limit rows
  at every discontinuity. Sinks: the in-memory table, and the existing
  lock-free `LiveRing`.
- **One result type** for every engine, constructible from C++ and
  Python, that knows its grid (`Uniform`, `Accepted`, `Requested`),
  carries the event rows, the mode record, typed run statistics, the
  diagnostics, the plan and the final snapshot as fields.
- **Measures on the result:** `avg`, `rms`, `integral`, `min`, `max`,
  `at(t)`: time-weighted, starting each interval from the right limit
  where an event fired. The existing `stats.py` functions become thin
  callers.
- **BREAKING:** the DSED result is a `SimulationResult`;
  `_PEDSimulationResult` is deleted. Its extra attributes
  (`n_accept`, `n_events`, …) remain as properties over the typed
  statistics.
- **BREAKING:** `result._trbdf2_stats` (a dict) becomes `result.stats`
  (typed); the private attribute stays as an alias for one release.
- **BREAKING:** `res.i(name)` for a capacitor returns the companion
  current the solver used instead of `C·np.gradient(v)` (see the
  design's Open Questions; the previous form remains available as an
  explicit derived measure).
- **BREAKING (improvement):** `res.i(name)` no longer raises
  `NotImplementedError` for a supported kind; where a quantity was not
  recorded and cannot be recomputed exactly from what was, the error is
  an `UnsupportedError` naming the probe to add.
- **Deleted at the end:** the ladder at PY:861-1188, the mimic at
  DD:330-476, the per-branch attribute assignments, and the fixed
  per-device current table that `add-unified-device-interface` records
  in the interim.

## Impact

- Affected specs: `simulation-results` (new).
- Affected code: new `core/include/pulsim/record/`;
  `solver/{result,state_trajectory}.hpp`; the nine result-push sites of
  the drivers (or the one `hooks.on_step` of the unified stepper);
  `streaming/live_ring.hpp` (sink adapter); `builder/circuit_builder.hpp`
  and `yaml/loader.hpp` (probe declarations); `python/bindings.cpp`,
  `python/dsed_bindings.cpp`;
  `python/pulsim/{__init__,_dsed_dispatch,_result_views,_diode_state,stats,losses,thermal,stream,scope}.py`;
  stubs; `AI_GUIDE.md`; docs.
- Depends on: `add-unified-device-interface` (generic pin currents,
  power and `outputs()`; its migration step 9).
- Uses, when present: `add-unified-event-queue` (`OutputSample` time
  events for `t_eval`; the left / right-limit order at an instant),
  `refactor-unified-transient-stepper` (one hook instead of nine push
  sites), `add-kernel-simulation-object` (`RecordOptions` inside
  `RunOptions`; diagnostics and plan as result fields).
- Unblocks: `add-in-kernel-event-losses` (event-energy probes and a
  thermal-node sink), `add-compiled-control-graph` (signal probes),
  `add-periodic-steady-state-shooting` (cycle measures).
- Performance constraints: recording stays allocation-free per step
  (pre-reserved contiguous buffer; chunked growth only on the
  irregular grid, amortised, outside the per-step path); probe
  evaluation is a flat plan with no per-probe indirection; with the
  default `what = all` the per-step recording cost stays at today's
  (about 35 ns/step, audit row 20) and the stored bytes are equal; with
  a probe selection the stored bytes are `n_probes` reals per sample.
- Results: with default options, `times`, `states`, `switch_states`,
  `v()`, and `i()` of linear and PWL kinds are unchanged
  (bit-identical on the fixed engine) at every migration step.
