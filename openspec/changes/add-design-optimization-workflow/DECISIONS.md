# Decisions taken while implementing the first slice

Slice = tasks 1–6 (groups 7 and the later changes are not in it). The
owner was away; the design's open questions were delegated. Each entry:
question → option → evidence → how to reverse. Numbers are measured on
the CCM buck of task 1.3 (24 V, 100 µH, 47 µF, 1 Ω, 10 mΩ inductor
resistance, `g_on = 1e3`, `g_off = 1e-9`, 100 kHz, D = 0.5) unless
stated.

## 1. How the KPIs integrate one period

→ Piecewise-linear, time-weighted, on the run's own grid, with the
right limit of every event inserted as a repeated instant
(`pulsim.time_average` / `time_window`, `device_loss_summary`). On the
`pss` route the KPI period is a separate `dsed` run from the orbit's
`x0` with the step capped at `T / samples_per_period`; default
`samples_per_period = 2000`.

Evidence (relative error against the matrix-exponential reference,
`python/tests/_design_reference.py`):

| samples / period | efficiency | total loss | ripple (sample max) | ripple (vertex) | evaluate |
|---|---|---|---|---|---|
| 100 | 8.5e-10 | −7.6e-11 | −1.8e-4 | 3.4e-7 | 5 ms |
| 500 | 3.4e-11 | −3.0e-12 | −7.5e-6 | −2.8e-9 | 4 ms |
| 2000 | 1.9e-12 | −8.2e-15 | −5.4e-7 | −6.0e-11 | 4–7 ms |
| 10000 | −1.8e-12 | −3.9e-13 | −5.2e-9 | −6.5e-10 | 9–12 ms |

Without the right limits a trapezoid over the gate edge smears the
switch current's jump over one step: the input power was 1 % high at
100 samples per period and 0.1 % at 1000 (142.58 W against 142.43 W),
first order in the step.

→ Reverse: change the default; the KPIs stay time-weighted.

## 2. Ripple extrema

→ A peak-to-peak is the difference of two extrema; an extremum that is a
smooth interior maximum of the samples (three distinct instants around
it, no event) is replaced by the vertex of the parabola through those
three samples; an extremum at an event (a kink: the inductor current at
a gate edge) is the sample itself. The sampled maximum is off by
O(h²) (−5.4e-7 of the ripple at 2000 samples, table above), the vertex
by O(h³) (−6.0e-11). → Reverse: `_p2p` in `design.py`.

## 3. Loss categories

→ A switch's or diode's resistor-pair dissipation (`P_avg` of
`device_loss_summary`) is `conduction`; when `loss_options` carries
`conduction_specs`, the datasheet model `P_cond_model_avg` replaces it
(it is "a separate estimate, not an addition", `losses.py`). A
switch's `P_sw_avg` is `switching`, a diode's is `recovery`; an
inductor's `P_core_avg` is `core`. A resistor that is not a load is
`resistive` — not `copper` or `esr` as design D1 lists: the evaluator
cannot know whether a resistor is a winding, an ESR or a snubber.
→ Reverse: a user map `{device: category}` in `Evaluator`.

## 4. Efficiency

→ `P_out` = the dissipation of the devices named in `loads`;
`P_in` = `−Σ V · mean(i)` over the DC voltage sources (the MNA branch
current flows + → − inside the source), measured on the source, not
inferred from the balance. On the oracle `P_in − P_out − P_loss` is
2e-12 relative. The efficiency KPI is `P_out / (P_out + P_loss)`
(decision 11 says why). A source of another kind (sine, PWM, pulse) is
**refused** (`UnsupportedError(engine='design.evaluate',
feature='sources', device=name)`), never guessed; name the DC sources
in `sources=`. → Reverse: integrate `v(t)·i(t)` for waveform sources.

## 5. What is `failed`

→ (a) A `RuntimeError` raised by an engine call — the kernel's
non-convergence (`did not settle`, chattering, step floor,
`SimulationAborted`) is a `RuntimeError`; a refusal is an
`UnsupportedError`, a `ValueError`, so it is never caught here;
cancellation (`pulsim.Cancelled`) is re-raised. Only the engine call is
wrapped, never `build` or the KPI code, so a bug there propagates.
(b) A `ValueError` of `periodic_steady_state` whose message is one of
its non-convergence diagnostics (no orbit within the budget, no
progress, singular `I − M`, a non-isolated orbit, a grazing event, a
failed period run, an irreproducible map). Its other `ValueError`s are
configuration errors (a schedule that is not periodic in `period`) and
propagate. (c) A non-finite KPI (`P_in = 0`). (d) `not_settled`.

The message match of (b) is fragile; the kernel is missing a typed
non-convergence (see "Missing from the kernel" below).
→ Reverse: catch the typed exception once it exists.

## 6. `build` and `period`

→ `build(design)` returns a builder or `(builder, switch_fn)`: the
steady-state solver and every engine need the schedule beside the
circuit, and the schedule is a design variable (duty, frequency).
`period` is a float or `period(design) -> T` (the worked example sweeps
`f_sw`). → Reverse: a `Circuit` object carrying both once
`add-immutable-circuit-ir` lands (D7).

## 7. Adaptors

→ `pulsim.design.as_objectives(record, names, *, maximise=(),
penalty=1e30)` and `as_constraints(record, names, *, penalty=1e30)`,
module functions (D3 says `pulsim.design.adapters`; a namespace of two
functions earns nothing). `maximise` negates, so every component is
minimised. `as_constraints` raises `KeyError` on a name the record has
no violation for — a misspelt constraint must not read as feasible.
→ Reverse: re-export them under an `adapters` namespace.

## 8. User KPIs

→ The default KPIs are always computed; `kpis=` adds
`Kpi(name, fn(period))` where `period` is a `SteadyPeriod` (result,
builder, design, window, seed) with time-weighted `average` / `rms`
helpers. → Reverse: let `kpis=` replace the defaults.

## 9. Settle criterion of `method='transient'` (open question of the design)

→ The run doubles from `min_periods = 16` (re-run from `t = 0`: at most
2× the cost of the final run, and deterministic) until the estimated
distance to the engine's own orbit is below `tol`, relative to each
state's magnitude over the last period (inductor currents, capacitor
voltages); `max_periods = 4096`, then `failed`, `not_settled`.
Default `tol = 1e-6`. The estimate is the larger of two geometric
extrapolations per state: over the last period (`d1·ρ/(1 − ρ)`) and
over the last half (`d_a·r/(1 − r)`, quarters). The state is read at
`kT` when the engine landed every period start, else as the mean over
the period ending there (interpolating across an edge that falls
between samples read 1e-4 of the state).

Evidence, DCM buck from rest (multiplier 0.989), error of `v_C` at
the section against the 50-digit orbit, relative to 9 V:

| engine | tol | periods | estimate | true error |
|---|---|---|---|---|
| dsed | 1e-4 | 1024 | 1.16e-5 | 1.15e-5 |
| dsed | 1e-6 | 2048 | 1.4e-10 | 1.4e-10 |
| trbdf2 | 1e-4 | 1024 | 1.15e-5 | 3.6e-5 |
| trbdf2 | 1e-6 | 2048 | 1.4e-10 | 2.2e-5 |

The trbdf2 rows show the second limit: `tol` bounds the distance to
the ENGINE's orbit, and trbdf2's own orbit at default tolerances is
2.2e-5 from the exact one. A first version took only the halves'
estimate and stopped at 32 periods with 6e-3 for a state 51 % off
(start-up overshoot: the halves contract fast, each period still moves
1.1 %); the test keeps that case. A first version also took the max
over states of each difference separately, mixing the inductor
current's start-up drop into the capacitor's ratio. A PWM voltage
source on trbdf2 (its edges not landed) never settles to 1e-6: the
period-to-period noise is 2.7e-4 of the state — reported as
`not_settled`, which is that engine's accuracy on that circuit.
→ Reverse: `SteadySpec(tol=..., min_periods=..., max_periods=...)`.

## 10. `auto`: the capability check

→ `SteadySpec(method='auto')` constructs the steady-state solver's
applicability checks (`periodic_steady_state._Solver`: circuit and
schedule) before running anything; an `UnsupportedError` there routes
to the transient, recorded in `counters['steady_method']` and
`counters['steady_route_reason']`. A failure AFTER the check (no orbit
within the budget) is `failed`; it never falls back. An evaluator with
`engine='pwl'` or `'trbdf2'` routes `auto` to the transient (the
steady-state solver runs on `dsed` only); `method='pss'` with those
engines is rejected in `__init__`. → Reverse: a public capability
query on `periodic_steady_state` (kernel list, item 2).

## 11. Losses the evaluator cannot account for

→ When `loads` are named, a device whose power is in neither
`device_loss_summary` nor the measured sources (anything but R,
switches, PWL and Shockley diodes, L, C and the DC voltage sources) is
refused (`UnsupportedError('design.evaluate', 'losses', name,
kind)`). Evidence: a buck with a Lauritzen diode read efficiency
0.99989 with the diode's 1.36 W of the 28.4 W input missing. Without
`loads` the other KPIs are returned, no `p_loss`, and
`counters['losses_not_covered']` names the devices. Efficiency is
`P_out / (P_out + P_loss)` (decision 4 kept `P_in` as a measured KPI):
with a datasheet conduction model or a loss table the losses include
energy the resistor-pair circuit does not dissipate, so `P_in` would
not contain it; without them the two agree (2e-12 on the oracle).
→ Reverse: extend `device_loss_summary` to the device (in-kernel
losses).

## 12. Batch scaling (open question of the design)

→ `evaluate_many` defaults to the thread backend, `backend='process'`
accepted; the first three designs run serially (the stop rule), the
rest on the pool. Measured (64 DCM-buck designs, `pss` route, shared
16 GB machine, observation, not asserted): 1 worker 0.50 s, 2 → 0.45 s,
4 → 0.43 s, 8 → 0.47 s. A design costs about 7 ms, of which the
kernel is 13 % (cProfile: `run_ped_native_builder` 27 ms of 210 ms for
32 designs); the Newton shooting and the KPI post-processing are
Python and hold the GIL, so threads stop scaling at 2 workers. 256
designs on processes: serial 1.83 s, 2 → 1.83 s, 4 → 1.43 s, 8 →
1.52 s (spawn cost against 7 ms jobs). Equal records on every backend
(tested). → Reverse: nothing to reverse; the lever is in the kernel
(item 3).

## Missing from the kernel (recorded, worked around in Python)

1. **A typed non-convergence.** What: a `RuntimeError` subclass, e.g.
   `pulsim.ConvergenceError(where, reason, t, counters)`, raised by
   the transient engines (Newton, event-round cap, step floor —
   `SimulationAborted` covers part of it) and by
   `periodic_steady_state` (today a plain `ValueError`). Why: the
   evaluator classifies `periodic_steady_state` failures by message
   (decision 5b). Smallest API: the class, raised at the existing
   raise sites; `periodic_steady_state` would raise it with the
   residual history it already formats into the message.
2. **A capability query for the steady-state solver.** What:
   `periodic_steady_state.check(builder, period, switch_fn)` raising
   the same `UnsupportedError` without building the solver. Why: the
   `auto` route reaches into the private `_Solver` constructor
   (decision 10). Smallest API: the two checks it already runs
   (`_check_circuit`, `_check_schedule`) behind one public name.
3. **The period map and the period integrals without the GIL.** What:
   the shooting Newton (monodromy products, saltation) and the
   one-period integrals of `v·i`, `i²` with right limits, in C++ behind
   one call. Why: 87 % of an evaluation is Python, so the thread
   backend does not scale (decision 12). Smallest API: a native
   `period_integrals(result, pairs)` first (the KPI half), the map
   second (`add-periodic-steady-state-shooting` group 6.2 shares it).
4. **Losses of the stateful and nonlinear devices.** What: per-device
   dissipated power for the Lauritzen diode, the level-1 MOSFET and
   IGBT, and the controlled sources' delivered power. Why: they are
   refused for efficiency today (decision 11). Smallest API: their
   `P_avg` in `device_loss_summary` (the in-kernel event losses change
   brings it for the devices it covers).
