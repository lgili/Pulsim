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
2e-12 relative. A source of another kind (sine, PWM, pulse) is
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
