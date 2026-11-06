# Losses & junction temperature

Pulsim ships a **PSIM-equivalent loss panel** that walks every relevant
branch of a simulation result and returns per-device dissipation
metrics, plus a Foster-network helper that pipes those losses through a
datasheet thermal impedance to produce `T_j(t)`. The whole pipeline is
post-hoc — you run the electrical simulation once with the kernel, then
call two pure-Python helpers on the result.

> **TL;DR**
>
> ```python
> res = p.simulate(b, t_end=..., dt=..., switch_fn=pwm)
> loss = p.device_loss_summary(b, res, switch_fn=pwm,
>                                switch_specs={...},
>                                diode_specs={...},
>                                core_loss_specs={...})
> therm = p.device_thermal_summary(b, res,
>                                    thermal_specs={
>                                      "M1": {"stages": [FosterStage(...), ...],
>                                             "T_ambient_C": 40.0},
>                                    },
>                                    # same loss kwargs forwarded:
>                                    switch_fn=pwm,
>                                    switch_specs={...},
>                                    diode_specs={...})
> ```

## What you get per device

`device_loss_summary` returns a list with one dict per branch it
recognises. Common fields:

| Field | Type | Where it appears |
|---|---|---|
| `branch_id`, `kind`, `name` | int / str / str | every device |
| `i_avg`, `i_rms`, `i_peak` | float | every device |
| `P_avg`, `E_total` | float | resistor / diode / switch |
| `duty_conducting` | float ∈ [0, 1] | diode |
| `duty_closed` | float ∈ [0, 1] | switch |

Switch and diode states are not inferred. Every engine records, with
each stored sample, the switch / diode mask it **stamped** for the solve
that produced it (`result.switch_states`, one bool column per
`builder.switch_index_of(name)`), so a switched branch's current is
exactly `(g_on if closed else g_off) · v`. A PWL diode needs this: its ON
state is a pure conductance, so a diode carrying 48 A at `g_on = 1e3`
sits at 48 mV — inside `0 < v_D < V_th`, where the voltage alone cannot
say whether it conducts. `switch_fn=` is only needed for a result built
without the record (it is then re-evaluated, with a warning). Duties are
time-weighted per step, so a variable-step grid does not bias them.

Row `k` of the record is the mask in force over the step
`(t[k-1], t[k]]`, and the sample's state is the solve under it. At a
commutation the stored sample is therefore the state just BEFORE the
switch flips (a TR-BDF2 landing, a DSED event), and the new mode appears
from the next sample on. A sample 0 seeded with `initial_state` was not
solved in the run (`result.switch_states_unsolved`): its recorded bits
are the mask carried in, and the reports read it with the first solve's
mask. A `resume_from` sample 0 carries the bits of the solve that
produced it and is not flagged. The raw `result.switch_states` keeps the
carried bits on such a row; check `switch_states_unsolved` before reading
it directly.

Time integrals (`i_avg`, `i_rms`, `P_avg`, `E_total`, the conduction
model, the forward charge of a recovery) integrate each step as the line
from its START to its end sample, exactly: `h·(a + c)/2` for a current,
`h·(a² + a·c + c²)/3` for a square. The start of a step is the sample
before it, except at a DSED event: the stored sample is the left limit
there, and `engine='dsed'` also stores the right limit the next step
starts from (`result.right_limit_index` / `right_limit_states` /
`right_limit_switch_states`). Without it a freewheel diode's turn-on
jump was drawn as a ramp over the whole next step, and a DCM boost
diode's `i_avg` read 0.0234 A against 0.0535 A (−56 %), with no warning.
The same right limit gives a switch's `E_on` its current AT the edge.
Integrating `np.trapezoid(result.i(name), result.times)` yourself on a
DSED result keeps that error (first order in the step); read the summary,
or start each step from the right limit. A step whose two ends carry
different conductances (a fixed-step or TR-BDF2 commutation step, which
has no stored right limit) keeps the trapezoid of its two end powers.

With `store_every > 1` a recorded row is the mask of the LAST solve
before the sample, not of the whole stored interval, so duties and
integrals alias on a decimated grid (a buck switch's duty read 0.554
instead of 0.500 at `store_every = 67`). Keep `store_every = 1` for loss
work.

PSIM-style datasheet annotations add:

| Field | Source |
|---|---|
| `E_sw_total`, `P_sw_avg`, `n_turn_off_events`, `n_turn_off_events_dropped`, `E_sw_dropped`, `n_turn_off_events_capped`, `n_turn_off_events_unresolved`, `f_sw_estimate`, `t_rr` | diode + `diode_specs` |
| `n_chatter_gaps` | every diode (see below) |
| `E_sw_on_total`, `E_sw_off_total`, `n_turn_on_events` | switch + `switch_specs` |
| `P_core_avg`, `E_core_total`, `B_peak` | inductor + `core_loss_specs` |

## Resistor

`P_avg = v_R² / R` reconstructed from node voltages. Zero new kwargs.

```python
b.add_resistor("R_load", "vout", "gnd", 4.0)
res = p.simulate(b, t_end=1e-3, dt=1e-5)
loss = p.device_loss_summary(b, res)
print(next(e for e in loss if e["name"] == "R_load")["P_avg"])
```

## Switch — PSIM-style E_on / E_off

```python
b.add_switch("M1", "ds", "gnd", g_on=1e3, g_off=1e-9)

switch_specs = {"M1": {
    "E_on_ref":  120e-6,    # 120 µJ from datasheet
    "E_off_ref": 180e-6,
    "V_ref":     48.0,      # datasheet test condition
    "I_ref":     10.0,
}}
loss = p.device_loss_summary(b, res, switch_fn=pwm,
                                switch_specs=switch_specs)
```

Pulsim detects turn-on / turn-off edges from your `switch_fn` mask and
scales the datasheet energy linearly by the actual blocking voltage and
load current at each edge:

```
E_on_event  = E_on_ref · (|V_blocking| / V_ref) · (|I_load| / I_ref)
E_off_event = E_off_ref · ...
```

Honest errors: `V_ref` or `I_ref` ≤ 0 → `ValueError`. Unknown name in
`switch_specs` → `KeyError`.

## Diode — Q_rr or E_rr_ref

Two equivalent entry points:

```python
# Raw recovery charge (hard-recovery diodes):
diode_specs = {"D1": {"Q_rr": 50e-9}}

# Datasheet energy + reference voltage (fast/soft recovery):
diode_specs = {"D1": {"E_rr_ref": 30e-6, "V_R_ref": 600.0}}

# Either form takes the recovery time (default 100 ns):
diode_specs = {"D1": {"Q_rr": 50e-9, "t_rr": 35e-9}}
```

A turn-off is the end of a conducting stretch of the diode's recorded
state, at `t_off` (the end of its last conducting step). Pulsim reads
the reverse voltage `V_R = max(−v_D, 0)` one recovery time later, at
`t_off + t_rr` — a physical instant, the same on every engine and step
size — and accumulates `E_rr = Q_rr · V_R` or
`E_rr = E_rr_ref · (V_R / V_R_ref)`. `v_D` is read from the blocking
samples paired into step midpoints, which cancels the fixed engine's
trapezoidal alternation around a node held only by `g_off`. A
hard-switched diode blocks its full voltage at once; a naturally
commutated one (a line rectifier, a DCM turn-off) recovers against the
little voltage that has built up after its current zero, so its `E_rr`
is small — as it is in hardware.

When the step that carries a turn-off is longer than `t_rr`, the grid
cannot show how `v_D` rose inside it and `E_rr` depends on the step size.
Those turn-offs are counted in `n_turn_off_events_unresolved` and a
`RuntimeWarning` names the diode. The exception is a FORCED turn-off
(another switch flips on the same step while the diode still carries
current) whose blocking voltage is flat after the step: the voltage
jumped at the turn-off, so the reading is the plateau. Use a step no
longer than `t_rr` around the turn-offs (`engine='pwl'` with
`dt <= t_rr`, or a smaller `dt_max`), or pass the datasheet `t_rr`.

A turn-off is the start of BLOCKING. A gap in a diode's conduction
across which no step average of `v_D` falls below `−2·V_th`, and that no
other switch forced (no other bit of the recorded mask flips in it), is
chatter inside one conduction and is not counted (`n_chatter_gaps`).
The fixed engine produces it on a `V_th > 0` diode with a capacitor
across it: the trapezoidal rule reflects each ON step of the 0.1 ns
`Cs`–`g_on` mode from `+V_th` to about `−V_th`, the stamp drops to OFF,
and the inductor recharges `Cs` to `V_th` within a few steps, over and
over. A 100 kHz boost with 100 nF across D1 counted 4580 turn-offs at
`dt = T/400` (15887 at `T/2000`) instead of 299. The stamped current,
duty and conduction loss still follow what the solver carried (the
current detours through `Cs` in each gap), so a `RuntimeWarning` names
the chatter when the gaps cover 1 % or more of the conduction time. A
smaller `dt` does not cure it (the gaps covered 28 % of the conduction
at `T/400` and 56 % at `T/2000`); `engine='auto'` does.

Two guards keep immaterial conduction out, and both are reported:

* a stretch whose forward charge `Q_F` is below `1e-3` of the recovery
  charge the spec asks for (`Q_rr` or `E_rr_ref / V_R_ref`) is not a
  recovery
  (`n_turn_off_events_dropped`; excluded from `n_turn_off_events`,
  `f_sw_estimate` and the total) — the fixed engine's trapezoidal stiff
  mode can turn a reverse-biased diode ON for single ~µA steps. With the
  cap below, a dropped stretch could add at most 1e-3 of one event's
  energy; `E_sw_dropped` reports what they would add together, and a
  `RuntimeWarning` fires if that exceeds 1 % of `E_sw_total`. The test
  is the stretch's own charge, not the run's peak current: a 4 kA
  start-up inrush made the old `1e-3 · i_peak` rule drop all 237 light-
  load recoveries of a boost (each above `Q_rr`), `E_sw` 4.9× and
  `f_sw_estimate` 5× low. An `E_rr_curve` is judged on its own current
  axis instead: a stretch whose peak forward current is below `1e-3` of
  the curve's smallest tabulated positive current is dropped. (E_rr is
  read off the same axis, at the current the diode carries when it is
  cut off; the charge a curve asks for scales with that current, so a
  curve through the origin would have asked a 36 µA chatter blip for
  1e-12 C and counted all of them.) A spec without any recovery (counts
  only) judges the peak forward current against `1e-3` of the median
  peak of the stretches that reach `1e-3` of the run's peak current;
* no event recovers more than its own forward charge: `E_rr` is capped
  at `Q_F · V_R` (`n_turn_off_events_capped`). Capped events usually
  are one-step re-conductions from a ringing the grid does not resolve
  (a snubber faster than `dt`), which also inflate `n_turn_off_events`
  and `f_sw_estimate`, so a `RuntimeWarning` reports them.

The `SwitchedDiode` kernel model has no physical recovery charge — this
is a datasheet annotation pass, matching the LTspice "ideal-diode-with-Qrr"
option and the PSIM diode loss panel.

## Magnetic core loss

```python
core_loss_specs = {"L1": {
    "material": "N87",       # built-in catalogue (or CoreMaterial obj)
    "N_turns":  20,
    "A_core":   1.0e-4,      # m²
    "V_core":   5.0e-6,      # m³
    "use_igse": True,        # default; falls back to Steinmetz if trace
                              #   is shorter than one period
}}
```

Internally:

1. Reconstruct `B(t) = L · i(t) / (N · A_core)` from the kernel state
   trace.
2. Apply iGSE (Venkatachalam 2002) or classic Steinmetz at the
   FFT-estimated dominant frequency.
3. Multiply by `V_core` to get watts.

Inline `{"K": ..., "alpha": ..., "beta": ...}` is accepted in place of
`material`. Non-positive `N_turns` / `A_core` / `V_core` → `ValueError`
(no silent zeros).

## Junction temperature — `device_thermal_summary`

Same kwargs as `device_loss_summary`, plus per-device Foster networks:

```python
foster_M1 = [
    p.FosterStage(R_th_K_per_W=0.30, tau_s=2e-3),   # die-to-case
    p.FosterStage(R_th_K_per_W=1.20, tau_s=80e-3),  # case-to-heatsink
]

therm = p.device_thermal_summary(
    b, res,
    thermal_specs={
        "M1": {"stages": foster_M1, "T_ambient_C": 40.0},
        "D1": {"stages": [p.FosterStage(R_th_K_per_W=2.0,
                                            tau_s=1e-3)]},
    },
    T_ambient_C=25.0,        # fallback default
    # everything below is forwarded into device_loss_summary:
    switch_fn=pwm,
    switch_specs=switch_specs,
    diode_specs=diode_specs,
    core_loss_specs=core_loss_specs,
)
```

Per-device output:

| Field | Meaning |
|---|---|
| `P_cond_avg` | Average conduction loss (W) |
| `P_sw_avg` | PSIM-style switching loss (W) — constant offset over the run |
| `P_core_avg` | Steinmetz / iGSE core loss (inductor only) |
| `P_total_avg` | `P_cond_avg + P_sw_avg + P_core_avg` |
| `T_j_trace` | `np.ndarray` — full `T_j(t)` (°C); on `engine='dsed'` built from the step-start (right-limit) power as well, so it agrees with `P_cond_avg` |
| `T_j_avg`, `T_j_peak` | scalars (°C) |
| `T_ambient_C` | from spec or top-level default |
| `R_th_total` | `Σ R_th_i` over the Foster stages (K/W) |

The switching-loss and core-loss contributions are layered on the
conduction trace as a **constant offset** (thermal time constants ≫
nanosecond switching transitions, so the impulse energy is smeared
across the cycle — standard PSIM convention).

## What's *not* covered

The kernel runs at fixed `dt`. We don't resolve sub-`dt` switching
waveform shapes — the PSIM annotations capture the right *energy* per
event, not the *instantaneous* voltage/current trajectory through the
nanosecond transition. For waveform-level accuracy on a single
switching edge, drop `dt` ≪ T_sw or use a sub-step Newton step.

If you need a custom `E_on(I, T_j)` curve that doesn't fit the linear
`*_specs` shape, fall back to a state-aware `step_observer` driving
`LossAccumulator.add_sample(...)` + `add_switching_event(E_sw)` —
documented in `KNOWN_LIMITATIONS.md`.

## End-to-end example

[`examples/scripts/run_device_loss_to_thermal.py`](https://github.com/lgili/Pulsim/blob/main/examples/scripts/run_device_loss_to_thermal.py)
ships a 130-line buck → losses → `T_j(t)` pipeline with an optional
`--plot` flag that overlays the steady-state Foster asymptote.
