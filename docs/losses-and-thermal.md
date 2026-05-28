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
| `duty_closed` | float ∈ [0, 1] | switch (needs `switch_fn=`) |

PSIM-style datasheet annotations add:

| Field | Source |
|---|---|
| `E_sw_total`, `P_sw_avg`, `n_turn_off_events`, `f_sw_estimate` | diode + `diode_specs` |
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
```

For every turn-off event in `result.commutation_events` matching the
branch, Pulsim interpolates `|V_R(t_event)|` from the trace and
accumulates `E_rr = Q_rr · V_R` or `E_rr = E_rr_ref · (V_R / V_R_ref)`.

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
| `T_j_trace` | `np.ndarray` — full `T_j(t)` (°C) |
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

[`examples/scripts/run_device_loss_to_thermal.py`](../examples/scripts/run_device_loss_to_thermal.py)
ships a 130-line buck → losses → `T_j(t)` pipeline with an optional
`--plot` flag that overlays the steady-state Foster asymptote.
