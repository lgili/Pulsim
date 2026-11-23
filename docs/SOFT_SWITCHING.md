# Soft switching: which node is V_DS, which branch is I_D

The soft-switching KPIs (`benchmarks/kpi`: `compute_zvs_fraction`,
`compute_zcs_fraction`, `compute_switching_loss_per_event`) read three
series per switch. This page fixes what each series is, per device kind, so
that a ZVS fraction means the same thing on every circuit.

## The three series

| series | meaning | sign |
|---|---|---|
| switch state | `True` while the device is commanded ON (the gate, not the conduction) | — |
| `V_DS` | voltage ACROSS the device, positive terminal first | positive when the device blocks forward voltage |
| `I_D` | current THROUGH the device, from its positive terminal to its negative one | positive when it conducts forward |

| device | positive terminal | `V_DS` | `I_D` |
|---|---|---|---|
| `add_switch("S", a, b, ...)` | `a` | `v(a) − v(b)` | `res.i("S")` |
| MOSFET / IGBT (`drain`/`collector`, `source`/`emitter`) | drain / collector | `v(drain) − v(source)` | `res.i(name)` (channel; the body diode is its own device) |
| low-side switch of a leg (`sw`, `gnd`) | `sw` | `v(sw)` | `res.i("Sl")` |
| high-side switch of a leg (`vdc`, `sw`) | `vdc` | `v(vdc) − v(sw)` | `res.i("Sh")` |
| diode (`anode`, `cathode`), for ZCS at turn-off | anode | `v(anode) − v(cathode)` | `res.i("D")` |

The switch state comes from the gate, never from the voltage: sample the
`switch_fn` at the result's times (`[fn(t).get(index) for t in res.times]`)
or use `res.switch_states[:, index]` for a gated switch. A diode has no
gate; its ZCS test uses its conduction flag.

## What each KPI reads

- **ZVS fraction**: at every OFF→ON edge of the switch state, `|V_DS|` at the
  sample `lookback_samples` before the edge (default 1) is compared with
  `threshold_v` (default 1 V). Fraction = soft turn-ons / turn-ons.
- **ZCS fraction**: the same with `|I_D|` against `threshold_a` (default 0.1 A).
- **Switching loss per event**: `∫ V_DS · I_D dt` over `window_s` after each
  transition (trapezoid over the samples, window end interpolated), summed and
  divided by the span. The window must contain the transition and end before
  the next one; it includes that window's conduction.

In a benchmark YAML the three KPIs take `switch_observable`,
`voltage_observable` / `current_observable`, `threshold_v` / `threshold_a`,
`lookback_samples` and `window_s`; the observables follow the table above.

## What the samples cannot show

- **A hard turn-on's capacitive energy** (`E_oss`): the collapse of a device
  capacitance through milliohms happens inside one step and is not in the
  samples, so it is not in `∫ V_DS · I_D dt` either. Book it from the stored
  energy: with the restart rule it is the event's impulse record
  (`res.reinit_records`, `dW`); the variable-step engine steps through the
  collapse. Closed forms for a half bridge with junction capacitances are
  in `python/tests/test_soft_switching_validation.py` (62.83 µJ and 1.688 µJ).
- **The sample before an edge on a variable grid** can be far before it. Use
  the engine's event samples (the exact engine stores both limits of an
  event; the fixed engine with `discontinuity_restart=True` stores the right
  limit beside the sample) or lower `threshold_v` only with a truth to hold
  it against.
- **The fixed engine without the restart rule** rings on the switch node at a
  dead-time transition and reads the gate at the step's end; its ZVS fraction
  on the LLC and phase-shifted full-bridge references is wrong by whole events
  (recorded as known disagreements in `benchmarks/manifests/v2_triage.yaml`).

## Reference circuits

`benchmarks/circuits/switching/isolated/llc_half_bridge_zvs.yaml` and
`psfb_full_bridge_zvs.yaml` (both soft),
`benchmarks/circuits/switching/buck/buck_hard_switching_reference.yaml` (hard). Their truths and the engine-by-engine results are in
`python/tests/test_soft_switching_validation.py`.
