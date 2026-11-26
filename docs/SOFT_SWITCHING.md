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
  divided by the trace's span. The window starts at the first sample that shows
  the new switch state and must end before the next transition; it includes that
  window's conduction. The benchmark runner skips the first stored sample (an
  algebraic node's first sample is not solved: a switch ON from t = 0 would count
  a turn-on with a made-up V_DS).

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

## The truth and the bounds

`benchmarks/tools/soft_switching_truth.py` (on `exact_truth.py`, mpmath, never
imports pulsim) gives, per switch, the left limit of `V_DS` at every turn-on, the
ZVS fraction and the switching energy of the windows above, exact. The KPI is a
trapezoid over a trace's samples, so it carries the quadrature error of that grid
even with exact states; the tool measures it on the same grid. The baselines
`benchmarks/baselines/{llc_half_bridge_zvs,psfb_full_bridge_zvs,buck_hard_switching_reference}.csv.gz`
are the same truth on the netlist grid with both limits at every event
(`soft_switching_truth.py baseline`). Checks and their bounds:
`python/tests/test_soft_switching_kpi.py` (runner columns, `pwl` with the
restart rule, `trbdf2`, `dsed`), run as a bench by
`benchmark_runner.py --benchmarks benchmarks/manifests/soft_switching.yaml`.

| benchmark | switch | ZVS (truth) | exact switching energy | avg over the run |
|---|---|---|---|---|
| LLC | S_hi / S_lo | 1.0 (\|V_DS\| ≤ 1.1 mV) | 7.022e-11 / 7.280e-11 J | 1.76 / 1.82 µW |
| PSFB | S1 / S2 / S3 / S4 | 1.0 (< 5 mV) | 5.702e-10 / 8.527e-10 / 3.178e-11 / 3.306e-11 J | (30 µs run) |
| buck (hard) | SH | 0 (24.02–24.08 V, then 1.42–6.01 V in DCM) | 7.406e-6 J | 14.81 mW |

The buck has no device capacitance: its "switching" energy is the conduction in
the 1 µs windows, not an `E_oss`.

## Known disagreements (strict xfails)

| engine | benchmark | measured | truth |
|---|---|---|---|
| plain `pwl` (trap 9) | LLC | V_DS at S_lo's 4 µs turn-on 0.918 V; S_hi energy 9.250e-11 J (+32 %) | −0.0011 V; 7.022e-11 J |
| plain `pwl` (trap 9) | PSFB | ZVS of S3 and S4 2/3 (8.53 V at 8 µs); S1 energy 1.046e-9 J (+84 %) | 1.0; 5.702e-10 J |
| plain `pwl` (trap 9) | buck | ZVS 0.04 (1 of 25 turn-ons read soft) | 0 of 24 |
| `trbdf2` (F-S-5) | PSFB | V(b) 599.8 V at 3e-16 s after a consistent start | 400 V |

Use the restart rule (`discontinuity_restart=True`), `trbdf2` or `dsed` for these KPIs.

## Reference circuits

`benchmarks/circuits/switching/isolated/llc_half_bridge_zvs.yaml` and
`psfb_full_bridge_zvs.yaml` (both soft),
`benchmarks/circuits/switching/buck/buck_hard_switching_reference.yaml` (hard). Their truths and the engine-by-engine results are in
`python/tests/test_soft_switching_validation.py`.
