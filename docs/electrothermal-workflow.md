# Electrothermal Workflow

This guide documents the supported backend path for coupled electrical + control + thermal simulation in PulsimCore.

## 1) Enable Global Loss and Thermal Blocks

Use `simulation.enable_losses` and `simulation.thermal` together:

```yaml
simulation:
  enable_losses: true
  thermal:
    enabled: true
    ambient: 25.0
    policy: loss_with_temperature_scaling
    default_rth: 1.0
    default_cth: 0.1
```

Field semantics:

- `enabled`: enables thermal state integration.
- `ambient`: ambient temperature in degC.
- `policy`: `loss_only` or `loss_with_temperature_scaling`.
- `default_rth`: non-strict fallback for missing per-device `rth`.
- `default_cth`: non-strict fallback for missing per-device `cth`.

## 2) Thermal-Capable Component Types

`component.thermal.enabled: true` is supported for:

- `resistor`
- `diode`
- `mosfet` (aliases: `m`, `nmos`, `pmos`)
- `igbt` (alias: `q`)
- `bjt_npn` (aliases: `bjtnpn`, `bjt-npn`)
- `bjt_pnp` (aliases: `bjtpnp`, `bjt-pnp`)

If thermal is enabled on an unsupported type, YAML parsing fails with:

- `PULSIM_YAML_E_THERMAL_UNSUPPORTED_COMPONENT`

## 3) Per-Component Thermal Port

```yaml
components:
  - type: mosfet
    name: M1
    nodes: [gate, drain, source]
    vth: 2.5
    kp: 0.01
    g_off: 1e-8
    loss:
      eon: 1.0e-6
      eoff: 1.0e-6
      err: 0.5e-6
    thermal:
      enabled: true
      rth: 0.8
      cth: 0.02
      temp_init: 25.0
      temp_ref: 25.0
      alpha: 0.004
```

Datasheet switching-loss surface (optional, backend-evaluated):

```yaml
components:
  - type: mosfet
    name: M1
    nodes: [gate, drain, source]
    loss:
      model: datasheet
      axes:
        current: [0.0, 10.0, 20.0]      # A
        voltage: [0.0, 200.0, 400.0]    # V
        temperature: [25.0, 125.0]      # degC
      eon:  [ ... flat table ... ]      # J
      eoff: [ ... flat table ... ]      # J
      err:  [ ... flat table ... ]      # J (optional)
```

Table order is row-major `(current, voltage, temperature)`:
`index = ((i_current * N_voltage) + i_voltage) * N_temperature + i_temperature`.

Validation for datasheet loss model:

- `axes.current/voltage/temperature` must be finite and strictly increasing
- each table length must match `N_current * N_voltage * N_temperature`
- each energy sample must be finite and `>= 0`

Validation rules when `thermal.enabled: true`:

- `rth` finite and `> 0`
- `cth` finite and `>= 0`
- `temp_init`, `temp_ref`, `alpha` finite

Invalid ranges fail with:

- `PULSIM_YAML_E_THERMAL_RANGE_INVALID`

Optional staged thermal networks:

```yaml
thermal:
  enabled: true
  network: foster      # single_rc | foster | cauer
  rth_stages: [0.3, 0.7]
  cth_stages: [0.01, 0.05]
  temp_init: 25.0
  temp_ref: 25.0
  alpha: 0.004
```

Staged thermal validation:

- `rth_stages` and `cth_stages` are required for `foster`/`cauer`
- stage arrays must be non-empty and same size
- each `rth_stages[i]` finite and `> 0`
- each `cth_stages[i]` finite and `>= 0`

Deterministic diagnostics:

- `PULSIM_YAML_E_THERMAL_NETWORK_INVALID`
- `PULSIM_YAML_E_THERMAL_DIMENSION_INVALID`

Optional shared sink coupling (multiple devices on one heatsink):

```yaml
components:
  - type: mosfet
    name: M1
    nodes: [gate, vin, sw]
    thermal:
      enabled: true
      rth: 0.7
      cth: 0.02
      shared_sink_id: hs_main
      shared_sink_rth: 0.35
      shared_sink_cth: 0.10
  - type: diode
    name: D1
    nodes: [0, sw]
    thermal:
      enabled: true
      rth: 1.2
      cth: 0.03
      shared_sink_id: hs_main
      shared_sink_rth: 0.35
      shared_sink_cth: 0.10
```

Shared sink rules:

- `shared_sink_id` groups devices into one common sink.
- all devices in the same `shared_sink_id` must use identical
  `shared_sink_rth/shared_sink_cth`.
- `shared_sink_rth` must be finite and `> 0`.
- `shared_sink_cth` must be finite and `>= 0`.
- `shared_sink_rth/shared_sink_cth` without `shared_sink_id` is invalid.

## 4) Closed-Loop Control Sampling (Important for Stability)

Control update policy is configured with `simulation.control`:

```yaml
simulation:
  control:
    mode: auto        # auto | continuous | discrete
    sample_time: 1e-4 # required when mode=discrete
```

Behavior:

- `auto`:
  - uses `sample_time` if provided;
  - otherwise infers `Ts = 1 / f_pwm_max` from the highest PWM frequency found in `pwm_generator`/PWM sources;
  - if no PWM is present, runs as continuous.
- `continuous`: updates control blocks at every accepted solver step.
- `discrete`: updates PI/PID blocks only when `dt_accum >= sample_time`.

Notes:

- In YAML strict validation, `mode: discrete` requires `sample_time > 0`.
- `sample_hold` has independent `sample_period` behavior.

## 5) Strict vs Non-Strict Thermal Validation

Strict parser mode requires per-component `rth` and `cth` when thermal is enabled:

```python
import pulsim as ps

opts = ps.YamlParserOptions()
opts.strict = True
parser = ps.YamlParser(opts)
```

If missing in strict mode:

- `PULSIM_YAML_E_THERMAL_MISSING_REQUIRED`

In non-strict mode, missing `rth`/`cth` are filled from `simulation.thermal.default_rth/default_cth`, and warnings are emitted:

- `PULSIM_YAML_W_THERMAL_DEFAULT_APPLIED`

## 6) Integrated Closed-Loop + Thermal YAML Example

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 20e-3
  dt: 1e-6
  step_mode: variable
  enable_events: true
  enable_losses: true
  control:
    mode: auto
  thermal:
    enabled: true
    ambient: 25.0
    policy: loss_with_temperature_scaling
    default_rth: 1.0
    default_cth: 0.05

components:
  - type: voltage_source
    name: Vin
    nodes: [vin, 0]
    waveform: {type: dc, value: 12.0}

  - type: voltage_source
    name: Vref
    nodes: [vref, 0]
    waveform: {type: dc, value: 6.0}

  - type: mosfet
    name: M1
    nodes: [0, vin, sw]
    vth: 3.0
    kp: 0.35
    g_off: 1e-8
    loss: {eon: 2e-6, eoff: 2e-6}
    thermal: {enabled: true, rth: 1.0, cth: 0.05}

  - type: diode
    name: D1
    nodes: [0, sw]
    g_on: 350.0
    g_off: 1e-9
    thermal: {enabled: true, rth: 2.0, cth: 0.03}

  - type: inductor
    name: L1
    nodes: [sw, vout]
    value: 220u

  - type: capacitor
    name: Cout
    nodes: [vout, 0]
    value: 220u

  - type: resistor
    name: Rload
    nodes: [vout, 0]
    value: 8.0
    thermal: {enabled: true, rth: 3.0, cth: 0.2}

  - type: pi_controller
    name: PI1
    nodes: [vref, vout, 0]
    kp: 0.08
    ki: 100.0
    output_min: 0.0
    output_max: 0.95
    anti_windup: 1.0

  - type: pwm_generator
    name: PWM1
    nodes: [0]
    frequency: 10000.0
    duty_from_channel: PI1
    duty_min: 0.0
    duty_max: 0.95
    target_component: M1
```

## 7) Python Result Consumption

`SimulationResult` keeps summary and per-component telemetry:

- `result.loss_summary`
- `result.thermal_summary`
- `result.component_electrothermal`
- `result.virtual_channels` (includes canonical thermal traces when enabled)
- `result.virtual_channel_metadata`

```python
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("examples/09_buck_closed_loop_loss_thermal_validation_backend.yaml")
options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

for item in result.component_electrothermal:
    print(
        item.component_name,
        "thermal=", item.thermal_enabled,
        "Pavg=", item.average_power,
        "Tfinal=", item.final_temperature,
        "Tpeak=", item.peak_temperature,
    )

t_m1 = result.virtual_channels["T(M1)"]
meta_m1 = result.virtual_channel_metadata["T(M1)"]
print("T(M1) samples:", len(t_m1), "time:", len(result.time))
print("metadata:", meta_m1.domain, meta_m1.source_component, meta_m1.unit)
```

Notes:

- All non-virtual components are listed in `component_electrothermal`.
- Components with thermal disabled are still reported with deterministic ambient-based thermal fields.
- When `enable_losses=true` and `thermal.enabled=true`, transient output includes canonical thermal traces
  in `result.virtual_channels` named `T(<component_name>)` (for thermal-enabled components).
- When `enable_losses=true`, transient output also includes canonical per-component loss traces:
  - `Pcond(<component>)`
  - `Psw_on(<component>)`
  - `Psw_off(<component>)`
  - `Prr(<component>)`
  - `Ploss(<component>)`
  All these channels are aligned with `result.time` and tagged in
  `result.virtual_channel_metadata` with `domain="loss"` and `unit="W"`.
- YAML parser validates `loss.eon/eoff/err` as finite and non-negative; invalid values
  fail deterministically with `PULSIM_YAML_E_LOSS_RANGE_INVALID`.
- Datasheet loss schema errors are deterministic:
  - `PULSIM_YAML_E_LOSS_MODEL_INVALID`
  - `PULSIM_YAML_E_LOSS_AXIS_INVALID`
  - `PULSIM_YAML_E_LOSS_DIMENSION_INVALID`
- Runtime now enforces a deterministic post-run consistency guard between canonical
  electrothermal channels and summary surfaces (`loss_summary`, `thermal_summary`,
  `component_electrothermal`). Any mismatch beyond tolerance fails the run with a
  deterministic diagnostic.
- Thermal traces are emitted only when all conditions hold:
  - `simulation.enable_losses: true`
  - `simulation.thermal.enabled: true`
  - `component.thermal.enabled: true` (or equivalent runtime thermal config)
- Consistency constraints for each thermal-enabled component `X`:
  - `component_electrothermal[X].final_temperature == last(T(X))`
  - `component_electrothermal[X].peak_temperature == max(T(X))`
  - `component_electrothermal[X].average_temperature == mean(T(X))`

## 8) KPI Gate Integration

The electrothermal gate validates consistency KPIs:

- `component_coverage_rate`
- `component_coverage_gap`
- `component_loss_summary_consistency_error`
- `component_thermal_summary_consistency_error`

Threshold files:

- `benchmarks/kpi_thresholds.yaml` (optional in general gate)
- `benchmarks/kpi_thresholds_electrothermal.yaml` (required in electrothermal gate)

## 9) Backend Contract (GUI Integration)

For each accepted transient sample index `k`, backend guarantees:

- `result.time[k]` exists and is monotonic.
- every exported `result.virtual_channels[name][k]` is aligned to the same `k`.
- channel metadata exists in `result.virtual_channel_metadata[name]`.

Electrothermal canonical guarantees:

- thermal channels use `T(<component_name>)`.
- loss channels use:
  - `Pcond(<component>)`
  - `Psw_on(<component>)`
  - `Psw_off(<component>)`
  - `Prr(<component>)`
  - `Ploss(<component>)`
- thermal/loss channels are backend-computed physics outputs (not UI post-processing).
- reductions are internally consistent:
  - `component_electrothermal[i].final_temperature == last(T(component))`
  - `component_electrothermal[i].peak_temperature == max(T(component))`
  - `component_electrothermal[i].average_temperature == mean(T(component))`

## 10) GUI Responsibility Boundary

GUI should own:

- form/wizard UX for scalar and datasheet model entry.
- unit conversion helpers and validation message presentation.
- curve import UX (CSV/PDF digitization) before sending numeric arrays to backend.
- plotting and dashboard composition.

GUI must not own:

- synthetic thermal curve generation.
- reconstruction of switching/conduction losses from electrical channels.
- replacement or smoothing of backend physical traces.
- heuristic domain inference from channel names when metadata is available.

## 11) Scalar-to-Datasheet Migration

Use the dedicated migration guide:

- `docs/electrothermal-migration-scalar-to-datasheet.md`
