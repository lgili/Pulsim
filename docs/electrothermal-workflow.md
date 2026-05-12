# Electrothermal Workflow

This guide shows the supported way to configure and consume electrothermal simulations in PulsimCore.

## Thermal-Capable Components

`component.thermal.enabled: true` is accepted only for these canonical component types:

- `mosfet` (aliases: `m`, `nmos`, `pmos`)
- `igbt` (alias: `q`)
- `bjt_npn` (aliases: `bjtnpn`, `bjt-npn`)
- `bjt_pnp` (aliases: `bjtpnp`, `bjt-pnp`)

If thermal is enabled on unsupported components (for example `resistor`), parsing fails with:

- `PULSIM_YAML_E_THERMAL_UNSUPPORTED_COMPONENT`

## Global Thermal Block (`simulation.thermal`)

Use `simulation.thermal` to define ambient and defaults:

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

Fields:

- `enabled`: enables electrothermal coupling
- `ambient`: ambient temperature (degC)
- `policy`: `loss_only` or `loss_with_temperature_scaling`
- `default_rth`: default thermal resistance used by non-strict fallback
- `default_cth`: default thermal capacitance used by non-strict fallback

## Per-Component Thermal Port (`component.thermal`)

Example:

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
    thermal:
      enabled: true
      rth: 0.5
      cth: 1e-4
      temp_init: 25.0
      temp_ref: 25.0
      alpha: 0.004
```

Thermal validation rules when `thermal.enabled=true`:

- `rth` must be finite and `> 0`
- `cth` must be finite and `>= 0`
- `temp_init`, `temp_ref`, `alpha` must be finite

Invalid ranges fail with:

- `PULSIM_YAML_E_THERMAL_RANGE_INVALID`

## Strict vs Non-Strict Parsing

In strict parser mode, `rth` and `cth` are required for each thermal-enabled component.

```python
import pulsim as ps

opts = ps.YamlParserOptions()
opts.strict = True
parser = ps.YamlParser(opts)
```

If missing in strict mode:

- `PULSIM_YAML_E_THERMAL_MISSING_REQUIRED`

In non-strict mode, missing `rth`/`cth` are defaulted from `simulation.thermal.default_rth/default_cth` and warnings are emitted:

- `PULSIM_YAML_W_THERMAL_DEFAULT_APPLIED`

## Python Result Consumption

`SimulationResult` keeps legacy summaries and now exposes unified per-component electrothermal telemetry:

- `result.loss_summary`
- `result.thermal_summary`
- `result.component_electrothermal`

Example:

```python
import pulsim as ps

parser = ps.YamlParser(ps.YamlParserOptions())
circuit, options = parser.load("benchmarks/circuits/buck_electrothermal.yaml")
options.newton_options.num_nodes = int(circuit.num_nodes())
options.newton_options.num_branches = int(circuit.num_branches())

sim = ps.Simulator(circuit, options)
result = sim.run_transient(circuit.initial_state())

for item in result.component_electrothermal:
    print(
        item.component_name,
        "thermal=", item.thermal_enabled,
        "P_loss=", item.total_loss,
        "E_loss=", item.total_energy,
        "T_peak=", item.peak_temperature,
    )
```

Notes:

- All non-virtual circuit components are included.
- Zero-loss components are still reported.
- Components without enabled thermal port still include deterministic thermal fields, with ambient-derived values.

## Programmatic Configuration (No YAML Thermal Block)

You can configure thermal coupling directly in Python:

```python
import pulsim as ps

options.enable_losses = True
options.thermal.enable = True
options.thermal.ambient = 25.0
options.thermal.policy = ps.ThermalCouplingPolicy.LossWithTemperatureScaling

cfg = ps.ThermalDeviceConfig()
cfg.enabled = True
cfg.rth = 0.5
cfg.cth = 1e-4
cfg.temp_init = 25.0
cfg.temp_ref = 25.0
cfg.alpha = 0.004
options.thermal_devices["M1"] = cfg
```

## Benchmark/KPI Gate Integration

The electrothermal gate now checks component-level consistency KPIs:

- `component_coverage_rate`
- `component_coverage_gap`
- `component_loss_summary_consistency_error`
- `component_thermal_summary_consistency_error`

Threshold files:

- `benchmarks/kpi_thresholds.yaml` (optional in general gate)
- `benchmarks/kpi_thresholds_electrothermal.yaml` (required in electrothermal gate)

## Junction-temperature KPI (Phase 27)

Phase 27 added a Python-side Foster-network estimator that turns an
observed V (across a dissipating element) into a junction temperature
trajectory `T_j(t)`. It's exposed as a `junction_temperature` metric in
the YAML KPI block:

```yaml
benchmark:
  id: electrothermal_resistor_self_heating
  kpi:
    - metric: junction_temperature
      observable: V(r_diss)           # voltage across the dissipating element
      r_resistor: 1.0                 # to derive P(t) = V²/R
      r_th_jc: 5.0                    # junction-to-case (K/W)
      c_th_jc: 0.1                    # junction-to-case (J/K)
      t_ambient_c: 25.0
      r_th_ca: 0.0                    # case-to-ambient (K/W), optional
      label: r_diss
```

Output columns: `kpi__t_j_max_c__r_diss`, `kpi__t_j_final_c__r_diss`,
`kpi__delta_t_j_c__r_diss`.

Behind the scenes the runner calls
`compute_power_dissipation_resistor` then
`compute_junction_temperature` (a forward-Euler integration of
`C_th · dT_j/dt + (T_j − T_amb)/R_th = P(t)`). The helpers are pure
Python and importable directly — chain multiple calls for
Cauer/Foster networks `J → C → HS → A`.

See [KPI Reference → Junction temperature](kpi-reference.md#11-junction-temperature-junction_temperature)
for the full parameter table and Python signature.

The shipped reference benchmark is
`benchmarks/circuits/electrothermal_resistor_self_heating.yaml` —
a 5 V across 1 Ω resistor (P = 25 W constant), R_th = 5 K/W,
C_th = 0.1 J/K, T_amb = 25 °C. Analytical steady state: T_j = 150 °C.
After 2 s (= 4 τ) the simulation reports `T_j_final = 147.76 °C`.

## See also

- [`components-reference.md`](components-reference.md#3-switching-devices) —
  full `thermal:` block schema on MOSFET / IGBT / BJT components.
- [`kpi-reference.md`](kpi-reference.md) — every KPI, including the
  full Foster-network signature.
