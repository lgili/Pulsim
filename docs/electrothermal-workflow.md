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
