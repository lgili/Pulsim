# Pulsim Netlist Format Reference (YAML)

Pulsim uses a **versioned YAML** netlist format. This document describes the required structure, supported component types, and waveform definitions.

## Top-Level Structure

```yaml
schema: pulsim-v1
version: 1
simulation:
  tstart: 0.0
  tstop: 1e-3
  dt: 1e-6
  step_mode: variable
components:
  - type: voltage_source
    name: V1
    nodes: [in, 0]
    waveform: {type: dc, value: 5.0}
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 1k
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema` | string | Must be `pulsim-v1` |
| `version` | int | Schema version (currently `1`) |
| `components` | list | Component list |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `simulation` | map | Simulation options |
| `models` | map | Reusable component models |

## Components

For the full canonical backend catalog (all supported YAML types), see
[Supported Components Catalog](supported-components-catalog.md). For GUI-focused
parity status, see [GUI Backend Parity](gui-component-parity.md).

### Passive Components

- **Resistor**: `type: resistor` (or `R`), `value`
- **Capacitor**: `type: capacitor` (or `C`), `value`, optional `ic`
- **Inductor**: `type: inductor` (or `L`), `value`, optional `ic`

### Sources

- **Voltage Source**: `type: voltage_source` (or `V`)
  - `waveform` supports `dc`, `pulse`, `sine`, `pwm`
- **Current Source**: `type: current_source` (or `I`) with `value` (DC)

### Switching Devices

- **Ideal Diode**: `type: diode` (or `D`), `g_on`, `g_off`
- **Ideal Switch**: `type: switch` (or `S`), `ron`/`roff` or `g_on`/`g_off`, `initial_state`
- **Voltage-Controlled Switch**: `type: vcswitch`, nodes `[ctrl, t1, t2]`, `v_threshold`, `g_on`, `g_off`

### Power Devices

- **MOSFET**: `type: mosfet`/`nmos`/`pmos` (or `M`)
- **IGBT**: `type: igbt` (or `Q`)

### Transformer

- **Transformer**: `type: transformer` (or `T`), `turns_ratio`

### Magnetic Core (MVP)

Supported components for `magnetic_core` block:

- `saturable_inductor`
- `coupled_inductor`
- `transformer`

Canonical fields (current MVP):

- `enabled` (bool)
- `model` (`saturation`)
- `saturation_current` (optional, component-dependent)
- `saturation_inductance` (optional, component-dependent)
- `saturation_exponent` (optional, component-dependent)
- `core_loss_k` (optional, default `0.0`)
- `core_loss_alpha` (optional, default `2.0`)

Example:

```yaml
- type: transformer
  name: Tmag
  nodes: [p1, 0, s1, 0]
  turns_ratio: 2.0
  magnetic_core:
    enabled: true
    model: saturation
    core_loss_k: 0.08
    core_loss_alpha: 2.0
```

When enabled with `core_loss_k > 0`, backend exports `"<component>.core_loss"` in
`result.virtual_channels` with metadata:

- `domain: loss`
- `unit: W`
- `source_component: <component_name>`

## Waveforms

```yaml
waveform:
  type: dc
  value: 12
```

```yaml
waveform:
  type: pulse
  v_initial: 0
  v_pulse: 5
  t_delay: 0
  t_rise: 1e-9
  t_fall: 1e-9
  t_width: 1e-6
  period: 2e-6
```

```yaml
waveform:
  type: sine
  amplitude: 2.5
  frequency: 1000
  offset: 0
  phase: 0
```

```yaml
waveform:
  type: pwm
  v_high: 10
  v_low: 0
  frequency: 20000
  duty: 0.5
  dead_time: 1e-6
  phase: 0
```

## Models and Overrides

```yaml
models:
  m1:
    type: mosfet
    params:
      vth: 2.0
      kp: 20e-6

components:
  - type: mosfet
    name: M1
    nodes: [g, d, s]
    use: m1
    params:
      kp: 40e-6  # override
```

## Simulation Options

```yaml
simulation:
  tstart: 0.0
  tstop: 1e-3
  dt: 1e-6
  step_mode: variable    # fixed | variable
  formulation: projected_wrapper  # projected_wrapper | direct
  direct_formulation_fallback: true
  dt_min: 1e-12
  dt_max: 1e-3
  adaptive_timestep: true # override avançado; prefira step_mode
  enable_events: true
  enable_losses: true
  integrator: trbdf2   # trapezoidal, bdf1, bdf2, trbdf2, rosenbrockw, sdirk2
  control:
    mode: auto         # auto | continuous | discrete
    sample_time: 1e-4  # required when mode=discrete
  thermal:
    enabled: true
    ambient: 25.0
    policy: loss_with_temperature_scaling
    default_rth: 1.0
    default_cth: 0.1
```

`simulation.backend` e `simulation.sundials` são chaves legadas e não fazem parte
da superfície suportada de transiente. Use `step_mode` + `formulation`.

### Control Update Mode

You can configure control update policy using either flat keys:

- `simulation.control_mode`
- `simulation.control_sample_time`

or nested keys:

- `simulation.control.mode`
- `simulation.control.sample_time`

Supported values for mode:

- `auto`: use explicit sample time if set, otherwise infer from highest PWM frequency.
- `continuous`: update control on every accepted simulation step.
- `discrete`: update PI/PID blocks only at sample boundaries.

In strict parsing, `discrete` requires `sample_time > 0`.

### Frequency Analysis (AC Sweep)

Use `simulation.frequency_analysis` for backend AC sweep workflows:

```yaml
simulation:
  frequency_analysis:
    enabled: true
    mode: open_loop_transfer
    anchor: auto
    sweep:
      scale: log
      f_start_hz: 10.0
      f_stop_hz: 100000.0
      points: 80
    injection_current_amplitude: 1.0
    perturbation: {positive: in, negative: 0}
    output: {positive: out, negative: 0}
```

Supported `mode` values:

- `open_loop_transfer`
- `closed_loop_transfer`
- `input_impedance`
- `output_impedance`

Supported `anchor` values:

- `dc`
- `periodic`
- `averaged`
- `auto`

The parser enforces deterministic validation for sweep ranges, point count, and port bindings.
For full backend contract and Python result usage, see
[Frequency Analysis (AC Sweep)](frequency-analysis-ac-sweep.md).

### Averaged Converter Mode

Use `simulation.averaged_converter` for backend averaged plant runs:

```yaml
simulation:
  averaged_converter:
    enabled: true
    topology: buck
    operating_mode: auto
    envelope_policy: warn
    vin_source: Vin
    inductor: L1
    capacitor: C1
    load_resistor: Rload
    output_node: out
    duty: 0.4
    duty_min: 0.0
    duty_max: 0.95
    switching_frequency_hz: 100000.0
    initial_inductor_current: 0.0
    initial_output_voltage: 0.0
    ccm_current_threshold: 0.0
```

Supported values:

- `topology`: `buck` | `boost` | `buck_boost`
- `operating_mode`: `ccm` | `dcm` | `auto`
- `envelope_policy`: `strict` | `warn`

Deterministic checks:

- required mapping fields when `enabled=true`
- valid duty bounds (`0 <= duty_min <= duty <= duty_max <= 1`)
- finite `ccm_current_threshold >= 0`
- finite `switching_frequency_hz > 0`
- mapped component types (`vin_source` voltage source, `inductor` inductor, `capacitor` capacitor, `load_resistor` resistor)

For full runtime/result/frontend contract and migration guidance, see
[Averaged Converter Modeling](averaged-converter-modeling.md).

### Control Blocks for Closed-Loop Converters

Common blocks for converter control loops:

- `pi_controller`: `kp`, `ki`, `output_min`, `output_max`, `anti_windup`
- `pid_controller`: `kp`, `ki`, `kd`, `output_min`, `output_max`, `anti_windup`
- `pwm_generator`: `frequency`, `duty`, `duty_min`, `duty_max`, `duty_from_input`, `duty_from_channel`, `duty_gain`, `duty_offset`, `target_component`
- `sample_hold`: `sample_period`

Typical closed-loop pattern:

```yaml
components:
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

`target_component` links the PWM output to a switch-like power device (`mosfet`,
`igbt`, `switch`, `vcswitch`), enabling mixed-domain control + electrical stamping.

### Thermal Ports and Supported Types

Per-component thermal block:

```yaml
components:
  - type: resistor
    name: Rload
    nodes: [vout, 0]
    value: 8.0
    thermal:
      enabled: true
      rth: 3.0
      cth: 0.2
```

Thermal-port enablement is supported for:

- `resistor`
- `diode`
- `mosfet`
- `igbt`
- `bjt_npn`
- `bjt_pnp`

For thermal-port rules, strict/non-strict behavior, and complete electrothermal examples, see
[Electrothermal Workflow](electrothermal-workflow.md).

### Solver Configuration

Use `simulation.solver` to control linear/iterative solver selection and nonlinear aids:

```yaml
simulation:
  solver:
    order: [klu, gmres]
    fallback_order: [sparselu]
    allow_fallback: true
    auto_select: true
    size_threshold: 400
    nnz_threshold: 2000
    diag_min_threshold: 1e-12
    preconditioner: ilut        # or amg (if available)
    ilut_drop_tolerance: 1e-3
    ilut_fill_factor: 10
    iterative:
      max_iterations: 200
      tolerance: 1e-8
      restart: 40
      preconditioner: ilut      # or amg (if available)
      ilut_drop_tolerance: 1e-3
      ilut_fill_factor: 10
      enable_scaling: true
      scaling_floor: 1e-12
    nonlinear:
      anderson:
        enable: true
        depth: 5
        beta: 0.5
      broyden:
        enable: false
        max_size: 8
      newton_krylov:
        enable: false
      trust_region:
        enable: true
        radius: 1.0
        shrink: 0.5
        expand: 1.2
        min: 1e-4
        max: 10.0
      reuse_jacobian_pattern: true
```

You can also set Newton options directly under `simulation.newton` if preferred.

```yaml
simulation:
  newton:
    max_iterations: 50
    enable_newton_krylov: true
    krylov_residual_cache_tolerance: 1e-8
    reuse_jacobian_pattern: true
```

### Periodic Steady-State

```yaml
simulation:
  shooting:
    period: 10e-6
    max_iterations: 30
    tolerance: 1e-6
    relaxation: 0.5
    store_last_transient: true

  harmonic_balance:
    period: 10e-6
    num_samples: 128
    max_iterations: 40
    tolerance: 1e-6
    relaxation: 0.5
    initialize_from_transient: true
```

## SI Prefixes

Values support SI prefixes: `f`, `p`, `n`, `u`, `m`, `k`, `meg`, `g`, `t`.

Examples: `1k` = 1000, `100n` = 100e-9, `4.7u` = 4.7e-6.
