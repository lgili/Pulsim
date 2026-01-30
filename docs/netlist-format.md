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
  dt_min: 1e-12
  dt_max: 1e-3
  adaptive_timestep: true
  enable_events: true
  enable_losses: true
```

## SI Prefixes

Values support SI prefixes: `f`, `p`, `n`, `u`, `m`, `k`, `meg`, `g`, `t`.

Examples: `1k` = 1000, `100n` = 100e-9, `4.7u` = 4.7e-6.
