# Pulsim Netlist Format Reference

Pulsim uses JSON as its primary netlist format. This document describes the complete schema and all supported component types.

## Table of Contents

1. [Netlist Structure](#netlist-structure)
2. [Component Types](#component-types)
3. [Waveform Sources](#waveform-sources)
4. [Simulation Options](#simulation-options)
5. [Output Configuration](#output-configuration)

---

## Netlist Structure

A Pulsim netlist is a JSON object with the following top-level structure:

```json
{
  "name": "Circuit Name",
  "description": "Optional description",
  "components": [...],
  "sources": [...],
  "simulation": {...},
  "output": {...},
  "options": {...}
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `components` | array | List of circuit components |
| `simulation` | object | Simulation configuration |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Circuit name |
| `description` | string | Circuit description |
| `sources` | array | Additional source definitions (PWM, etc.) |
| `output` | object | Output configuration |
| `options` | object | Solver options |

---

## Component Types

### Passive Components

#### Resistor (R)

```json
{
  "type": "R",
  "name": "R1",
  "nodes": ["node1", "node2"],
  "value": 1000
}
```

| Parameter | Type | Unit | Description |
|-----------|------|------|-------------|
| `value` | number | Ω | Resistance |

#### Capacitor (C)

```json
{
  "type": "C",
  "name": "C1",
  "nodes": ["node1", "node2"],
  "value": 1e-6,
  "ic": 0
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `value` | number | F | required | Capacitance |
| `ic` | number | V | 0 | Initial voltage |

#### Inductor (L)

```json
{
  "type": "L",
  "name": "L1",
  "nodes": ["node1", "node2"],
  "value": 100e-6,
  "ic": 0
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `value` | number | H | required | Inductance |
| `ic` | number | A | 0 | Initial current |

---

### Voltage Sources

#### DC Voltage Source

```json
{
  "type": "V",
  "name": "V1",
  "nodes": ["positive", "negative"],
  "value": 12
}
```

#### Pulse Voltage Source

```json
{
  "type": "V",
  "name": "Vpulse",
  "nodes": ["out", "0"],
  "waveform": {
    "type": "pulse",
    "v1": 0,
    "v2": 5,
    "td": 0,
    "tr": 1e-9,
    "tf": 1e-9,
    "pw": 1e-6,
    "period": 2e-6
  }
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `v1` | number | V | 0 | Initial/low voltage |
| `v2` | number | V | required | Pulse/high voltage |
| `td` | number | s | 0 | Delay time |
| `tr` | number | s | 0 | Rise time |
| `tf` | number | s | 0 | Fall time |
| `pw` | number | s | required | Pulse width |
| `period` | number | s | required | Period |

#### Sinusoidal Voltage Source

```json
{
  "type": "V",
  "name": "Vsin",
  "nodes": ["out", "0"],
  "waveform": {
    "type": "sin",
    "vo": 0,
    "va": 1,
    "freq": 1000,
    "td": 0,
    "theta": 0,
    "phi": 0
  }
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `vo` | number | V | 0 | DC offset |
| `va` | number | V | required | Amplitude |
| `freq` | number | Hz | required | Frequency |
| `td` | number | s | 0 | Delay time |
| `theta` | number | 1/s | 0 | Damping factor |
| `phi` | number | deg | 0 | Phase |

#### PWL (Piecewise Linear) Voltage Source

```json
{
  "type": "V",
  "name": "Vpwl",
  "nodes": ["out", "0"],
  "waveform": {
    "type": "pwl",
    "points": [
      [0, 0],
      [1e-3, 5],
      [2e-3, 5],
      [3e-3, 0]
    ]
  }
}
```

---

### Current Sources

#### DC Current Source

```json
{
  "type": "I",
  "name": "I1",
  "nodes": ["in", "out"],
  "value": 0.001
}
```

Current flows from first node to second node (conventional current direction).

#### Pulsed/Sinusoidal Current Sources

Same waveform options as voltage sources:

```json
{
  "type": "I",
  "name": "Ipulse",
  "nodes": ["in", "out"],
  "waveform": {
    "type": "pulse",
    "v1": 0,
    "v2": 1,
    "period": 1e-3,
    "pw": 0.5e-3
  }
}
```

---

### Switching Devices

#### Ideal Switch

```json
{
  "type": "SWITCH",
  "name": "S1",
  "nodes": ["node1", "node2"],
  "control": "ctrl_signal",
  "ron": 0.001,
  "roff": 1e9,
  "vth": 0.5
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `control` | string | - | required | Control signal name |
| `ron` | number | Ω | 0.001 | On-state resistance |
| `roff` | number | Ω | 1e9 | Off-state resistance |
| `vth` | number | V | 0.5 | Threshold voltage |

#### Ideal Diode

```json
{
  "type": "D",
  "name": "D1",
  "nodes": ["anode", "cathode"],
  "model": "ideal"
}
```

#### Shockley Diode

```json
{
  "type": "D",
  "name": "D1",
  "nodes": ["anode", "cathode"],
  "model": "shockley",
  "is": 1e-14,
  "n": 1.0,
  "rs": 0,
  "cjo": 0,
  "vj": 0.7,
  "m": 0.5,
  "tt": 0,
  "bv": 1000,
  "ibv": 1e-10
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `is` | number | A | 1e-14 | Saturation current |
| `n` | number | - | 1.0 | Emission coefficient |
| `rs` | number | Ω | 0 | Series resistance |
| `cjo` | number | F | 0 | Zero-bias junction capacitance |
| `vj` | number | V | 0.7 | Junction potential |
| `m` | number | - | 0.5 | Grading coefficient |
| `tt` | number | s | 0 | Transit time |
| `bv` | number | V | ∞ | Breakdown voltage |
| `ibv` | number | A | 1e-10 | Breakdown current |

---

### Semiconductor Devices

#### MOSFET (Level 1)

```json
{
  "type": "MOSFET",
  "name": "M1",
  "nodes": ["drain", "gate", "source", "body"],
  "model": "NMOS",
  "type": "n",
  "vth": 2.0,
  "kp": 20e-6,
  "lambda": 0.01,
  "w": 100e-6,
  "l": 10e-6,
  "cgs": 10e-12,
  "cgd": 5e-12,
  "cds": 2e-12
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `type` | string | - | "n" | "n" for NMOS, "p" for PMOS |
| `vth` | number | V | 2.0 | Threshold voltage |
| `kp` | number | A/V² | 20e-6 | Transconductance parameter |
| `lambda` | number | 1/V | 0.01 | Channel-length modulation |
| `w` | number | m | 100e-6 | Channel width |
| `l` | number | m | 10e-6 | Channel length |
| `cgs` | number | F | 0 | Gate-source capacitance |
| `cgd` | number | F | 0 | Gate-drain capacitance |
| `cds` | number | F | 0 | Drain-source capacitance |
| `rds_on` | number | Ω | - | On-state resistance (alternative model) |

#### Power MOSFET (Simplified)

For power electronics, use the simplified model with datasheet parameters:

```json
{
  "type": "MOSFET",
  "name": "Q1",
  "nodes": ["drain", "gate", "source", "0"],
  "model": "IRF540N",
  "rds_on": 0.044,
  "vth": 3.0,
  "ciss": 1700e-12,
  "coss": 310e-12,
  "crss": 60e-12,
  "qg": 71e-9,
  "body_diode": {
    "is": 1e-12,
    "n": 1.5,
    "trr": 150e-9
  }
}
```

#### IGBT

```json
{
  "type": "IGBT",
  "name": "Q1",
  "nodes": ["collector", "gate", "emitter"],
  "vce_sat": 1.5,
  "vth": 5.0,
  "rg": 10,
  "cies": 1000e-12,
  "coes": 50e-12,
  "cres": 10e-12,
  "td_on": 50e-9,
  "tr": 30e-9,
  "td_off": 100e-9,
  "tf": 50e-9
}
```

---

### Transformers

#### Ideal Transformer

```json
{
  "type": "TRANSFORMER",
  "name": "T1",
  "primary": ["p1", "p2"],
  "secondary": ["s1", "s2"],
  "turns_ratio": 10,
  "model": "ideal"
}
```

#### Transformer with Parasitics

```json
{
  "type": "TRANSFORMER",
  "name": "T1",
  "primary": ["p1", "p2"],
  "secondary": ["s1", "s2"],
  "turns_ratio": 10,
  "lm": 1e-3,
  "llk_pri": 1e-6,
  "llk_sec": 100e-9,
  "rp": 0.1,
  "rs": 0.001
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `turns_ratio` | number | - | required | N_primary / N_secondary |
| `lm` | number | H | ∞ | Magnetizing inductance |
| `llk_pri` | number | H | 0 | Primary leakage inductance |
| `llk_sec` | number | H | 0 | Secondary leakage inductance |
| `rp` | number | Ω | 0 | Primary winding resistance |
| `rs` | number | Ω | 0 | Secondary winding resistance |

---

## Waveform Sources

### PWM Source

Defined in the `sources` array:

```json
{
  "sources": [
    {
      "type": "PWM",
      "name": "PWM1",
      "frequency": 100e3,
      "duty": 0.5,
      "phase": 0,
      "dead_time": 100e-9,
      "inverted": false
    }
  ]
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `frequency` | number | Hz | required | Switching frequency |
| `duty` | number | - | 0.5 | Duty cycle (0-1) |
| `phase` | number | deg | 0 | Phase shift |
| `dead_time` | number | s | 0 | Dead time |
| `inverted` | boolean | - | false | Invert output |

### Reference in Components

```json
{
  "type": "SWITCH",
  "name": "S1",
  "nodes": ["in", "out"],
  "control": "PWM1"
}
```

---

## Simulation Options

### Transient Analysis

```json
{
  "simulation": {
    "type": "transient",
    "stop_time": 1e-3,
    "timestep": 1e-8,
    "max_timestep": 1e-6,
    "abstol": 1e-12,
    "reltol": 1e-3,
    "method": "trap"
  }
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `type` | string | - | required | "transient" |
| `stop_time` | number | s | required | End time |
| `timestep` | number/string | s | required | Time step or "auto" |
| `max_timestep` | number | s | - | Maximum adaptive timestep |
| `abstol` | number | - | 1e-12 | Absolute tolerance |
| `reltol` | number | - | 1e-3 | Relative tolerance |
| `method` | string | - | "trap" | "be" (Backward Euler) or "trap" (Trapezoidal) |

### DC Analysis

```json
{
  "simulation": {
    "type": "dc",
    "abstol": 1e-12,
    "reltol": 1e-3
  }
}
```

### AC Analysis

```json
{
  "simulation": {
    "type": "ac",
    "start_freq": 1,
    "stop_freq": 1e6,
    "points_per_decade": 20,
    "sweep_type": "decade"
  }
}
```

| Parameter | Type | Unit | Default | Description |
|-----------|------|------|---------|-------------|
| `start_freq` | number | Hz | required | Start frequency |
| `stop_freq` | number | Hz | required | Stop frequency |
| `points_per_decade` | number | - | 10 | Points per decade |
| `sweep_type` | string | - | "decade" | "decade", "linear", "octave" |

---

## Output Configuration

```json
{
  "output": {
    "format": "csv",
    "variables": ["V(out)", "I(L1)", "P(R1)"],
    "interval": 10,
    "file": "results.csv"
  }
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | string | "csv", "json", "hdf5", "parquet" |
| `variables` | array | List of variables to output |
| `interval` | number | Output every N timesteps |
| `file` | string | Output filename |

### Variable Syntax

| Syntax | Description |
|--------|-------------|
| `V(node)` | Voltage at node |
| `V(n1,n2)` | Voltage between nodes |
| `I(component)` | Current through component |
| `P(component)` | Power dissipation |
| `T(thermal_node)` | Temperature |

---

## Solver Options

```json
{
  "options": {
    "max_newton_iterations": 50,
    "gmin": 1e-12,
    "gmin_stepping": true,
    "source_stepping": false,
    "pivot_threshold": 0.01,
    "sparse_solver": "klu"
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_newton_iterations` | number | 50 | Maximum Newton iterations |
| `gmin` | number | 1e-12 | Minimum conductance |
| `gmin_stepping` | boolean | false | Enable Gmin stepping for convergence |
| `source_stepping` | boolean | false | Enable source stepping |
| `pivot_threshold` | number | 0.01 | Pivot threshold for LU |
| `sparse_solver` | string | "eigen" | "eigen" or "klu" |

---

## Complete Example

```json
{
  "name": "Buck Converter with Thermal Model",
  "description": "Synchronous buck converter with MOSFET thermal modeling",

  "components": [
    {"type": "V", "name": "Vin", "nodes": ["vin", "0"], "value": 12},

    {"type": "MOSFET", "name": "Q_high", "nodes": ["vin", "gh", "sw", "0"],
     "model": "IRF540N", "rds_on": 0.044,
     "thermal": {"node": "Tj_high", "rth_jc": 0.5}},

    {"type": "MOSFET", "name": "Q_low", "nodes": ["sw", "gl", "0", "0"],
     "model": "IRF540N", "rds_on": 0.044,
     "thermal": {"node": "Tj_low", "rth_jc": 0.5}},

    {"type": "L", "name": "L1", "nodes": ["sw", "vout"], "value": 100e-6, "ic": 0},
    {"type": "C", "name": "C1", "nodes": ["vout", "0"], "value": 100e-6, "ic": 5},
    {"type": "R", "name": "R_load", "nodes": ["vout", "0"], "value": 2.5}
  ],

  "sources": [
    {"type": "PWM", "name": "gh", "frequency": 100e3, "duty": 0.417},
    {"type": "PWM", "name": "gl", "frequency": 100e3, "duty": 0.417,
     "inverted": true, "dead_time": 100e-9}
  ],

  "thermal": {
    "ambient": 25,
    "nodes": [
      {"name": "Tj_high", "heatsink_rth": 2.0},
      {"name": "Tj_low", "heatsink_rth": 2.0}
    ]
  },

  "simulation": {
    "type": "transient",
    "stop_time": 1e-3,
    "timestep": 1e-9,
    "method": "trap"
  },

  "output": {
    "format": "hdf5",
    "variables": ["V(vout)", "I(L1)", "T(Tj_high)", "T(Tj_low)"],
    "interval": 100
  },

  "options": {
    "abstol": 1e-12,
    "reltol": 1e-3,
    "gmin_stepping": true
  }
}
```
