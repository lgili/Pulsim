# Migration Guide: JSON → YAML and Unified v1 Core

This guide summarizes the changes introduced by the unified v1 core and the YAML netlist format.

## 1) Netlist Format: JSON → YAML

Pulsim now uses **versioned YAML netlists**. JSON netlists are no longer supported by the loader.

### Required top-level fields

```yaml
schema: pulsim-v1
version: 1
components:
  - type: resistor
    name: R1
    nodes: [in, out]
    value: 1k
```

### Waveform example

```yaml
- type: voltage_source
  name: Vpwm
  nodes: [ctrl, 0]
  waveform:
    type: pwm
    v_high: 10.0
    v_low: 0.0
    frequency: 20000
    duty: 0.5
    dead_time: 0.0
```

### Model reuse

```yaml
models:
  m1:
    type: mosfet
    vth: 3.0
    kp: 0.02

components:
  - type: mosfet
    name: Q1
    nodes: [d, g, s, 0]
    use: m1
    kp: 0.03  # local override
```

## 2) Simulation API: Unified v1 Core

Python and CLI flows now execute through `pulsim::v1::Simulator`.

### C++ usage

```cpp
#include <pulsim/v1/core.hpp>

pulsim::v1::parser::YamlParser parser;
auto [circuit, options] = parser.load("circuit.yaml");

pulsim::v1::Simulator sim(circuit, options);
auto result = sim.run_transient();
```

### Python usage (programmatic circuits)

```python
import pulsim as sl

circuit = sl.Circuit()
circuit.add_voltage_source("V1", "in", "0", 5.0)
circuit.add_resistor("R1", "in", "out", 1000.0)
circuit.add_capacitor("C1", "out", "0", 1e-6)

opts = sl.SimulationOptions()
opts.tstop = 1e-3
opts.dt = 1e-6

sim = sl.Simulator(circuit, opts)
result = sim.run_transient()
```

## 3) Behavior Changes

- **Event detection**: switch events are bisected to refine timing.
- **Loss accumulation**: conduction and switching losses are tracked per device and summarized in results.
- **Determinism**: fixed ordering and explicit configuration improve reproducibility.

## 4) Common YAML Changes

| JSON field | YAML field | Notes |
| --- | --- | --- |
| `name` | `name` | unchanged |
| `components` | `components` | list of component maps |
| `waveform.type` | `waveform.type` | supports `dc`, `pulse`, `sine`, `pwm` |
| `model` | `use` | model inheritance + overrides |
| `value` | `value` | supports SI suffixes (`1k`, `10u`) |

## 5) Troubleshooting

- **Missing `schema`/`version`**: YAML parser will reject the netlist.
- **Unknown fields**: strict validation rejects unsupported keys.
- **Convergence**: reduce timestep or enable adaptive settings if needed.
