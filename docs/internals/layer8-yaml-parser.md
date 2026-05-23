# Layer 8 — YAML circuit parser

The full v2 stack (CircuitBuilder + power-device helpers +
Python bindings) supported coding in C++ or Python. V8 closes
the loop by adding a **YAML loader**: engineers describe a
circuit and simulation options in a config file, the loader
returns a populated `CircuitBuilder` + `SimulationOptions`
ready to feed into `PwlStateSpaceCache` and `run_transient`.

```yaml
# my_buck.yaml
circuit:
  devices:
    - type: voltage_source
      name: Vin
      from: vin
      to: gnd
      V: 24.0
    - type: mosfet_with_body_diode
      name: Q1
      drain: vin
      source: sw
    - type: inductor
      name: L1
      from: sw
      to: vout
      L: 100e-6
    - type: capacitor
      name: Cout
      from: vout
      to: gnd
      C: 47e-6
    - type: resistor
      name: R_L
      from: vout
      to: gnd
      R: 5.0

simulation:
  t_start: 0.0
  t_end: 5e-3
  dt: 1e-7
```

```python
import pulsim as p

loaded = p.load_yaml_file("my_buck.yaml")
cache = p.PwlStateSpaceCache(loaded.builder.graph,
                              loaded.builder.pool)
cache.build(loaded.options.dt)

result = p.run_transient(
    cache, loaded.builder.graph, loaded.builder.pool,
    loaded.options,
    switch_fn=lambda t: p.SwitchStateMask(0),
)
print(result.states[-1])
```

## Supported device types

The YAML loader supports every Layer 2 V2 device. Each
device entry's `type:` field selects the schema:

| `type` | Required fields | Optional |
|--------|-----------------|----------|
| `voltage_source` | `from`, `to`, `V` | — |
| `resistor` | `from`, `to`, `R` (Ω) | — |
| `capacitor` | `from`, `to`, `C` (F) | — |
| `inductor` | `from`, `to`, `L` (H) | — |
| `diode` | `anode`, `cathode`, `g_on`, `g_off` | `V_th` (default 0) |
| `nonlinear_diode` | `anode`, `cathode` | `V_F0` (0.7), `R_d` (0.01), `G_off` (1e-9), `kappa` (20) |
| `switch` | `from`, `to`, `g_on`, `g_off` | — |
| `mosfet` | `drain`, `source` | `R_on` (1mΩ), `R_off` (1GΩ) |
| `mosfet_with_body_diode` | `drain`, `source` | `R_on` (1mΩ), `R_off` (1GΩ), `V_F` (0.7), `g_on_diode` (1k), `g_off_diode` (1n) |
| `igbt` | `collector`, `emitter` | `R_on` (10mΩ), `R_off` (1GΩ) |
| `transformer` | `p_from`, `p_to`, `s_from`, `s_to`, `L_p`, `L_s` | `k` (1.0) |

`name:` is optional on every device but recommended for
clearer error messages.

`"gnd"` / `"GND"` / `"0"` are aliases for the ground node
(same as Layer 6's `CircuitBuilder`).

## `simulation:` block

All fields optional; defaults match `SimulationOptions{}`:

```yaml
simulation:
  t_start: 0.0
  t_end:   1e-3
  dt:      1e-7
  enable_newton_line_search: false
  enable_newton_lm: false
  enable_substep_state_correction: false
  max_event_iterations: 16
  max_newton_iterations: 50
  tol_newton_dx: 1e-9
  tol_newton_res: 1e-9
```

## API

### C++

```cpp
#include "pulsim/yaml/loader.hpp"

auto loaded = pulsim::yaml::load_file("my_circuit.yaml");
// loaded.builder, loaded.options
```

### Python

```python
import pulsim as p

loaded = p.load_yaml_file("path/to/circuit.yaml")
# loaded.builder, loaded.options

# Or from an in-memory string:
loaded = p.load_yaml_string("""
circuit:
  devices:
    - type: resistor
      name: R1
      from: a
      to: b
      R: 100.0
""")
```

## Error handling

The loader throws `std::runtime_error` (in C++) /
`RuntimeError` (in Python) on:

- Missing required fields per device (e.g. a resistor
  without `R:`).
- Unknown `type:` value.
- Malformed YAML (re-wrapped from yaml-cpp's parse errors).
- Missing top-level `circuit:` or `circuit.devices:`.

Errors mention the device's `name:` if provided, otherwise
the device's index in the YAML's `devices:` list, plus the
missing field name:

```
yaml::load: device 'R_load' (resistor) is missing required
field 'R'
```

## Sample YAMLs

Three working examples ship in `examples/`:

- `half_wave_rectifier.yaml` — V_sine source + diode + R_L.
- `buck.yaml` — synchronous buck with high-side MOSFET +
  body diode, low-side freewheeling diode, LC output filter.
- `flyback.yaml` — isolated flyback with a 2-winding
  transformer (k=0.95), MOSFET, rectifier diode, output
  cap.

These are starting points — engineers iterate by editing
the YAML, no recompile / reimport needed.

## What V0 deliberately does NOT do

- **Parameter expressions / formulae** (e.g.
  `R: ${R_load * 2}`). V0 takes literal numbers only. V1
  may add expression evaluation.
- **`include:` directive** for composing YAMLs. V1.
- **Device-by-name lookup** at runtime. Names are
  diagnostic-only in V0.
- **Schema versioning** (`version:` field). Forward
  compatibility lives in "extra unknown fields are
  silently ignored" — V1 may add explicit version
  negotiation.
- **Symbolic units** (`R: "10 kΩ"`). Only raw numbers in
  V0. SPICE-style suffixes (k, m, u, n, p) are V1.
- **Schedule (switch_fn / b_extra_fn) definition** inside
  the YAML. The user supplies these from C++ or Python.
  V1 may add `schedule:` blocks (e.g. PWM frequency,
  duty cycle) that generate switch_fn callbacks.

## Test coverage

**C++** — 17 cases in
`core/tests/v2/yaml/test_yaml_loader.cpp`:

- Round-trip per device type (10 cases).
- `simulation:` block parsing.
- Missing-field errors (with device name in message).
- Unknown-type errors.
- Malformed YAML errors.
- Missing top-level `circuit:` throws.
- Direct vs YAML equivalence on half-wave rectifier.
- DC solve roundtrip via YAML.

**Python** — 4 cases in
`python/tests/v2/test_v2_python_bindings.py`:

- `load_yaml_string` basic.
- `load_yaml_string` missing-field error.
- `load_yaml_file` reads disk + parses the half-wave
  example.
- `load_yaml_file` reads buck.yaml + verifies branch count.

## Files

- NEW `core/include/pulsim/yaml/loader.hpp`
- NEW `core/tests/v2/yaml/test_main.cpp`
- NEW `core/tests/v2/yaml/test_yaml_loader.cpp`
- MODIFIED `core/CMakeLists.txt` (yaml test target)
- MODIFIED `python/bindings_v2_kernel.cpp`
- MODIFIED `python/pulsim/v2.py` (re-export YAML symbols)
- MODIFIED `python/tests/v2/test_v2_python_bindings.py`
  (+ 4 cases)
- NEW `examples/half_wave_rectifier.yaml`
- NEW `examples/buck.yaml`
- NEW `examples/flyback.yaml`
- NEW `openspec/changes/pulsim-v2-yaml-parser/`
