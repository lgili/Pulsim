# Design — `pulsim-v2-yaml-parser` (Layer 8 V0)

## Schema

```yaml
# Top-level: `circuit` (required) and `simulation` (optional).
circuit:
  # Optional: explicit node declarations. Auto-created from
  # device terminal references if omitted.
  nodes:
    - vin
    - sw
    - vout

  devices:
    - type: voltage_source
      name: Vin             # optional, used in error messages
      from: vin
      to: gnd
      V: 24.0

    - type: resistor
      name: R1
      from: vin
      to: out
      R: 100.0              # ohms

    - type: capacitor
      name: C1
      from: out
      to: gnd
      C: 1e-6               # farads

    - type: inductor
      name: L1
      from: sw
      to: vout
      L: 100e-6             # henries

    - type: diode
      name: D1
      anode: gnd
      cathode: sw
      g_on: 1e3
      g_off: 1e-9
      V_th: 0.7

    - type: nonlinear_diode
      name: D2
      anode: a
      cathode: c
      V_F0: 0.7
      R_d: 0.01
      G_off: 1e-9
      kappa: 20.0

    - type: switch
      name: SW1
      from: vin
      to: sw
      g_on: 1e3
      g_off: 1e-9

    - type: mosfet
      name: Q1
      drain: vin
      source: sw
      R_on: 1e-3            # default 1 mΩ
      R_off: 1e9            # default 1 GΩ

    - type: mosfet_with_body_diode
      name: Q2
      drain: sw
      source: gnd
      R_on: 1e-3
      R_off: 1e9
      V_F: 0.7              # body diode V_F

    - type: igbt
      name: T1
      collector: vin
      emitter: sw
      R_on: 10e-3
      R_off: 1e9

    - type: transformer
      name: T1
      p_from: pri+
      p_to: pri-
      s_from: sec+
      s_to: sec-
      L_p: 100e-6
      L_s: 25e-6
      k: 0.95               # default 1.0

# Optional simulation block.
simulation:
  t_start: 0.0
  t_end: 1e-3
  dt: 1e-7
  enable_newton_line_search: false      # default
  enable_newton_lm: false               # default
  enable_substep_state_correction: false
  max_event_iterations: 16
  max_newton_iterations: 50
  tol_newton_dx: 1e-9
  tol_newton_res: 1e-9
```

## API

```cpp
namespace pulsim::v2::yaml {

struct LoadedCircuit {
    builder::CircuitBuilder builder;
    solver::SimulationOptions options;
};

/// Parse a YAML file from disk.
[[nodiscard]] LoadedCircuit load_file(const std::string& path);

/// Parse a YAML string in memory.
[[nodiscard]] LoadedCircuit load_string(const std::string& yaml_text);

}  // namespace pulsim::v2::yaml
```

Python equivalent:

```python
loaded = pulsim.v2.load_yaml_file("buck.yaml")
loaded = pulsim.v2.load_yaml_string(yaml_text)

# loaded.builder, loaded.options as usual.
```

## Validation rules

Required fields are validated PER DEVICE TYPE. Missing
fields raise `std::runtime_error` with:
- The device's `name` if provided, otherwise the device's
  index in the YAML's `devices:` list.
- The missing field name.
- A short suggestion of the required schema.

Example: a `resistor` without `R:`:
```
yaml::load_file: device 'R1' (resistor) is missing required
field 'R'. Required fields for resistor: from, to, R.
```

Optional fields fall through to the builder's defaults (R_on,
R_off, k, etc.).

## Schema design choices

- **Single device type per entry**: each YAML list item has
  exactly one `type` field. Avoids
  `devices: { resistors: [...], capacitors: [...], … }`-style
  schemas that hide ordering.

- **String node names everywhere**: never integer indices.
  Matches `CircuitBuilder`'s ergonomic convention.

- **SI units throughout**: ohms, farads, henries, volts,
  amperes — no implicit conversions. Matches Layer 6.

- **Names are diagnostic-only**: V0 doesn't index devices
  by name (no `get_device_by_name`); names appear in error
  messages and YAML readability only. V1 may add a lookup.

- **Optional sections**: `circuit.nodes` is optional (auto-
  created by device references); `simulation:` is optional
  (caller may run their own simulation).

## Error policy

V0 throws on:
- Missing required fields per device.
- Unknown `type:` value.
- Malformed YAML (yaml-cpp throws; we re-throw with file
  path context).

V0 does NOT throw on:
- Extra unknown fields (silently ignored — forward-compatible
  for V1 schema extensions).
- Zero or negative values (the kernel handles those; the YAML
  parser is just a translator).

## Sample YAMLs

`examples/v2/half_wave_rectifier.yaml`:

```yaml
circuit:
  devices:
    - type: voltage_source
      name: Vin
      from: n0
      to: gnd
      V: 0.0                # baseline; modulated via b_extra
                            # at runtime
    - type: diode
      name: D1
      anode: n0
      cathode: n1
      g_on: 1e3
      g_off: 1e-9
      V_th: 0.0
    - type: resistor
      name: R_L
      from: n1
      to: gnd
      R: 10.0

simulation:
  t_start: 0.0
  t_end: 0.0333             # 2 cycles at 60 Hz
  dt: 1e-4
```

`examples/v2/buck.yaml`:

```yaml
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
    - type: diode
      name: D1
      anode: gnd
      cathode: sw
      g_on: 1e3
      g_off: 1e-9
      V_th: 0.7
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

`examples/v2/flyback.yaml`:

```yaml
circuit:
  devices:
    - type: voltage_source
      name: Vin
      from: vin
      to: gnd
      V: 48.0
    - type: mosfet
      name: Q1
      drain: sw
      source: gnd
      R_on: 100e-3
    - type: transformer
      name: T1
      p_from: vin
      p_to: sw
      s_from: sec_anode
      s_to: gnd
      L_p: 100e-6
      L_s: 25e-6
      k: 0.95
    - type: diode
      name: D1
      anode: sec_anode
      cathode: vout
      g_on: 1e3
      g_off: 1e-9
      V_th: 0.7
    - type: capacitor
      name: Cout
      from: vout
      to: gnd
      C: 100e-6
    - type: resistor
      name: R_L
      from: vout
      to: gnd
      R: 5.0

simulation:
  t_start: 0.0
  t_end: 2e-4
  dt: 1e-8
```

## Test plan

`core/tests/v2/yaml/test_yaml_loader.cpp`:

1. **Empty circuit throws** (a YAML with empty `devices` is
   non-functional).
2. **Each device type round-trips**: for each of the 11
   device types, build a single-device circuit via YAML
   string and verify the resulting builder has 1 (or 2
   for MOSFET+body, transformer) branch(es) with the
   correct pool entry.
3. **simulation: block populates SimulationOptions**.
4. **Missing required field throws** with the device's
   name in the message.
5. **Unknown device type throws** with the unknown name.
6. **Integration**: load buck.yaml, run the simulation,
   verify the result has expected properties (cap voltage
   tracks output, no NaN).
7. **Direct vs YAML equivalence**: build half-wave
   rectifier via direct builder and via YAML; verify
   sample-by-sample match.

`python/tests/v2/test_v2_python_bindings.py`:

8. **`load_yaml_string` returns a builder**: build a
   simple circuit from a YAML string and verify
   `num_branches`.
9. **`load_yaml_file` reads disk**: load
   `examples/v2/buck.yaml` and run.

## What V0 deliberately does NOT do

- **Parameter expressions** (e.g. `R: ${R_L * 2}`): V0
  takes only literal numbers. V1 may add expression
  evaluation.
- **`include:` directive** to compose YAML files: V1.
- **Device-by-name lookup** at runtime: names are
  diagnostic-only.
- **Schema versioning**: V0 has no `version:` field.
  Schema changes in V1 will need migration tooling.
- **Symbolic units** (e.g. `R: "10 ohms"`): only raw
  numbers. SPICE-style suffixes (k, m, u, n, p) are V1.
- **Schedule definition** for switch_fn / b_extra_fn:
  the YAML defines the circuit + sim options only. The
  user supplies switch_fn / b_extra_fn from code (C++ or
  Python).

## Files

- NEW `core/include/pulsim/v2/yaml/loader.hpp`
- NEW `core/tests/v2/yaml/test_main.cpp`
- NEW `core/tests/v2/yaml/test_yaml_loader.cpp`
- MODIFIED `core/CMakeLists.txt` (yaml test target)
- MODIFIED `python/bindings_v2_kernel.cpp`
- MODIFIED `python/tests/v2/test_v2_python_bindings.py`
- NEW `examples/v2/half_wave_rectifier.yaml`
- NEW `examples/v2/buck.yaml`
- NEW `examples/v2/flyback.yaml`
- NEW `docs/pulsim-v2/layer8-yaml-parser.md`
