## Why

v2's full surface (`CircuitBuilder`, MOSFET / transformer
helpers, Python bindings) requires the user to write code in
either C++ or Python. For many SMPS workflows, an engineer
just wants to iterate on circuit parameters: change `R_load`
from 10Ω to 5Ω, sweep `dt` from 100ns to 10ns, swap a
diode model — without recompiling C++ or even touching
Python.

V8 ships a **YAML loader** that reads a circuit description
+ simulation options from a file and constructs a fully-
populated `CircuitBuilder` + `SimulationOptions`. The
engineer's iteration loop becomes:

```bash
$ vim my_buck.yaml          # edit parameters
$ python -m my_runner my_buck.yaml
```

Same parser surfaces from C++:

```cpp
auto loaded = pulsim::v2::yaml::load_file("my_buck.yaml");
PwlStateSpaceCache cache(loaded.builder.graph(),
                          loaded.builder.pool());
cache.build(loaded.options.dt);
auto result = run_transient(...);
```

This is the final lap of "v2 is usable by real engineers" —
no compiler, no Python, just YAML + CLI.

## What Changes

**Scope decision — Layer 8 V0** (YAML parser):

- New header `core/include/pulsim/v2/yaml/loader.hpp`
  with:
  - `yaml::LoadedCircuit { CircuitBuilder builder;
      SimulationOptions options; }`
  - `yaml::load_file(path) → LoadedCircuit`
  - `yaml::load_string(yaml_text) → LoadedCircuit`

- YAML schema supports every Layer 2 V2 device:
  - `voltage_source` (V)
  - `resistor` (R in ohms)
  - `capacitor` (C in farads)
  - `inductor` (L in henries)
  - `diode` (g_on, g_off, V_th)
  - `nonlinear_diode` (V_F0, R_d, G_off, kappa)
  - `switch` (g_on, g_off)
  - `mosfet` (R_on, R_off)
  - `mosfet_with_body_diode` (R_on, R_off, V_F)
  - `igbt` (R_on, R_off)
  - `transformer` (L_p, L_s, k)

- Optional `simulation:` block configures `t_start`, `t_end`,
  `dt`, Newton globalization flags, event-iteration limits.

- Python bindings: `pulsim.v2.load_yaml_file(path)` and
  `pulsim.v2.load_yaml_string(text)`.

- Validation: missing required fields (e.g. `from`, `to`)
  throw with a clear message including the device name +
  YAML line number.

- Sample YAMLs in `examples/v2/`:
  - `buck.yaml`
  - `boost.yaml`
  - `half_wave_rectifier.yaml`
  - `flyback.yaml`

- Tests:
  - Unit: round-trip each device type via YAML matches the
    direct-builder version.
  - Integration: load buck.yaml, run a simulation, verify
    sample-by-sample match with the manual builder version.
  - Python: load YAML from disk and run a full simulation.
  - Error: missing field throws with the device name in
    the message.

## Impact

- **Affected specs**: ADDED requirement on
  `kernel-v2-solver` for `pulsim::v2::yaml` loader.
- **Affected code** (~500 LOC):
  - NEW `core/include/pulsim/v2/yaml/loader.hpp`
  - NEW `core/tests/v2/yaml/test_yaml_loader.cpp`
  - NEW `core/tests/v2/yaml/test_main.cpp`
  - MODIFIED `core/CMakeLists.txt` (add yaml test target
    linking yaml-cpp)
  - MODIFIED `python/bindings_v2_kernel.cpp` (+ Python
    bindings)
  - MODIFIED `python/tests/v2/test_v2_python_bindings.py`
    (+ YAML tests)
  - NEW sample YAMLs in `examples/v2/`
  - NEW `docs/pulsim-v2/layer8-yaml-parser.md`
- **Migration**: zero. Pure additive.
- **Risk**: medium. The schema is the public contract —
  any rename in V1 is a breaking change. V0 keeps the
  surface intentionally small.
