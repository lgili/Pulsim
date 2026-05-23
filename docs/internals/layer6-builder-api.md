# Layer 6 — CircuitBuilder API

V2's kernel layer (`Graph`, `DevicePool`,
`PwlStateSpaceCache`, `run_transient`) is the well-tested
numerical core. Its public surface is intentionally low-
level: raw `Index` values, conductance / susceptance
parameters, explicit branch numbering. That's the right
design for the kernel but a poor interface for user code.

V6 ships `CircuitBuilder` — a thin wrapper that hides the
two-object `Graph + DevicePool` split and exposes string
node names, SI-unit parameters, and named devices.

## API

```cpp
#include "pulsim/builder/circuit_builder.hpp"

using pulsim::builder::CircuitBuilder;

CircuitBuilder b;
b.add_voltage_source("Vin", "n0", "gnd", 5.0);
b.add_resistor      ("R1",  "n0", "n1", 100.0);  // ohms
b.add_capacitor     ("C1",  "n1", "gnd", 1e-6);  // farads
b.add_inductor      ("L1",  "n1", "n2", 1e-3);   // henries
b.add_diode         ("D1",  "n2", "gnd",
                      /*g_on=*/1e3, /*g_off=*/1e-9,
                      /*V_th=*/0.7);

// Hand the built circuit to the kernel.
PwlStateSpaceCache cache(b.graph(), b.pool());
cache.build(dt);
```

## Behaviour

### Node mapping

| User name | Index |
|-----------|-------|
| `"gnd"`, `"GND"`, `"0"` | `graph().ground()` |
| Any other name | Auto-created on first use. Subsequent uses return the cached index. |

Case-sensitive: `"n0"` and `"N0"` are distinct.

### Unit conversion

| User param | Kernel param |
|------------|--------------|
| `R_ohms` | `Resistor::Params{ .G = 1/R_ohms }` |
| `C_farads` | `Capacitor::Params{ C_farads }` |
| `L_henries` | `Inductor::Params{ L_henries }` |
| `V` | `VoltageSource::Params{ V }` |

### BranchKind dispatch

| Device | BranchKind |
|--------|------------|
| voltage source | `Source` |
| resistor / capacitor / inductor | `PassiveLinear` |
| switched diode | `Switch` |
| nonlinear diode | `Nonlinear` |
| switch | `Switch` |

### Method chaining

All `add_*` methods return `CircuitBuilder&`. Chain or
plain sequential calls — both idiomatic.

```cpp
b.add_voltage_source("Vin", "n0", "gnd", 5.0)
 .add_resistor      ("R1",  "n0", "n1", 100.0)
 .add_capacitor     ("C1",  "n1", "gnd", 1e-6);
```

## Accessors

```cpp
const Graph&       graph();          // const ref, owned by builder
const DevicePool&  pool();           // const ref, owned by builder
Size               num_branches();   // total branches added
Index              node_id_of(name); // throws on unknown name
```

The builder owns its `Graph` and `DevicePool`. Caller must
keep the builder alive while using the const refs (a
typical lifetime: builder is a local of the `main` /
`TEST_CASE` body, alongside the cache and the
`SimulationResult`).

## Comparison: manual vs builder

### Manual (kernel API)

```cpp
Graph g;
g.add_node("n0");
g.add_node("n1");
g.add_branch(0, g.ground(), BranchKind::Source);
g.add_branch(0, 1,          BranchKind::Switch);
g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

DevicePool pool;
pool.add_voltage_source(0, {.V = 0.0});
pool.add_diode(1, /*g_on=*/1e3, /*g_off=*/1e-9, /*V_th=*/0.0);
pool.add_resistor(2, {.G = 1.0 / 10.0});

PwlStateSpaceCache cache(g, pool);
```

### Builder (Layer 6)

```cpp
CircuitBuilder b;
b.add_voltage_source("Vin", "n0", "gnd", 0.0)
 .add_diode         ("D1",  "n0", "n1", 1e3, 1e-9, 0.0)
 .add_resistor      ("R_L", "n1", "gnd", 10.0);

PwlStateSpaceCache cache(b.graph(), b.pool());
```

Same circuit. Sample-for-sample bit-identical output
through `run_transient` (verified by the half-wave
rectifier parity test).

## What V0 deliberately does NOT do

- **YAML / JSON parser**: V0 is C++ only. A YAML layer
  that constructs a builder from a config file is V7.
- **Validation**: V0 trusts the user. Duplicate device
  names, dangling branches, zero or negative parameter
  values — all accepted. V1 adds validation.
- **Python bindings**: V0 is C++ only. V8 wraps the
  builder via nanobind.
- **Subcircuit / hierarchical composition**: V0 is flat.
  A "compose builders" API is V1.

## Test coverage

11 test cases in
`core/tests/v2/builder/test_circuit_builder.cpp`:

- Node alias: `"gnd"` / `"GND"` / `"0"` → ground.
- Node lookup: creates on first use, reuses on second.
- `node_id_of` throws on unknown name.
- Implicit node creation via device methods.
- Unit conversion correctness for R, C, L.
- Method chaining works.
- Round-trip V_dc + R: same answer as manual setup.
- Half-wave rectifier: builder-built circuit produces
  sample-by-sample bit-identical output to the manual
  setup through `run_transient`.
- Nonlinear diode: DC load-line via the builder + Newton
  solve.

## Files

- NEW `core/include/pulsim/builder/circuit_builder.hpp`
- NEW `core/tests/v2/builder/test_main.cpp`
- NEW `core/tests/v2/builder/test_circuit_builder.cpp`
- MODIFIED `core/CMakeLists.txt`
  (added `pulsim_v2_builder_tests` target)
