# Design — `pulsim-v2-builder-api` (Layer 6 V0)

## The two-layer split

v2's KERNEL layer (`Graph`, `DevicePool`, `PwlStateSpaceCache`,
`run_transient`) is the carefully-engineered numerical core.
It accepts low-level inputs (raw indices, conductance values,
`Eigen::SparseMatrix`) for performance and clarity inside the
solver. That's the right design — the kernel is what we test
exhaustively against analytical answers.

But it's NOT the right interface for users building a
circuit. V6 adds a thin BUILDER layer on top:

```
+--------------------------------------+
| User code (tests, apps, Python bindings) |
+--------------------------------------+
              ↓
+--------------------------------------+
| LAYER 6: CircuitBuilder              |
|   - string node names                |
|   - ohms / farads / henries          |
|   - implicit node creation           |
|   - "gnd" alias                      |
+--------------------------------------+
              ↓ delegates to
+--------------------------------------+
| KERNEL: Graph + DevicePool (Layer 1/2) |
|   - raw Index values                 |
|   - conductance / susceptance        |
|   - explicit branch numbering        |
+--------------------------------------+
```

The builder NEVER does numerical work. It's pure
bookkeeping + unit conversion + the string→Index map.

## API

```cpp
namespace pulsim::v2::builder {

class CircuitBuilder {
public:
    /// Look up or create a node by name. Returns the node's
    /// integer index in the underlying Graph.
    /// "gnd" / "GND" / "0" → graph.ground().
    Index node(std::string name);

    /// Add a constant voltage source from `from` to `to`.
    /// Internal: graph.add_branch(from_idx, to_idx, Source);
    ///           pool.add_voltage_source(branch_id, {.V = V}).
    CircuitBuilder& add_voltage_source(
        std::string name, std::string from, std::string to,
        Real V);

    /// Add a linear resistor with resistance R (in OHMS).
    /// Internal converts R → G = 1/R.
    CircuitBuilder& add_resistor(
        std::string name, std::string from, std::string to,
        Real R_ohms);

    /// Add a linear capacitor with capacitance C (in FARADS).
    CircuitBuilder& add_capacitor(
        std::string name, std::string from, std::string to,
        Real C_farads);

    /// Add a linear inductor with inductance L (in HENRIES).
    CircuitBuilder& add_inductor(
        std::string name, std::string from, std::string to,
        Real L_henries);

    /// Add a binary switched diode (Layer 5 V2's
    /// SwitchedDiode model). `g_on` / `g_off` in siemens;
    /// `V_th` in volts.
    CircuitBuilder& add_diode(
        std::string name, std::string anode,
        std::string cathode,
        Real g_on, Real g_off, Real V_th);

    /// Add a smooth-blend IdealDiode (Layer 4 V3's
    /// AD-driven nonlinear model).
    CircuitBuilder& add_nonlinear_diode(
        std::string name, std::string anode,
        std::string cathode,
        models::IdealDiode::Params params);

    /// Add an IdealSwitch (controlled by switch_fn at
    /// simulation time).
    CircuitBuilder& add_switch(
        std::string name, std::string from, std::string to,
        Real g_on, Real g_off);

    /// Accessors. The caller must keep the builder alive
    /// while using these refs.
    [[nodiscard]] const topology::Graph& graph() const noexcept;
    [[nodiscard]] const pwl::DevicePool& pool() const noexcept;

    /// Total number of branches added.
    [[nodiscard]] Size num_branches() const noexcept;

    /// Returns the node index for `name`. Throws
    /// `std::out_of_range` if the name was never registered
    /// (use this for verification, not for lookups inside
    /// a build loop — the device methods auto-create).
    [[nodiscard]] Index node_id_of(
        const std::string& name) const;

private:
    Index resolve_node_(const std::string& name);

    topology::Graph                                 graph_;
    pwl::DevicePool                                 pool_;
    std::unordered_map<std::string, Index>          node_map_;
};

}  // namespace pulsim::v2::builder
```

## Behaviour spec

### Node mapping

| User name | Index |
|-----------|-------|
| `"gnd"`, `"GND"`, `"0"` | `graph.ground()` |
| Any other name | Auto-created on first use; cached in `node_map_`. Subsequent uses return the cached index. |

Case-sensitive: `"n0"` and `"N0"` are distinct nodes.

### BranchKind mapping

The builder picks the kernel's BranchKind based on the
device type:

| Device | BranchKind |
|--------|------------|
| voltage source | `Source` |
| resistor | `PassiveLinear` |
| capacitor | `PassiveLinear` |
| inductor | `PassiveLinear` |
| diode (switched) | `Switch` |
| nonlinear diode | `Nonlinear` |
| switch | `Switch` |

### Unit conventions

| User param | Kernel param |
|------------|--------------|
| `R_ohms` | `Resistor::Params{ .G = 1 / R_ohms }` |
| `C_farads` | `Capacitor::Params{ .C = C_farads }` |
| `L_henries` | `Inductor::Params{ .L = L_henries }` |
| `V` | `VoltageSource::Params{ .V = V }` |
| `g_on, g_off, V_th` | passed verbatim to `SwitchedDiode` |
| `params` (NL diode) | passed verbatim to `IdealDiode::Params` |
| `g_on, g_off` | passed verbatim to `SwitchParams` |

### Error handling (V0)

- Looking up a non-existent name via `node_id_of` throws
  `std::out_of_range`.
- All other input is accepted as-is. V0 does NOT detect:
  - duplicate device names
  - degenerate values (R = 0, C < 0, etc.)
  - dangling branches
  These are V1 add-ons.

## Why not just a fluent factory?

We considered:
```cpp
auto circuit = pulsim::circuit()
    .with_node("n0")
    .with_voltage_source("Vin", "n0", "gnd", 5.0)
    .with_resistor("R1", "n0", "n1", 100.0)
    .build();
```

This requires templated `Builder<NodeList, BranchList>` to
return the right type, or a single mutable builder where
each method returns `*this`. We picked the latter (less
template machinery, less compile time).

Method-chaining is OPTIONAL: each `add_*` returns
`CircuitBuilder&` so the user CAN chain, but plain
sequential calls are equally idiomatic.

## Test plan

In `core/tests/v2/builder/`:

1. **Node alias `"gnd"` → ground**.
2. **Implicit node creation**: `add_voltage_source("V",
   "a", "b", ...)` should create nodes "a" and "b" if not
   pre-declared.
3. **Round-trip V_dc circuit**: build V=5V + R(1Ω) →
   ground, cache.build(), cache.solve() should give
   v_node = 5V (the V3 sanity check).
4. **Half-wave rectifier parity**: build the V2 layer5_v2
   half-wave rectifier via the builder; run `run_transient`;
   verify v_n1 sequence matches the manual-setup test
   sample-by-sample within 1 µV.
5. **Buck converter parity**: build the V1.5 buck
   converter via the builder; verify peak inductor current
   matches manual setup.
6. **Resistor unit conversion**: assert `add_resistor("R",
   ..., 100.0)` produces `pool.resistor_params(branch_id)
   .G == 1.0 / 100.0` within FP tolerance.

## What V0 deliberately does NOT do

- **YAML parser**: V0 ships only the C++ builder API. A
  YAML / JSON layer that constructs a builder is V7.
- **Validation**: V0 trusts the user. Duplicate device
  names, dangling branches, sign-wrong values — all
  accepted. V1 ships validation.
- **Python bindings**: V0 is C++ only. Python bindings
  wrap the builder in V8.
- **Subcircuit / hierarchical composition**: V0 is flat.
  A "compose two circuits" API is V1.

## Files

- NEW `core/include/pulsim/v2/builder/circuit_builder.hpp`
- NEW `core/tests/v2/builder/test_main.cpp`
- NEW `core/tests/v2/builder/test_circuit_builder.cpp`
- MODIFIED `core/CMakeLists.txt` (add `pulsim_v2_builder_tests`)
- NEW `docs/pulsim-v2/layer6-builder-api.md`
