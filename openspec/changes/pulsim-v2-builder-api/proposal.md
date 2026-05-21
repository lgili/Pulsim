## Why

v2's canonical setup requires the user to manually wire two
parallel objects: a `Graph` (topology) and a `DevicePool`
(per-branch parameters). Branch indices must be tracked by
hand, with `Graph::add_branch` returning the index that the
user then passes to `pool.add_*`. Node lookups use raw `Index`
values, not names. Parameter structs use internal conventions
(e.g., `Resistor::Params{ .G = 1.0 / R }` instead of ohms).

The result: every test file (and any future user code) has 5
lines of bookkeeping per component, and any reordering /
insertion breaks downstream indices. This pattern is fine for
the kernel layer but TERRIBLE for users.

```cpp
// Current canonical:
Graph g;
g.add_node("n0");
g.add_node("n1");
g.add_branch(0, g.ground(), BranchKind::Source);
g.add_branch(0, 1,          BranchKind::PassiveLinear);
g.add_branch(1, g.ground(), BranchKind::PassiveLinear);

DevicePool pool;
pool.add_voltage_source(0, {.V = 5.0});
pool.add_resistor(1, {.G = 1.0 / 100.0});
pool.add_resistor(2, {.G = 1.0 / 10.0});
```

V6 ships a `CircuitBuilder` that hides this:

```cpp
// V6 builder:
CircuitBuilder b;
b.add_voltage_source("Vin", "n0", "gnd", 5.0);
b.add_resistor      ("R1",  "n0", "n1", 100.0);
b.add_resistor      ("R2",  "n1", "gnd", 10.0);

PwlStateSpaceCache cache(b.graph(), b.pool());
cache.build(dt);
```

This is the gateway between Pulsim's well-tested low-level
kernel and real-world user code. Without it, every external
adopter has to learn (and stay consistent with) the
two-object pattern — a needless adoption tax.

## What Changes

**Scope decision — Layer 6 V0** (CircuitBuilder API):

- New header `pulsim/v2/builder/circuit_builder.hpp`:
  ```cpp
  namespace pulsim::v2::builder {

  class CircuitBuilder {
  public:
      // Node management (mostly implicit; advanced users
      // can declare explicitly).
      Index node(std::string name);   // returns existing or
                                       // creates new

      // Device methods. Each returns a reference to `*this`
      // for optional chaining. All methods are noexcept on
      // duplicate device names (would throw with the diag
      // message in V1; V0 simply allows duplicates).
      CircuitBuilder& add_voltage_source(
          std::string name, std::string from, std::string to,
          Real V);

      CircuitBuilder& add_resistor(
          std::string name, std::string from, std::string to,
          Real R_ohms);

      CircuitBuilder& add_capacitor(
          std::string name, std::string from, std::string to,
          Real C_farads);

      CircuitBuilder& add_inductor(
          std::string name, std::string from, std::string to,
          Real L_henries);

      CircuitBuilder& add_diode(
          std::string name, std::string anode,
          std::string cathode,
          Real g_on, Real g_off, Real V_th);

      CircuitBuilder& add_nonlinear_diode(
          std::string name, std::string anode,
          std::string cathode,
          models::IdealDiode::Params params);

      CircuitBuilder& add_switch(
          std::string name, std::string from, std::string to,
          Real g_on, Real g_off);

      // Accessors. `graph()` and `pool()` return const refs
      // so the user can hand them to PwlStateSpaceCache /
      // run_transient. They live inside the builder; the
      // user must keep the builder alive.
      [[nodiscard]] const topology::Graph& graph() const noexcept;
      [[nodiscard]] const pwl::DevicePool& pool() const noexcept;

      // Diagnostic / debugging.
      [[nodiscard]] Size num_branches() const noexcept;
      [[nodiscard]] Index node_id_of(const std::string& name) const;
  };

  }  // namespace pulsim::v2::builder
  ```

- "gnd" / "GND" / "0" all map to `graph.ground()` (the
  implicit reference node).

- Node names are case-sensitive otherwise. Looking up a
  non-existent name in `node_id_of` throws
  `std::out_of_range`.

- Resistor params use `R_ohms`; the builder converts to
  `G = 1/R` internally.

- All device methods auto-create missing nodes via the same
  internal node-mapping logic.

- **Tests** (~10 cases):
  - "gnd" alias maps to ground.
  - Implicit node creation via device methods.
  - Round-trip: build V_dc + R via the builder, solve via
    `PwlStateSpaceCache`, verify v_node = V_dc.
  - Build the V2 half-wave rectifier via the builder; run
    `run_transient`; verify the rectifier output matches
    the manual-setup test within 1 µV.
  - Build the V1.5 buck converter via the builder; verify
    parity with the manual setup.
  - All component types (R, C, L, V, diode, NL diode,
    switch) accept user-friendly units and produce the
    correct pool entries.

## Impact

- **Affected specs**: ADDED requirement on
  `kernel-v2-solver` for `CircuitBuilder`.
- **Affected code** (~300 LOC):
  - NEW `core/include/pulsim/v2/builder/circuit_builder.hpp`
  - NEW `core/tests/v2/builder/test_circuit_builder.cpp`
  - NEW `core/tests/v2/builder/test_main.cpp`
- **Migration**: zero. Existing test files keep using the
  manual `Graph + DevicePool` pattern. New code can opt-in
  to the builder.
- **Risk**: low. Pure additive wrapper that delegates to
  existing `Graph::add_*` and `DevicePool::add_*` methods.
  Behaviour parity is verified via integration tests
  (half-wave rectifier, buck converter).
