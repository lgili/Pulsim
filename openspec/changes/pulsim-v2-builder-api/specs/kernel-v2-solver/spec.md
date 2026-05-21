## ADDED Requirements

### Requirement: CircuitBuilder class

A `CircuitBuilder` class SHALL be available in
`pulsim::v2::builder` that lets users construct a v2 circuit
using string node names and SI-unit parameter values
(ohms, farads, henries, volts). The builder MUST hide the
two-object `Graph + DevicePool` split, exposing only
high-level "add component" methods.

The builder MUST own its `Graph` and `DevicePool` instances
internally and expose them via `const` accessors `graph()`
and `pool()` so that the caller can construct a
`PwlStateSpaceCache` from them.

The builder MUST handle the `"gnd"`, `"GND"`, and `"0"`
node names as aliases for `Graph::ground()`. Any other name
not yet registered MUST be auto-created on first use; the
mapping name → index MUST be preserved across the builder's
lifetime.

For resistors, capacitors, and inductors, the builder MUST
accept SI-unit values (`R_ohms`, `C_farads`, `L_henries`)
and convert them to the kernel's `Resistor::Params{ .G = 1/R }`,
`Capacitor::Params{ .C }`, and `Inductor::Params{ .L }`
internally.

#### Scenario: `"gnd"` alias maps to graph.ground()

- **GIVEN** a fresh `CircuitBuilder`
- **WHEN** the user calls `builder.node("gnd")`
- **THEN** the returned `Index` SHALL equal
  `builder.graph().ground()`.

#### Scenario: Implicit node creation via device method

- **GIVEN** a fresh `CircuitBuilder` with no explicit
  `.node()` calls
- **WHEN** the user calls
  `builder.add_voltage_source("Vin", "n0", "gnd", 5.0)`
- **THEN** the builder's graph SHALL contain a node named
  `"n0"`
- **AND** the source SHALL be registered on a single new
  branch in the pool.

#### Scenario: Resistor unit conversion (ohms → siemens)

- **GIVEN** a fresh `CircuitBuilder`
- **WHEN** the user calls
  `builder.add_resistor("R1", "a", "b", 100.0)`
- **THEN** the resulting resistor entry in the pool SHALL
  have `Params::G == 1.0 / 100.0` within numerical
  precision.

#### Scenario: Builder produces a circuit equivalent to the manual setup

- **GIVEN** the canonical V2 half-wave rectifier built via
  manual `Graph::add_node` / `add_branch` + `DevicePool`
  calls (the existing layer5_v2 test fixture)
- **AND** the same circuit built via `CircuitBuilder` with
  identical component values
- **WHEN** both runs of `run_transient` are executed over
  the same time window with the same b_extra function and
  switch_fn
- **THEN** the recorded `(times, states)` SHALL be
  bit-identical sample by sample within 1 µV per state
  entry.

### Requirement: node_id_of validates existence

`CircuitBuilder::node_id_of(name)` SHALL return the
internal `Index` of a previously-registered node.
If the name has never been used (neither via explicit
`.node()` nor through any device method), the function
MUST throw `std::out_of_range`.

#### Scenario: Querying an unknown node name throws

- **GIVEN** a `CircuitBuilder` with no registered nodes
- **WHEN** the user calls `builder.node_id_of("never_added")`
- **THEN** the call SHALL throw `std::out_of_range`.
