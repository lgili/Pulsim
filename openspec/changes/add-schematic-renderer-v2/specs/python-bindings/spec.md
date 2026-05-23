# Python Bindings — Schematic V2 Surface

## ADDED Requirements

### Requirement: Component Descriptor Enumeration
The `CircuitBuilder` Python binding SHALL expose a `components()`
method returning a list of `ComponentDescriptor` records — one per
device added through any `add_*` call. The list order SHALL match the
insertion order so that consumers (renderer, exporter, future GUI)
get a stable iteration.

#### Scenario: Enumerate components of a buck circuit
- **WHEN** the user constructs a buck circuit via the six
  `b.add_*()` calls
- **AND** calls `b.components()`
- **THEN** the returned list has length 6
- **AND** each entry exposes `.name`, `.kind`, `.nodes`, `.params`

#### Scenario: Stable iteration order
- **WHEN** a circuit is built and `components()` is called twice
- **THEN** the two lists are equal element-by-element

#### Scenario: Component kind is a canonical string
- **WHEN** the user inspects `components()` entries
- **THEN** `.kind` is one of the canonical strings: `"resistor"`,
  `"capacitor"`, `"inductor"`, `"voltage_source"`, `"current_source"`,
  `"pwm_voltage_source"`, `"sine_voltage_source"`,
  `"pulse_voltage_source"`, `"switch"`, `"diode"`,
  `"nonlinear_diode"`, `"mosfet_level1"`, `"igbt_level1"`, `"vcvs"`,
  `"transformer"`, `"saturable_inductor"`

#### Scenario: Component params dict
- **WHEN** the user reads `entry.params` for a resistor
- **THEN** the dict contains the key `"R_ohms"` (or the canonical
  parameter name documented for that kind) mapping to a float

### Requirement: Graph Branch Enumeration
The `Graph` Python binding SHALL expose a `branches` property
returning a list of branch records, each carrying `id`, `from_`, `to`,
and `kind` (an enum exposed as `pulsim.BranchKind`).

#### Scenario: Branches of a 2-component circuit
- **WHEN** a circuit has been built with one voltage source and one
  resistor
- **AND** the user reads `b.graph.branches`
- **THEN** the returned list has exactly 2 entries
- **AND** the first entry's `kind == pulsim.BranchKind.Source`
- **AND** the second entry's `kind == pulsim.BranchKind.PassiveLinear`

#### Scenario: BranchKind enum values
- **WHEN** the user inspects `pulsim.BranchKind`
- **THEN** at least the values `Source`, `PassiveLinear`, `Switch`,
  `Nonlinear` are present
- **AND** each value compares equal to itself and is iterable via
  the standard `pybind11` enum API

### Requirement: Position Hint Accessors
The `CircuitBuilder` Python binding SHALL expose a
`node_position_hint(node_id)` method returning a string role hint and
a `position_hints()` method returning the full hint dict. V0 of this
change MAY return `"internal"` for every non-ground node and the
ground role for the ground node; future work can refine the
classification.

#### Scenario: Ground node is hinted as ground
- **WHEN** the user calls
  `b.node_position_hint(b.graph.ground)`
- **THEN** the returned string equals `"ground"`

#### Scenario: Default hint for unrecognized roles
- **WHEN** the user calls `b.node_position_hint(some_node_id)` for a
  node that has no explicit hint registered
- **THEN** the returned string equals `"internal"`

#### Scenario: position_hints returns a dict
- **WHEN** the user calls `b.position_hints()`
- **THEN** the result is a `dict[int, str]` mapping every node ID
  used in the circuit to its current hint
