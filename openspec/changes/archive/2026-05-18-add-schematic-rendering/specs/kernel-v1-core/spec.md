## ADDED Requirements

### Requirement: Circuit Component Introspection
The Circuit type SHALL expose a deterministic enumeration of every physical component added to it, returning each component's name, canonical kind, terminal node indices in pin order, and a parameter map.

#### Scenario: Enumerate components on a code-built circuit
- **WHEN** code calls `circuit.components()` after invoking `add_resistor`, `add_capacitor`, and `add_voltage_source`
- **THEN** the returned sequence contains one descriptor per `add_*` call, in insertion order
- **AND** each descriptor exposes `name`, `kind`, `nodes`, and `params`

#### Scenario: Enumerate components on a YAML-loaded circuit
- **WHEN** code loads a netlist via `YamlParser.load(...)` and calls `components()`
- **THEN** the descriptor list mirrors the YAML `components:` array in order
- **AND** node IDs match the integer IDs resolved by the parser

#### Scenario: Determinism across runs
- **WHEN** the same circuit is built twice with identical `add_*` calls
- **THEN** both `components()` sequences are byte-identical (same names, kinds, nodes, params)

#### Scenario: Read-only accessor
- **WHEN** consumer code retains a reference to the components sequence
- **THEN** subsequent `add_*` calls do not mutate previously returned descriptors

### Requirement: Node Position Hint
The Circuit type SHALL expose a node-position-hint helper that classifies each node into a role used by downstream layout consumers.

#### Scenario: Ground node classified as ground
- **WHEN** a consumer queries the role of `circuit.ground()`
- **THEN** the helper returns the `ground` role

#### Scenario: Voltage-source-fed node classified as source positive
- **WHEN** a node is the positive terminal of a voltage source
- **THEN** the helper returns the `source_pos` role

#### Scenario: Resistive load node classified as load
- **WHEN** a node is connected to a resistor whose other terminal is ground
- **THEN** the helper returns the `load` role

#### Scenario: Internal node has no specific role
- **WHEN** a node has no role-defining connection
- **THEN** the helper returns the `internal` role
