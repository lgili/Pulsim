## ADDED Requirements

### Requirement: Circuit Parameters And Expressions In YAML

The YAML netlist SHALL accept an optional `circuit.parameters` list of
`{name, value, unit}` entries, and SHALL treat a string value of a
numeric device field as an expression over those parameters. Files
without the key SHALL load exactly as before. An invalid expression
SHALL fail loading with an error naming the device and the field.

#### Scenario: Parameterised netlist
- **GIVEN** `parameters: [{name: Lbase, value: 1e-3}]` and an inductor
  with `L: "2*Lbase"`
- **WHEN** the file is loaded and simulated
- **THEN** the result is bit-identical to the same file with `L: 2e-3`

#### Scenario: Existing files unchanged
- **WHEN** every shipped v2 example is loaded
- **THEN** it produces the same circuit and the same warnings as
  before

#### Scenario: Bad expression refused
- **WHEN** a field is `"2*"` or names an undefined parameter
- **THEN** loading fails with an error naming the device and the field

### Requirement: Subsystem Scopes In YAML

The YAML netlist SHALL accept an optional `circuit.subsystems` list of
nested blocks, each with a `name`, an optional `kind`, and its own
`parameters`, `devices` and `subsystems`. Elements inside a block SHALL
receive the block's scope path as a name prefix joined by `/`.

#### Scenario: Nested instances
- **GIVEN** a subsystem `leg_a` containing a device `Q1`
- **WHEN** the file is loaded
- **THEN** `components()` lists `leg_a/Q1`

#### Scenario: Scope survives a round trip
- **WHEN** a circuit with nested subsystems is written with `to_yaml`
  and loaded again
- **THEN** the scope tree is the same

### Requirement: Node Domains In YAML

An entry of `circuit.nodes` SHALL be either a name (an electrical
node) or a mapping `{name, domain}` with `domain` one of `electrical`,
`thermal`, `mechanical`. An unknown domain SHALL be refused by name.

#### Scenario: Thermal node declared
- **WHEN** `nodes: [{name: tj, domain: thermal}]` is loaded
- **THEN** the node `tj` has the thermal domain

#### Scenario: Unknown domain refused
- **WHEN** a node declares `domain: hydraulic`
- **THEN** loading fails with an error naming the node and the domain

### Requirement: YAML Emitter And Complete Device Coverage

The kernel SHALL provide an emitter that writes any `Circuit` as a
netlist the loader accepts, and the loader SHALL accept every built-in
device kind, including `shockley_diode`, `lauritzen_diode`,
`nonlinear_capacitor`, `pmsm_mna` and the flux-table inductor. The
loader and the emitter SHALL be driven by one schema table.

#### Scenario: Every kind loads
- **WHEN** a netlist containing one device of each built-in kind is
  loaded
- **THEN** no kind is refused as unknown

#### Scenario: Emitted netlist is accepted
- **WHEN** the emitter's output for any harness circuit is loaded with
  strict validation
- **THEN** it loads without warnings and has the same value hash as
  the source circuit

#### Scenario: Expressions are written as written
- **WHEN** a circuit with a field `"2*Lbase"` is emitted
- **THEN** the YAML contains the expression text and the parameter
  definition, not the evaluated number
