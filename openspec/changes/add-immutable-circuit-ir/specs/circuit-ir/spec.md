## ADDED Requirements

### Requirement: Immutable Circuit Value

A circuit SHALL be representable as an immutable `Circuit` value
obtained from the builder with `build()`. It SHALL hold the nodes,
devices, scopes, named parameters, initial conditions, aliases and
declarations of the circuit; copying it SHALL be cheap; and it SHALL be
safe to read from several threads. No operation on a `Circuit`, and no
simulation of it, SHALL change it or the builder it came from.

#### Scenario: Simulation does not mutate the circuit
- **GIVEN** a circuit with a floating sub-network that the preflight
  regularises
- **WHEN** it is simulated with automatic regularisation
- **THEN** the builder and its `Circuit` have the same value hash
  before and after, and the regularising ties are reported on the
  result
- **AND** a second run with regularisation disabled raises the
  original singular-matrix error

#### Scenario: Lowering reproduces the builder exactly
- **WHEN** a circuit is built, converted to a `Circuit` and lowered
- **THEN** the device list, the unknown indices, the switch order and
  the initial state equal those of the builder, and the simulation
  result is bit-identical

#### Scenario: Shared across threads
- **GIVEN** one `Circuit`
- **WHEN** eight simulations of it run on four threads
- **THEN** the results equal the sequential results

### Requirement: Named Parameters And Expressions

A device field SHALL accept a literal or an expression over named
parameters. Expressions SHALL support `+`, `-`, `*`, `/`, `^`,
parentheses and a fixed list of functions, SHALL be evaluated when the
circuit is lowered, and SHALL be stored as written. A dependency
cycle, an unknown name, a non-finite value or a value outside the
field's bounds SHALL be refused with an error naming the device, the
field and the expression.

#### Scenario: Expression equals its literal
- **GIVEN** `Lbase = 1e-3` and an inductor with `L = "2*Lbase"`
- **WHEN** the circuit is simulated
- **THEN** the result is bit-identical to the circuit written with
  `L = 2e-3`

#### Scenario: Unknown name refused
- **WHEN** a field is `"2*Lbsae"` and no such parameter exists
- **THEN** an error names the device, the field and `Lbsae`

#### Scenario: Cycle refused
- **WHEN** parameter `a` is `"b+1"` and `b` is `"a+1"`
- **THEN** an error names both parameters as a cycle

### Requirement: Derived Circuits Share Structure

`with_params` and `with_value` SHALL return a new `Circuit` and leave
the receiver unchanged. `with_value` SHALL work for every field of
every device kind. Circuits that differ only in values SHALL have the
same structure hash and SHALL reuse the structure-dependent analysis.

#### Scenario: Any field of any kind
- **WHEN** `with_value("D1.I_S", 1e-12)` is applied to a circuit with a
  Shockley diode `D1`
- **THEN** the derived circuit simulates bit-identically to a circuit
  built with that value, and the original is unchanged

#### Scenario: One symbolic analysis per structure
- **GIVEN** a 100-point sweep of one parameter
- **WHEN** it runs through `with_params`
- **THEN** the fill-reducing analysis is performed once, and every
  point's result equals the result of building that point from scratch

#### Scenario: Structure hash ignores values
- **WHEN** two circuits differ only in a resistor value
- **THEN** their structure hashes are equal and their value hashes
  differ

### Requirement: Hierarchy Is Part Of The Circuit

The circuit SHALL keep a tree of scopes, with every device, node and
parameter belonging to one scope. The flattened name of an element
SHALL be its scope path and its name joined by `/`, equal to the name
used before scopes existed, so that every name-based accessor keeps
working.

#### Scenario: Names unchanged
- **GIVEN** a circuit built with nested subsystems
- **WHEN** `components()` and `res.v("leg_a/mid")` are used
- **THEN** the names and values equal those obtained with the previous
  prefix-based implementation

#### Scenario: Scoped parameter lookup
- **GIVEN** a parameter `Ron` defined in two subsystem instances with
  different values
- **WHEN** devices inside each instance use `Ron`
- **THEN** each uses its own instance's value

### Requirement: Domain-Tagged Nodes

Every node SHALL have a domain (`electrical` by default, `thermal`,
`mechanical`; `magnetic` reserved), and every device pin SHALL declare
the domain it connects to. Connecting a pin to a node of another domain
SHALL be refused with an error naming the device, the pin and the
node. Each domain SHALL have its own reference check, and probes SHALL
report the unit of their domain. All domains SHALL be solved in the
same system of equations.

#### Scenario: Cross-domain connection refused
- **WHEN** a thermal pin is connected to an electrical node
- **THEN** an error names the device, the pin and the node

#### Scenario: Untagged circuits are unchanged
- **WHEN** a circuit that declares no domain is simulated
- **THEN** the result is bit-identical to the previous build

#### Scenario: Mechanical nodes of a machine
- **GIVEN** an MNA PMSM with its speed and angle nodes tagged
  mechanical
- **WHEN** the circuit is simulated and the speed node is probed
- **THEN** the result is unchanged and the probe's unit is `rad/s`

#### Scenario: Floating thermal network
- **WHEN** a thermal network has no path to the ambient reference
- **THEN** the preflight reports it as a thermal-domain finding, naming
  the nodes

### Requirement: Complete Serialisation

A `Circuit` SHALL be convertible to YAML text and back without loss:
`from_yaml(to_yaml(c))` SHALL have the value hash of `c`, and
`to_yaml` SHALL be a fixed point after one round. Every device kind
SHALL be covered. `clone()` and pickling SHALL be provided on the same
basis. Objects that are not part of a circuit (callables attached to a
run) SHALL NOT be required for the round trip.

#### Scenario: Round trip of every kind
- **GIVEN** one circuit per device kind, every shipped example and
  every harness circuit
- **WHEN** each is written to YAML and read back
- **THEN** the value hash is equal and the simulation result is
  bit-identical

#### Scenario: A kind without a schema fails the build's tests
- **WHEN** a device kind is registered without YAML schema entries
- **THEN** the round-trip coverage test fails naming the kind

#### Scenario: Pickle and clone
- **WHEN** a `Circuit` is pickled and unpickled, or cloned
- **THEN** the copy has the same value hash

#### Scenario: Metadata survives
- **GIVEN** a device carrying part-provenance metadata
- **WHEN** the circuit is round-tripped
- **THEN** the metadata is unchanged

### Requirement: No Circuit State Outside The Circuit

Everything that defines the circuit SHALL be stored in the builder and
the `Circuit`: initial conditions, aliases, probe declarations,
declarative control and gate declarations, and every device parameter.
No module SHALL attach private attributes to a builder to carry
circuit state.

#### Scenario: State survives a C++ round trip
- **GIVEN** a circuit with initial conditions, aliases and a C-block
  declaration
- **WHEN** it is written to YAML and loaded by a C++ program
- **THEN** the loaded circuit has the same initial conditions, aliases
  and declarations

#### Scenario: No private attributes
- **WHEN** the package sources are scanned
- **THEN** no assignment to a private attribute of a builder object
  exists
