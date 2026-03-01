## ADDED Requirements

### Requirement: Netlist Parsing
The kernel SHALL parse circuit descriptions from JSON and YAML formats into an internal representation (IR).

#### Scenario: Parse JSON netlist
- **WHEN** a valid JSON netlist file is provided
- **THEN** the parser produces an IR with all components and connections
- **AND** validates node references exist

#### Scenario: Parse YAML netlist
- **WHEN** a valid YAML netlist file is provided
- **THEN** the parser produces equivalent IR to JSON format

#### Scenario: Invalid netlist handling
- **WHEN** an invalid netlist is provided (syntax error, missing fields)
- **THEN** the parser returns a descriptive error with line/column location

### Requirement: Netlist Schema
The netlist format SHALL support a well-defined schema with components, nodes, and simulation parameters.

#### Scenario: Component definition
- **WHEN** a component is defined in the netlist
- **THEN** it MUST include: type, name, connected nodes, and type-specific parameters
- **AND** parameters are validated against the component type schema

#### Scenario: Node naming
- **WHEN** nodes are referenced in the netlist
- **THEN** node names are case-sensitive strings
- **AND** node "0" or "gnd" is reserved as ground reference

#### Scenario: Simulation parameters
- **WHEN** simulation options are specified
- **THEN** they include: tstart, tstop, dt (timestep), and optional solver settings

### Requirement: Subcircuit Support
The parser SHALL support hierarchical circuit definitions via subcircuits.

#### Scenario: Subcircuit definition
- **WHEN** a subcircuit is defined
- **THEN** it specifies interface ports and internal components
- **AND** can be instantiated multiple times with different parameters

#### Scenario: Subcircuit instantiation
- **WHEN** a subcircuit is instantiated in a netlist
- **THEN** interface ports are connected to parent circuit nodes
- **AND** internal nodes are scoped to avoid naming conflicts

### Requirement: Parameter Expressions
The parser SHALL support mathematical expressions and parameter references.

#### Scenario: Numeric parameters
- **WHEN** a component parameter is a number
- **THEN** it is parsed as a double-precision floating point
- **AND** SI prefixes (p, n, u, m, k, M, G) are recognized

#### Scenario: Parameter references
- **WHEN** a parameter references another parameter by name
- **THEN** the reference is resolved at parse time or runtime
- **AND** circular references produce an error

#### Scenario: Mathematical expressions
- **WHEN** a parameter is an expression (e.g., "2*R1")
- **THEN** the expression is evaluated with standard operators (+, -, *, /, ^)
- **AND** common functions are supported (sin, cos, exp, log, sqrt)

### Requirement: Model Library References
The parser SHALL support referencing models from a library.

#### Scenario: Built-in model reference
- **WHEN** a component references a built-in model (e.g., "model: ideal_diode")
- **THEN** the model parameters are loaded from the standard library

#### Scenario: Custom model definition
- **WHEN** a custom model is defined inline or in a separate file
- **THEN** it overrides or extends built-in model parameters

### Requirement: Source Waveforms
The parser SHALL support various source waveform specifications.

#### Scenario: DC source
- **WHEN** a DC source is defined
- **THEN** it provides constant voltage or current

#### Scenario: Pulse source
- **WHEN** a pulse source is defined with parameters (V1, V2, td, tr, tf, pw, per)
- **THEN** it generates a periodic pulse waveform

#### Scenario: Sinusoidal source
- **WHEN** a sine source is defined with parameters (Voff, Vamp, freq, td, theta)
- **THEN** it generates a sinusoidal waveform

#### Scenario: PWL source
- **WHEN** a piece-wise linear source is defined with time-value pairs
- **THEN** it interpolates linearly between points

#### Scenario: Custom waveform
- **WHEN** a source references an external waveform file (CSV, HDF5)
- **THEN** the waveform is loaded and interpolated during simulation
