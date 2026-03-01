# netlist-yaml Specification

## Purpose
TBD - created by archiving change unify-v1-core. Update Purpose after archive.
## Requirements
### Requirement: Versioned YAML Netlist
The system SHALL require a `schema` identifier and a `version` field in the YAML netlist and reject unsupported values.

#### Scenario: Missing version
- **WHEN** a YAML netlist omits the `version` field
- **THEN** parsing fails with a clear diagnostic

#### Scenario: Missing schema
- **WHEN** a YAML netlist omits the `schema` field
- **THEN** parsing fails with a clear diagnostic

### Requirement: YAML Parsing via yaml-cpp
The system SHALL parse YAML netlists using `yaml-cpp` integrated via CMake FetchContent.

#### Scenario: Load a YAML file
- **WHEN** the parser loads a `.yaml` or `.yml` file
- **THEN** the netlist is parsed using `yaml-cpp`

### Requirement: Components and Simulation Sections
The system SHALL support `simulation` and `components` sections with explicit component types, nodes, and parameters.

#### Scenario: Simple RC circuit
- **WHEN** the netlist defines `simulation` and `components`
- **THEN** the circuit is constructed correctly and simulation options are applied

### Requirement: Waveforms and Models
The system SHALL support waveform definitions and reusable models referenced by components, including model inheritance with local overrides.

#### Scenario: PWM source using a model
- **WHEN** a component uses a model with a PWM waveform
- **THEN** the waveform is instantiated with the model parameters

#### Scenario: Model override
- **WHEN** a component references a model and overrides one parameter locally
- **THEN** the local parameter overrides the inherited model value

### Requirement: Strict Validation
The system SHALL provide strict validation and diagnostics for unknown fields or invalid values.

#### Scenario: Unsupported field in component
- **WHEN** a component contains an unknown field
- **THEN** parsing fails with a diagnostic indicating the field

### Requirement: Solver Configuration in YAML
The YAML netlist SHALL allow explicit solver configuration under the `simulation` section.

#### Scenario: Configure solver stack
- **WHEN** the netlist defines `simulation.solver` options (linear, nonlinear, preconditioner, fallback order)
- **THEN** the parser SHALL map them to the v1 simulation options
- **AND** invalid values SHALL produce a clear diagnostic

#### Scenario: Strict validation
- **WHEN** strict validation is enabled
- **THEN** unknown solver fields SHALL cause parsing to fail

### Requirement: Solver Order Configuration
The YAML netlist SHALL allow specifying primary and fallback solver orders.

#### Scenario: Separate orders
- **WHEN** `simulation.solver.order` and `simulation.solver.fallback_order` are provided
- **THEN** the parser SHALL map them to distinct primary and fallback orders

### Requirement: Advanced Solver Options
The YAML netlist SHALL allow configuration for JFNK, preconditioners, and stiff integrators.

#### Scenario: JFNK in YAML
- **WHEN** the netlist enables `simulation.solver.nonlinear.jfnk`
- **THEN** the solver SHALL use the JFNK path

#### Scenario: Preconditioner selection
- **WHEN** the netlist sets `solver.iterative.preconditioner` to `ilut` or `amg`
- **THEN** the parser SHALL accept the value or error with a clear diagnostic if unavailable

#### Scenario: Stiff integrator selection
- **WHEN** the netlist sets `simulation.integration` (or `simulation.integrator`) to `tr-bdf2` or `rosenbrock`
- **THEN** the parser SHALL apply the selected integrator

#### Scenario: Periodic steady-state options
- **WHEN** the netlist sets `simulation.shooting` or `simulation.harmonic_balance` (`simulation.hb`)
- **THEN** the parser SHALL map the options to periodic steady-state configuration

#### Scenario: Residual cache tuning
- **WHEN** the netlist sets `simulation.newton.krylov_residual_cache_tolerance`
- **THEN** the parser SHALL apply the specified residual cache tolerance

### Requirement: Unified YAML Parsing Across Python and Kernel Paths
The system SHALL use the same v1 YAML parsing semantics for Python runtime execution as for kernel-native execution paths.

#### Scenario: Parse benchmark YAML from Python runtime
- **WHEN** benchmark tooling loads a YAML netlist through Python bindings
- **THEN** parsing behavior and option mapping match the v1 YAML parser semantics
- **AND** supported solver/integrator/periodic fields map identically

### Requirement: Strict Diagnostic Parity in Python
Strict validation diagnostics from the YAML parser SHALL be exposed consistently in Python runtime paths.

#### Scenario: Unknown solver field in strict mode
- **WHEN** a YAML netlist contains an unknown field and strict mode is enabled via Python-exposed parser options
- **THEN** parsing fails with a clear diagnostic message
- **AND** benchmark tooling surfaces that diagnostic as an explicit failure reason

