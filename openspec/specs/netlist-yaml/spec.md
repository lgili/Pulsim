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

