## ADDED Requirements

### Requirement: Python API Coverage for Missing GUI Components

Python bindings SHALL expose component-construction APIs or descriptor-based APIs sufficient to instantiate all currently missing GUI component types in runtime circuits.

#### Scenario: Build parity circuit from Python
- **GIVEN** Python code defining a circuit with each previously missing GUI component family
- **WHEN** the circuit is constructed via bindings
- **THEN** all components are accepted and mapped to backend runtime representations
- **AND** no missing-binding error is raised for those component families

### Requirement: Parameter Structs and Validation Exposure

Python bindings SHALL expose parameter structures and validation diagnostics for newly supported models.

#### Scenario: Invalid parameter validation from Python
- **WHEN** Python configures an invalid parameter set for a new component type
- **THEN** a structured exception/diagnostic is returned with component type and parameter context

### Requirement: Instrumentation Result Access

Python bindings SHALL expose probe/scope/routing outputs as structured result channels.

#### Scenario: Read probe/scope channels
- **GIVEN** a simulation containing probes and scopes
- **WHEN** Python reads simulation results
- **THEN** per-channel metadata and waveform values are available without post-hoc GUI-only reconstruction

### Requirement: Backward-Compatible Existing APIs

Existing Python runtime APIs SHALL remain functional while new component APIs are introduced.

#### Scenario: Existing script compatibility
- **WHEN** an existing script using current `Circuit.add_*` methods runs
- **THEN** behavior remains compatible
- **AND** introduction of new component APIs does not break prior workflows
