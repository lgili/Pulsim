## ADDED Requirements
### Requirement: Typed Python Magnetic-Core Configuration Surface
Python bindings SHALL expose typed configuration surfaces for nonlinear magnetic-core models used by supported magnetic components.

#### Scenario: Configure nonlinear magnetic-core model from Python
- **GIVEN** Python code configuring a supported magnetic component with nonlinear core options
- **WHEN** simulation options/circuit are built
- **THEN** configuration maps to kernel-equivalent runtime fields deterministically
- **AND** no YAML-only workaround is required.

#### Scenario: Invalid Python magnetic configuration
- **GIVEN** Python code providing invalid magnetic-core parameters
- **WHEN** configuration validation runs
- **THEN** bindings raise deterministic typed diagnostics
- **AND** diagnostics include parameter context and reason code.

### Requirement: Python Exposure of Magnetic Telemetry and Metadata
Python simulation results SHALL expose magnetic-core channels, metadata, and summary fields required for tooling/frontend consumption.

#### Scenario: Read magnetic channels and metadata from Python
- **GIVEN** a completed nonlinear magnetic-core simulation
- **WHEN** Python code inspects result channels and metadata
- **THEN** magnetic quantities are accessible with deterministic keys and metadata
- **AND** channel classification does not require regex heuristics.

#### Scenario: Summary consistency in Python surface
- **GIVEN** magnetic channel time series and summary payloads
- **WHEN** Python compares reductions against summaries
- **THEN** values are consistent within declared tolerance
- **AND** mismatch is surfaced as deterministic diagnostics when checks are enabled.

### Requirement: Python Deterministic Diagnostic Propagation
Python bindings SHALL propagate parser/runtime magnetic-core failure diagnostics with stable machine-readable semantics.

#### Scenario: Parser magnetic error propagation
- **GIVEN** a YAML netlist with invalid magnetic-core block
- **WHEN** Python parser loads the netlist
- **THEN** Python receives deterministic structured diagnostics with field-path context
- **AND** tooling can classify failure without free-text parsing.

#### Scenario: Runtime magnetic failure propagation
- **GIVEN** a runtime magnetic-core execution failure
- **WHEN** simulation is executed via Python
- **THEN** Python receives deterministic typed failure information
- **AND** result status remains unambiguous for CI gates.
