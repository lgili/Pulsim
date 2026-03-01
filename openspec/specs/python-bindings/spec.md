# python-bindings Specification

## Purpose
TBD - created by archiving change remove-cli-benchmark-dependency. Update Purpose after archive.
## Requirements
### Requirement: Runtime-Complete Simulation Objects
Python bindings SHALL expose runtime-complete simulation objects equivalent to v1 kernel execution controls, including `SimulationOptions` and `Simulator`.

#### Scenario: Configure and run simulation through class APIs
- **WHEN** Python code creates `SimulationOptions`, configures solver/integrator fields, and instantiates `Simulator`
- **THEN** the simulation executes with the configured options
- **AND** behavior matches kernel runtime semantics for those options

### Requirement: Periodic Steady-State API Exposure
Python bindings SHALL expose periodic steady-state methods supported by the v1 kernel.

#### Scenario: Run shooting method from Python
- **WHEN** Python code calls `Simulator.run_periodic_shooting(...)`
- **THEN** the method executes and returns structured periodic result fields (status, iterations, residuals)

#### Scenario: Run harmonic balance from Python
- **WHEN** Python code calls `Simulator.run_harmonic_balance(...)`
- **THEN** the method executes and returns structured harmonic-balance result fields

### Requirement: YAML Parser Exposure
Python bindings SHALL expose the v1 YAML parser interfaces needed to load netlists and simulation options with strict diagnostics.

#### Scenario: Parse YAML netlist through bound parser
- **WHEN** Python code loads a YAML netlist via exposed parser bindings
- **THEN** it receives parsed circuit and simulation options objects
- **AND** parser errors/warnings are accessible via Python

### Requirement: Structured Runtime Telemetry Exposure
Python bindings SHALL expose simulation result telemetry fields required by benchmark and validation workflows.

#### Scenario: Access telemetry after transient run
- **WHEN** Python code runs a transient simulation
- **THEN** result telemetry (iterations, rejections, runtime, solver telemetry) is available as structured fields
- **AND** can be consumed without parsing console text

### Requirement: Backward-Compatible Procedural API
Existing procedural entrypoints (`run_transient`, `dc_operating_point`, etc.) SHALL remain available during migration to class-based runtime APIs.

#### Scenario: Existing script uses procedural run API
- **WHEN** an existing Python script calls the procedural API
- **THEN** the script continues to run without mandatory migration in the same release window

### Requirement: Compatibility-Preserving Migration Surface
Python bindings SHALL preserve procedural API compatibility during the migration window while exposing canonical runtime surfaces for new development.

#### Scenario: Existing procedural script
- **WHEN** an existing script calls procedural entrypoints supported in the migration window
- **THEN** execution remains functional without mandatory rewrites
- **AND** deprecation guidance maps the call to canonical runtime APIs

#### Scenario: Canonical runtime usage
- **WHEN** new code uses canonical class-based runtime APIs
- **THEN** behavior matches the same underlying v1 kernel semantics as procedural compatibility paths
- **AND** telemetry parity is preserved

### Requirement: Structured Error and Failure Surface
Python bindings SHALL surface structured kernel diagnostics with stable reason codes and context fields.

#### Scenario: Invalid configuration from Python
- **WHEN** Python provides invalid simulation or solver configuration
- **THEN** bindings raise a structured error with deterministic reason code
- **AND** error context includes relevant field/path metadata

#### Scenario: Runtime failure propagation
- **WHEN** kernel execution terminates with a typed failure reason
- **THEN** Python receives the same reason code and terminal diagnostics
- **AND** no console-text parsing is required

### Requirement: Extension Introspection Surface
Python bindings SHALL expose read-only introspection for registered device, solver, and integrator capabilities.

#### Scenario: Query available runtime capabilities
- **WHEN** Python requests registered extension capabilities
- **THEN** bindings return structured metadata for available devices/solvers/integrators
- **AND** reported capabilities match active kernel registry state

### Requirement: KPI Telemetry Contract for Tooling
Python simulation results SHALL expose KPI-critical telemetry fields required by benchmark and regression tooling.

#### Scenario: Collect KPI telemetry from Python run
- **WHEN** benchmark tooling runs scenarios through Python bindings
- **THEN** result objects include fields needed for convergence, accuracy, runtime, event, and fallback KPIs
- **AND** the field schema is stable for automated consumers

