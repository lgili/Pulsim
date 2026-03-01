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

