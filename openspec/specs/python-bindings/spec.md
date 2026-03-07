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

### Requirement: Typed Electrothermal Characterization Bindings
Python bindings SHALL expose typed structures for datasheet-grade loss characterization and thermal-network configuration.

#### Scenario: Configure datasheet characterization from Python
- **WHEN** Python code configures semiconductor loss and thermal-network structures via typed bindings
- **THEN** runtime receives equivalent backend configuration without requiring YAML-only pathways
- **AND** invalid assignments fail with deterministic typed errors

### Requirement: Canonical Electrothermal Channel Metadata in Python
Python simulation results SHALL expose canonical loss and thermal channels with structured metadata sufficient for frontend routing.

#### Scenario: Frontend adapter reads channels via Python
- **WHEN** Python tooling enumerates `result.virtual_channels` and metadata
- **THEN** it can identify electrothermal channels by metadata fields (domain, quantity, source component, unit)
- **AND** no name-regex heuristic is required for channel classification

### Requirement: Backward-Compatible Summary and Telemetry Surface
Python bindings SHALL preserve existing summary payloads while adding richer per-sample electrothermal channels.

#### Scenario: Existing script consumes summaries only
- **WHEN** a script reads legacy `loss_summary`, `thermal_summary`, and `component_electrothermal`
- **THEN** behavior remains backward compatible
- **AND** summary values are consistent with reductions over canonical electrothermal channels

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

### Requirement: Python Configuration for SUNDIALS Backend
Python bindings SHALL expose SUNDIALS backend configuration fields equivalent to runtime simulation options.

#### Scenario: Configure SUNDIALS backend from Python
- **WHEN** Python code sets transient backend mode to `SUNDIALS` or `Auto` and selects solver family/tolerances
- **THEN** `Simulator` and procedural APIs SHALL run with the same backend configuration semantics as kernel options
- **AND** invalid backend-family combinations SHALL produce clear Python exceptions

### Requirement: Python Telemetry for Backend Selection
Python bindings SHALL expose backend telemetry including whether SUNDIALS was used and why escalation happened.

#### Scenario: Inspect backend telemetry after run
- **WHEN** Python code accesses transient simulation result telemetry
- **THEN** it SHALL read backend selection, solver family, escalation counters, and backend failure diagnostics when applicable
- **AND** these fields SHALL be structured (not string-parsed)

### Requirement: Backward-Compatible Defaults Without SUNDIALS
Python APIs SHALL remain backward-compatible on builds without SUNDIALS.

#### Scenario: Existing script on non-SUNDIALS build
- **WHEN** an existing Python script uses default transient APIs without backend overrides
- **THEN** behavior SHALL remain compatible with the native backend defaults
- **AND** capability inspection SHALL report SUNDIALS unavailable without breaking imports

### Requirement: Python-Only Supported Runtime Surface
Python bindings SHALL be the only supported user-facing runtime interface for simulation workflows.

#### Scenario: User follows supported workflow
- **WHEN** a user executes simulation workflows documented as supported
- **THEN** all workflows are available through the Python package interface
- **AND** documentation does not require direct C++ or legacy CLI usage

### Requirement: Full v1 Configuration Exposure
Python bindings SHALL expose all v1 solver, integrator, periodic, and thermal configuration required by declared converter workflows.

#### Scenario: Configure advanced converter run
- **WHEN** Python code configures declared v1 runtime options for a converter case
- **THEN** equivalent options are available through bindings without undocumented C++-only fallback

### Requirement: Converter Component and Thermal API Coverage
Python bindings SHALL expose APIs to build and run declared converter component sets with associated thermal and loss models.

#### Scenario: Build electro-thermal converter in Python
- **WHEN** a benchmark converter case uses declared electrical and thermal components
- **THEN** the case can be constructed and executed from Python without legacy adapters

### Requirement: Deprecated Surface Retirement Policy
Deprecated Python entrypoints SHALL include a migration path and versioned removal policy.

#### Scenario: Deprecated entrypoint present
- **WHEN** an entrypoint is marked for removal
- **THEN** bindings and docs provide a supported replacement and removal version
- **AND** CI includes migration coverage during the deprecation window

### Requirement: Canonical Mode-Based Transient Configuration
Python bindings SHALL expose a canonical transient mode selection equivalent to YAML `step_mode` semantics (`fixed` or `variable`).

#### Scenario: Configure fixed mode from Python
- **WHEN** Python code selects fixed mode through the canonical runtime API
- **THEN** the transient run executes with fixed-step macro-grid semantics
- **AND** output sampling follows deterministic fixed-grid behavior

#### Scenario: Configure variable mode from Python
- **WHEN** Python code selects variable mode through the canonical runtime API
- **THEN** the transient run executes with adaptive-step semantics
- **AND** telemetry includes adaptive acceptance/rejection metrics

### Requirement: Hybrid Segment Runtime Semantics in Python
Python bindings SHALL map canonical mode selection to hybrid segment-first runtime behavior without requiring legacy backend selectors.

#### Scenario: Canonical mode uses segment-first runtime
- **WHEN** Python config selects either `fixed` or `variable` canonical mode
- **THEN** runtime attempts state-space segment solve as primary path
- **AND** uses nonlinear DAE fallback deterministically when segment admissibility fails

### Requirement: Expert Override Exposure
Python bindings SHALL provide explicit expert override controls without requiring them for standard transient use.

#### Scenario: Standard use without expert overrides
- **WHEN** Python code configures only canonical mode and timing fields
- **THEN** the simulation runs with deterministic mode-derived default profiles

#### Scenario: Expert override application
- **WHEN** Python code supplies expert override controls
- **THEN** overrides are applied on top of canonical mode defaults
- **AND** invalid expert keys or values produce structured errors

### Requirement: Legacy Transient Configuration Migration Diagnostics
Python bindings SHALL provide deterministic migration diagnostics for removed legacy transient-backend controls.

#### Scenario: Deprecated legacy backend field usage
- **WHEN** Python code attempts to use deprecated backend-specific transient controls removed from supported runtime
- **THEN** bindings raise a structured configuration error
- **AND** include migration guidance to canonical mode-based controls

### Requirement: Electrothermal and Loss Surface Support
Python bindings SHALL expose loss and thermal configuration/results in canonical mode workflows.

#### Scenario: Configure electrothermal options from Python
- **WHEN** Python config enables losses and thermal coupling on a canonical mode run
- **THEN** runtime accepts the configuration without requiring legacy backend controls
- **AND** simulation results include `loss_summary` and `thermal_summary`

#### Scenario: Invalid electrothermal override values
- **WHEN** Python config provides invalid thermal/loss override values
- **THEN** bindings raise structured configuration errors
- **AND** preserve deterministic error messaging

### Requirement: Unified Per-Component Electrothermal Telemetry in Python
Python bindings SHALL expose a unified per-component electrothermal telemetry surface in `SimulationResult`.

#### Scenario: Access per-component losses and temperatures
- **WHEN** Python runs a transient simulation with electrothermal options
- **THEN** `SimulationResult` exposes per-component entries keyed by component identity
- **AND** each entry includes both loss and temperature fields with deterministic schema and ordering

#### Scenario: Thermal-disabled entry shape remains stable
- **WHEN** a component has no enabled thermal port
- **THEN** the component entry still includes thermal fields
- **AND** thermal status is explicit (`thermal_enabled=false`) with deterministic default values

### Requirement: Backward-Compatible Summary Surfaces
Python bindings SHALL keep existing `loss_summary` and `thermal_summary` surfaces while introducing unified per-component telemetry.

#### Scenario: Existing tooling reads legacy summaries
- **WHEN** Python tooling continues to consume `loss_summary` and `thermal_summary`
- **THEN** behavior remains backward compatible
- **AND** aggregate values remain consistent with reductions of the unified per-component telemetry

