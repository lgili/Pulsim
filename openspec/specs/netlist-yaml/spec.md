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

### Requirement: Schema Evolution Lifecycle
The YAML netlist capability SHALL define a versioned schema-evolution lifecycle with deterministic behavior for added, deprecated, and removed fields.

#### Scenario: Deprecated field inside migration window
- **WHEN** a YAML netlist uses a deprecated field still within the supported migration window
- **THEN** parsing succeeds with a structured deprecation diagnostic
- **AND** diagnostic output includes canonical replacement guidance

#### Scenario: Removed field after migration window
- **WHEN** a YAML netlist uses a field removed from the active schema version
- **THEN** parsing fails deterministically
- **AND** failure diagnostics include migration guidance to supported fields

### Requirement: Deterministic Structural Diagnostics
YAML parsing SHALL report strict diagnostics with stable reason codes and precise field-path context.

#### Scenario: Unknown field in strict mode
- **WHEN** strict validation is enabled and YAML contains an unknown field
- **THEN** parsing fails with a deterministic reason code
- **AND** diagnostics include the exact YAML path of the invalid field

#### Scenario: Invalid type at known field
- **WHEN** a known field has a value type that violates schema constraints
- **THEN** parsing fails with a deterministic type-mismatch diagnostic
- **AND** expected and received type classes are reported

### Requirement: Extension Namespace Validation
The YAML schema SHALL support explicit extension namespaces validated against registered runtime capabilities.

#### Scenario: Valid extension block
- **WHEN** YAML provides an extension block under a recognized namespace and valid contract
- **THEN** parsing maps the extension to runtime options deterministically
- **AND** extension metadata is available in structured parse results

#### Scenario: Unknown extension namespace
- **WHEN** YAML provides an extension namespace not recognized by the active registry
- **THEN** parsing fails (or warns in non-strict mode per policy) with deterministic diagnostics
- **AND** diagnostics identify the unsupported namespace

### Requirement: Compatibility Corpus Enforcement
The YAML parser SHALL maintain a compatibility corpus to prevent behavioral drift for supported legacy netlists.

#### Scenario: Supported legacy corpus run
- **WHEN** compatibility tests execute against the approved legacy YAML corpus
- **THEN** all supported netlists parse with equivalent semantic mappings
- **AND** regressions are flagged before merge

### Requirement: Model Regularization YAML Surface
The YAML schema SHALL provide a `simulation.model_regularization` configuration block for numerical regularization controls used by nonlinear/switching component models.

#### Scenario: Explicit regularization policy in YAML
- **GIVEN** a netlist with `simulation.model_regularization` overrides
- **WHEN** the parser loads the netlist
- **THEN** the runtime options include the configured regularization policy values
- **AND** invalid ranges are reported as parser errors in strict mode

#### Scenario: Safe defaults when block is omitted
- **GIVEN** a netlist without `simulation.model_regularization`
- **WHEN** the parser loads the netlist
- **THEN** runtime uses conservative defaults
- **AND** behavior remains backward-compatible for existing netlists

