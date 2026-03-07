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

### Requirement: Datasheet Loss-Model YAML Surface
The YAML schema SHALL provide a backend-complete semiconductor loss-model block that supports both scalar and datasheet-grade characterization.

#### Scenario: Datasheet-grade loss block accepted
- **WHEN** YAML defines a component loss-model block with supported axes/tables for conduction and switching terms
- **THEN** parser maps the block into runtime characterization structures
- **AND** strict validation confirms dimensional consistency and required fields

#### Scenario: Scalar loss block remains valid
- **WHEN** YAML uses legacy scalar `eon/eoff/err` style fields
- **THEN** parsing succeeds with backward-compatible mapping
- **AND** runtime behavior remains compatible with existing scalar workflows

### Requirement: Thermal Network YAML Surface
The YAML schema SHALL support thermal-network family selection and staged thermal parameters for thermal-enabled components.

#### Scenario: Foster or Cauer network declaration
- **WHEN** YAML declares thermal network family `foster` or `cauer` with stage parameters
- **THEN** parser validates stage schema and maps to runtime thermal-network configuration
- **AND** invalid stage dimensions or ranges fail with deterministic diagnostics

#### Scenario: Single-RC declaration compatibility
- **WHEN** YAML provides classic single `rth/cth` thermal fields
- **THEN** parser maps them as `single_rc` behavior
- **AND** migration remains backward compatible

### Requirement: Deterministic Table and Axis Validation
The YAML parser SHALL validate datasheet table schema deterministically, including axis monotonicity and dimension matching.

#### Scenario: Invalid table dimensions
- **WHEN** table dimensions do not match declared axes cardinality
- **THEN** parsing fails with deterministic reason code
- **AND** diagnostics include exact YAML field path for the mismatch

#### Scenario: Non-monotonic axis values
- **WHEN** an axis expected to be monotonic is not monotonic
- **THEN** strict validation fails with deterministic diagnostics

### Requirement: GUI-Agnostic Backend-Complete Contract
YAML electrothermal schema SHALL remain GUI-agnostic so simulation correctness does not depend on frontend heuristics.

#### Scenario: Headless execution with full electrothermal definition
- **WHEN** a netlist provides full loss and thermal definitions and runs via backend/Python only
- **THEN** simulation produces complete electrothermal outputs without GUI participation
- **AND** all required physical calculations are backend-owned

### Requirement: YAML Coverage for Missing GUI Component Types

The YAML parser SHALL accept and validate all currently missing GUI component types:

- `BJT_NPN`, `BJT_PNP`, `THYRISTOR`, `TRIAC`, `SWITCH`
- `OP_AMP`, `COMPARATOR`
- `FUSE`, `CIRCUIT_BREAKER`, `RELAY`
- `PI_CONTROLLER`, `PID_CONTROLLER`, `MATH_BLOCK`, `PWM_GENERATOR`, `INTEGRATOR`, `DIFFERENTIATOR`, `LIMITER`, `RATE_LIMITER`, `HYSTERESIS`, `LOOKUP_TABLE`, `TRANSFER_FUNCTION`, `DELAY_BLOCK`, `SAMPLE_HOLD`, `STATE_MACHINE`
- `SATURABLE_INDUCTOR`, `COUPLED_INDUCTOR`, `SNUBBER_RC`
- `VOLTAGE_PROBE`, `CURRENT_PROBE`, `POWER_PROBE`, `ELECTRICAL_SCOPE`, `THERMAL_SCOPE`, `SIGNAL_MUX`, `SIGNAL_DEMUX`

#### Scenario: Parse full GUI-parity netlist
- **WHEN** a YAML netlist includes at least one instance of each listed component type with valid pins and parameters
- **THEN** parsing succeeds without unsupported-type errors
- **AND** each component is mapped to its backend runtime representation

### Requirement: Strict Pin and Parameter Validation for New Types

Strict mode SHALL validate pin count, required parameters, and parameter ranges for newly supported component types.

#### Scenario: Invalid relay pin configuration
- **WHEN** a relay component is defined with missing `COM/NO/NC` terminals
- **THEN** parsing fails with a diagnostic naming the missing pins

#### Scenario: Invalid saturation parameters
- **WHEN** a saturable inductor is defined with non-physical parameter combination
- **THEN** parsing fails with clear parameter-range diagnostics

### Requirement: Canonical Name and Alias Mapping

The parser SHALL support canonical backend names and GUI/YAML aliases for new component families while producing canonical internal representation.

#### Scenario: Alias normalization
- **WHEN** a netlist uses accepted aliases for a new component type
- **THEN** the parser normalizes the alias to canonical type identifier
- **AND** downstream runtime and diagnostics reference canonical naming

### Requirement: Converter Coverage Schema Completeness
The YAML schema SHALL represent all declared converter component and solver fields required by the support matrix.

#### Scenario: Declared converter case in YAML
- **WHEN** a netlist uses only fields listed in the declared converter support matrix
- **THEN** parsing succeeds in strict mode
- **AND** options map to v1 runtime without requiring legacy parsers

### Requirement: Electro-Thermal Configuration in YAML
The YAML format SHALL support explicit electro-thermal simulation configuration for declared workflows.

#### Scenario: Coupled electro-thermal netlist
- **WHEN** a netlist enables thermal coupling settings and thermal model parameters
- **THEN** parser maps these settings to v1 simulation options and thermal runtime configuration
- **AND** invalid thermal fields produce strict diagnostics

### Requirement: Explicit Legacy JSON Rejection Diagnostics
The YAML parser SHALL emit actionable diagnostics when users provide legacy JSON netlists in supported workflows.

#### Scenario: JSON netlist provided to YAML loader
- **WHEN** a user passes a JSON-format netlist to the v1 YAML parser path
- **THEN** parsing fails with a message that JSON netlists are unsupported
- **AND** the diagnostic points to the YAML migration guidance

### Requirement: Canonical Timestep Mode Field
The YAML schema SHALL define a canonical transient mode selector at `simulation.step_mode` with accepted values `fixed` and `variable`.

#### Scenario: Fixed mode selection in YAML
- **WHEN** `simulation.step_mode: fixed` is provided
- **THEN** the parser maps simulation options to fixed-step macro-grid semantics
- **AND** validates required fixed-mode fields

#### Scenario: Variable mode selection in YAML
- **WHEN** `simulation.step_mode: variable` is provided
- **THEN** the parser maps simulation options to adaptive-step semantics
- **AND** validates required variable-mode fields

### Requirement: Mode-Derived Default Profiles
The parser SHALL apply deterministic solver/integrator default profiles derived from `simulation.step_mode`.

#### Scenario: Fixed mode with minimal fields
- **WHEN** fixed mode is provided with only required timing fields
- **THEN** the parser applies fixed-mode default solver policies
- **AND** emits no requirement for expert-only tuning fields

#### Scenario: Variable mode with minimal fields
- **WHEN** variable mode is provided with only required timing fields
- **THEN** the parser applies adaptive-mode default solver policies
- **AND** initializes adaptive controller defaults deterministically

### Requirement: Hybrid Segment-Solver Defaults
The parser SHALL map canonical timestep modes to hybrid segment-solver defaults without requiring backend-selection fields.

#### Scenario: Fixed mode enables deterministic hybrid profile
- **WHEN** `simulation.step_mode: fixed` is configured
- **THEN** runtime options select fixed macro-grid semantics with state-space segment-first policy
- **AND** nonlinear DAE fallback remains enabled as internal recovery path

#### Scenario: Variable mode enables adaptive hybrid profile
- **WHEN** `simulation.step_mode: variable` is configured
- **THEN** runtime options select adaptive semantics with state-space segment-first policy
- **AND** nonlinear DAE fallback remains enabled as internal recovery path

### Requirement: Expert Override Namespace
Advanced solver tuning keys SHALL be accepted under an explicit expert namespace without changing canonical mode semantics.

#### Scenario: Expert override in fixed mode
- **WHEN** YAML includes expert override keys under the documented expert namespace
- **THEN** overrides are applied on top of fixed-mode defaults
- **AND** unsupported expert keys fail with strict diagnostics in strict mode

### Requirement: Legacy Backend-Key Removal Diagnostics
Legacy transient-backend keys in YAML SHALL produce deterministic migration diagnostics in the supported schema.

#### Scenario: Deprecated backend key in strict mode
- **WHEN** YAML contains deprecated backend-selection keys removed by this change
- **THEN** parsing fails with a diagnostic that names the deprecated key
- **AND** suggests `simulation.step_mode` migration mapping

### Requirement: Electrothermal and Loss Configuration Compatibility
The YAML schema SHALL support losses and thermal configuration in canonical mode-based runs.

#### Scenario: Global thermal policy in canonical mode
- **WHEN** `simulation.thermal` is provided with canonical `step_mode`
- **THEN** parser accepts and maps thermal policy fields deterministically
- **AND** rejects invalid thermal policy names with structured diagnostics

#### Scenario: Per-component loss and thermal parameters
- **WHEN** a component provides `loss` and/or `thermal` blocks
- **THEN** parser maps those fields into runtime loss/thermal configuration
- **AND** runtime enables thermal processing automatically when thermal blocks are present

### Requirement: Thermal-Port YAML Contract
The YAML schema SHALL define `component.thermal` as the thermal-port configuration block with deterministic validation and diagnostics.

#### Scenario: Valid thermal-port configuration
- **WHEN** a component defines `thermal.enabled=true` with valid thermal constants
- **THEN** the parser maps `rth`, `cth`, `temp_init`, `temp_ref`, and `alpha` into runtime thermal-device configuration
- **AND** the thermal port is activated for that component

#### Scenario: Unsupported thermal port activation
- **WHEN** `component.thermal.enabled=true` is declared for a component type without thermal capability
- **THEN** parsing fails with deterministic unsupported-thermal-port diagnostics

### Requirement: Strict and Non-Strict Missing-Parameter Policy
The parser SHALL support explicit strict-mode behavior for missing required thermal constants when a thermal port is enabled.

#### Scenario: Missing thermal constants in strict mode
- **GIVEN** strict parsing is enabled
- **WHEN** `component.thermal.enabled=true` and `rth` or `cth` is missing
- **THEN** parsing fails with field-path diagnostics identifying missing keys

#### Scenario: Missing thermal constants in non-strict mode
- **GIVEN** strict parsing is disabled
- **WHEN** `component.thermal.enabled=true` and `rth` or `cth` is missing
- **THEN** missing values are defaulted from `simulation.thermal.default_rth/default_cth`
- **AND** deterministic warning diagnostics are emitted

### Requirement: Thermal Constant Range Validation
Thermal-port constants SHALL be validated for finite numeric ranges.

#### Scenario: Invalid thermal numeric ranges
- **WHEN** YAML thermal fields contain non-finite values, `rth <= 0`, or `cth < 0`
- **THEN** parsing fails with deterministic range diagnostics
- **AND** diagnostics include the exact YAML field path

