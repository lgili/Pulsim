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

### Requirement: Top-Level Analysis Section
The YAML schema SHALL accept a top-level `analysis:` array, parallel to `simulation:`, allowing one or more frequency-domain analyses on the same netlist.

#### Scenario: Single AC analysis
- **GIVEN** a netlist with:
```yaml
analysis:
  - type: ac
    f_start: 1
    f_stop: 1e6
    points_per_decade: 20
    perturbation_source: Vin
    measurement_nodes: [vout]
```
- **WHEN** the parser loads
- **THEN** an `AcSweepOptions` is constructed and bound to the simulation
- **AND** the analysis runs after the transient (or alone if no transient is requested)

#### Scenario: AC and FRA combined
- **GIVEN** an `analysis:` array with one `ac` entry and one `fra` entry
- **WHEN** the simulator runs
- **THEN** both analyses execute sequentially using the same DC operating point
- **AND** results are emitted under named keys in the result bundle

### Requirement: Strict Validation of Analysis Configuration
Strict validation SHALL reject malformed analysis blocks with deterministic diagnostics.

#### Scenario: Unknown analysis type
- **GIVEN** `analysis: [{ type: bode_diagram }]` (not a recognized type)
- **WHEN** strict parsing runs
- **THEN** parsing fails with reason `unknown_analysis_type`
- **AND** the diagnostic suggests the supported types

#### Scenario: Inconsistent frequency range
- **GIVEN** an `ac` analysis with `f_start >= f_stop`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `invalid_frequency_range`
- **AND** the diagnostic shows the offending values

#### Scenario: Missing perturbation source
- **GIVEN** an analysis lacking `perturbation_source`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `missing_required_parameter` and the field name

### Requirement: Saturable Magnetic Component Types
The YAML schema SHALL accept `saturable_inductor` and `saturable_transformer` component types with magnetic-specific parameters.

#### Scenario: Saturable inductor with table B-H
- **GIVEN** a netlist:
```yaml
- type: saturable_inductor
  name: L1
  nodes: [n1, n2]
  bh_curve:
    type: table
    points: [[0, 0], [50, 0.1], [100, 0.18], [200, 0.22]]  # H[A/m], B[T]
  n_turns: 35
  area_m2: 1.5e-4
```
- **WHEN** the parser loads
- **THEN** the device is instantiated with the table-based B-H curve
- **AND** the magnetic effective parameters are computed from N and Ae

#### Scenario: Saturable transformer with core model reference
- **GIVEN** a netlist with `core_model: ferroxcube/N87` and per-winding leakage values
- **WHEN** the parser loads
- **THEN** the catalog material parameters are loaded
- **AND** Steinmetz parameters are wired from the catalog

#### Scenario: Inline Steinmetz override
- **GIVEN** a saturable device with both `core_model:` and inline `steinmetz: { k: 3.0 }`
- **WHEN** parser loads
- **THEN** the inline parameter overrides the catalog default
- **AND** the override is reflected in `BackendTelemetry.magnetic_overrides`

### Requirement: Hysteresis Configuration in YAML
The YAML schema SHALL support `simulation.hysteresis_model: none | jiles_atherton` and per-device `jiles_atherton:` parameter overrides.

#### Scenario: Global hysteresis off
- **GIVEN** a netlist with `simulation.hysteresis_model: none`
- **WHEN** parser loads
- **THEN** all saturable devices use anhysteretic curves only

#### Scenario: Per-device hysteresis enable
- **GIVEN** global `hysteresis_model: none` but a device with `jiles_atherton: {...}`
- **WHEN** parser loads
- **THEN** that specific device uses Jiles-Atherton; others remain anhysteretic
- **AND** strict validation reports the override in diagnostics

### Requirement: Strict Validation of Magnetic Parameters
Strict validation SHALL reject malformed magnetic parameters (non-monotonic B-H, negative Steinmetz coefficients, missing required fields) with deterministic diagnostics.

#### Scenario: Non-monotonic B-H table
- **GIVEN** a `bh_curve.points` with non-monotonic H values
- **WHEN** strict parsing runs
- **THEN** parsing fails with reason `non_monotonic_bh_curve`
- **AND** the diagnostic identifies the offending point indices

#### Scenario: Missing Ae or N
- **GIVEN** a `saturable_inductor` lacking `area_m2` or `n_turns`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `missing_required_parameter` and the field name

### Requirement: Catalog Device Component Types
The YAML netlist schema SHALL accept new component types `mosfet_catalog`, `igbt_catalog`, `diode_catalog`, each requiring a `model` reference or inline `parameters` block.

#### Scenario: Catalog device by model name
- **GIVEN** a netlist component:
```yaml
- type: mosfet_catalog
  name: Q1
  nodes: [drain, gate, source]
  model: wolfspeed/C3M0065090J
```
- **WHEN** the parser loads the netlist
- **THEN** the device is instantiated from the named catalog YAML
- **AND** the device participates in the simulation

#### Scenario: Catalog device with inline params
- **GIVEN** a netlist component with `parameters: { rds_on_25c: 19e-3, vth: 3.5, ... }`
- **WHEN** the parser loads
- **THEN** the device is instantiated with the inline parameter set
- **AND** missing required parameters produce a clear diagnostic

#### Scenario: Catalog device with override
- **GIVEN** a netlist component with both `model: vendor/part` and `parameters: { rds_on_25c: 25e-3 }`
- **WHEN** the parser loads
- **THEN** the inline parameter overrides the catalog default
- **AND** all other catalog parameters are preserved

### Requirement: Strict Validation of Catalog References
Strict validation SHALL fail with a deterministic diagnostic when a catalog `model:` reference cannot be resolved.

#### Scenario: Unknown model
- **GIVEN** a netlist with `model: vendor/nonexistent_part`
- **WHEN** strict mode parsing runs
- **THEN** parsing fails with reason `catalog_model_not_found`
- **AND** the diagnostic suggests `pulsim catalog list <vendor>` to inspect available parts

### Requirement: Catalog Parameter Schema in YAML
The YAML schema SHALL document required and optional parameters for each catalog device type, validated by strict mode.

#### Scenario: Missing required parameter
- **GIVEN** an inline `mosfet_catalog` lacking `rds_on_25c`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `missing_required_parameter` and the missing field name
- **AND** the diagnostic includes the schema YAML path

### Requirement: Motor and Mechanical Component Types
The YAML schema SHALL accept new component types for motors (`pmsm`, `induction_motor`, `bldc_motor`, `dc_motor`), mechanical (`shaft`, `gearbox`, `flywheel_load`, `constant_torque_load`, `fan_load`), sensors (`encoder_quadrature`, `hall_sensor`, `resolver`), and frame transforms (`abc_to_dq`, `dq_to_abc`).

#### Scenario: PMSM with shaft and load
- **GIVEN** a netlist:
```yaml
- type: pmsm
  name: M1
  electrical_nodes: [a, b, c, n]
  mechanical_port: shaft1
  parameters: { rs: 0.5, ld: 1e-3, lq: 1.5e-3, psi_pm: 0.18, pole_pairs: 4 }
- type: shaft
  name: shaft1
  parameters: { j: 0.001, b_friction: 0.0005 }
- type: flywheel_load
  name: load1
  mechanical_port: shaft1
  parameters: { j_load: 0.005 }
```
- **WHEN** the parser loads
- **THEN** the PMSM, shaft, and flywheel are connected via the shared mechanical port `shaft1`
- **AND** the simulation runs with the combined inertia

### Requirement: Mixed-Port Component Definition
The schema SHALL distinguish electrical from mechanical port lists explicitly via `electrical_nodes` and `mechanical_port` fields.

#### Scenario: Mismatch in port domain
- **GIVEN** a netlist linking `electrical_nodes` from a motor to a `shaft`'s mechanical port (incompatible)
- **WHEN** strict parsing runs
- **THEN** parsing fails with `port_domain_mismatch`
- **AND** the diagnostic indicates which entity expected which port domain

### Requirement: Strict Validation of Motor Parameters
Strict validation SHALL reject motor parameter combinations that are physically inconsistent.

#### Scenario: Negative inductance
- **GIVEN** a `pmsm` with `ld: -1e-3`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `invalid_parameter` and a value/range message

#### Scenario: Saturation table out of range
- **GIVEN** a PMSM with a saturation table that has `ld(id)` decreasing then increasing
- **WHEN** strict parsing runs
- **THEN** parsing fails with `non_monotonic_saturation`

### Requirement: Three-Phase Component Types
The YAML schema SHALL accept new types for three-phase sources, PLLs, frame transforms, sequence decomposition, anti-islanding blocks, and grid-tied inverter templates.

#### Scenario: Programmable three-phase source with events
- **GIVEN** a netlist:
```yaml
- type: three_phase_source_programmable
  name: Vgrid
  nodes_abc: [a, b, c]
  neutral: n
  parameters:
    v_rms: 230
    frequency: 50
    phase_seq: abc
    events:
      - { type: sag, phase: A, ratio: 0.5, t_start: 0.1, t_end: 0.2 }
```
- **WHEN** the parser loads
- **THEN** the source is built with the sag event programmed
- **AND** strict validation checks event ranges (`0 < ratio < 1.5`, `t_end > t_start`)

#### Scenario: Grid-following inverter template
- **GIVEN** `type: grid_following_inverter_template` with `vdc, p_ref, q_ref` parameters
- **WHEN** parser loads
- **THEN** the template expands into inverter + LCL + control sub-netlist
- **AND** auto-tuned defaults produce a stable closed loop

### Requirement: Strict Validation of Three-Phase Configurations
Strict validation SHALL detect inconsistent three-phase configurations.

#### Scenario: Mismatched phase sequence in connected components
- **GIVEN** a three-phase source with `phase_seq: abc` connected to a load expecting `acb`
- **WHEN** strict parsing runs
- **THEN** parsing fails or warns with `phase_sequence_mismatch` per configured policy
- **AND** the diagnostic suggests verifying `phase_seq` on each three-phase component

#### Scenario: Invalid PLL parameters
- **GIVEN** an `srf_pll` with `kp: -10` (negative)
- **WHEN** strict parsing runs
- **THEN** parsing fails with `invalid_parameter` and the field name

