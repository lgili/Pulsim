## ADDED Requirements

### Requirement: YAML Preset Field

The YAML parser SHALL recognize a `simulation.preset` field
accepting case-insensitive string values `auto`, `fast`, `robust`,
or `high_fidelity`.

When `simulation.preset` is set, the parser SHALL construct the
underlying `SimulationOptions` by calling the equivalent of
`SimulationOptions::from_preset(...)` and then apply any other
`simulation.*` fields as overrides ON TOP of the preset's
materialized defaults.

#### Scenario: Preset alone produces a complete configuration

- **GIVEN** a YAML file containing only `simulation: { preset: robust,
  tstop: 1e-3, dt: 1e-6 }`
- **WHEN** the parser loads the file
- **THEN** the resulting `SimulationOptions` matches
  `SimulationOptions::from_preset(Preset::Robust, 1e-6, 1e-3)`
- **AND** the parser emits no errors or warnings

#### Scenario: Explicit field overrides preset default

- **GIVEN** a YAML file with
  `simulation: { preset: robust, integrator: bdf1, tstop: 1e-3 }`
- **WHEN** the parser loads the file
- **THEN** the resulting `SimulationOptions` has the Robust preset's
  defaults for every field EXCEPT `integrator`, which is `BDF1`

#### Scenario: Unknown preset string rejected

- **GIVEN** a YAML file with `simulation.preset: super-mega-fast`
- **WHEN** the parser loads the file
- **THEN** an error is emitted with code
  `PULSIM_YAML_E_INVALID_PRESET`
- **AND** the simulation does not load

#### Scenario: Preset key is case-insensitive

- **GIVEN** YAML files with `simulation.preset: Robust`,
  `simulation.preset: ROBUST`, and `simulation.preset: robust`
- **WHEN** each file is parsed
- **THEN** all three produce the same `SimulationOptions`

### Requirement: YAML Advanced Namespace

The YAML parser SHALL recognize a `simulation.advanced` block
containing sub-blocks `newton`, `timestep`, `lte`, `bdf_order`,
`dc`, `stiffness`, `fallback`, `formulation`, and `linear_solver`.
Fields within each sub-block SHALL match the corresponding C++
struct field names.

#### Scenario: Advanced namespace round-trips through parser

- **GIVEN** YAML containing
  `simulation: { advanced: { newton: { max_iterations: 100 } } }`
- **WHEN** the parser loads the file
- **THEN** the resulting `SimulationOptions::advanced::newton::max_iterations`
  equals 100

#### Scenario: Legacy flat fields still parse with deprecation warning

- **GIVEN** YAML containing
  `simulation: { newton_options: { max_iterations: 100 } }`
- **WHEN** the parser loads the file
- **THEN** the value is forwarded to
  `SimulationOptions::advanced::newton::max_iterations`
- **AND** the parser emits a warning with code
  `PULSIM_YAML_W_DEPRECATED_FIELD` referencing the new path

### Requirement: Simulation Block Schema

The YAML `simulation:` block SHALL recognize the following top-level
fields and no others (additional keys SHALL trigger a
`PULSIM_YAML_W_UNKNOWN_FIELD` warning in strict mode):

- `preset` (optional, picks a `Preset` value)
- `tstart`, `tstop`, `dt`, `dt_min`, `dt_max` (timing)
- `step_mode` (`fixed | variable`)
- `switching_mode` (`auto | ideal | behavioral`)
- `integrator` (one of the curated 5 values:
  `trapezoidal | bdf1 | bdf2 | trbdf2 | rosenbrockw`)
- `linear_solver` (one of `auto | direct | iterative` — the
  collapsed public enum)
- `enable_losses`, `enable_events`, `enable_bdf_order_control`
- `advanced` (nested namespace, see ADDED requirement above)

Removed-from-top-level (still recognized as deprecated for one
release): `newton_options`, `timestep_config`, `lte_config`,
`bdf_config`, `dc_config`, `stiffness_config`, `fallback_policy`,
`formulation`, `adaptive_timestep`, `direct_formulation_fallback`.

#### Scenario: Slim top-level shape

- **GIVEN** a minimal valid simulation block
- **WHEN** the parser loads
- **THEN** the user can express any common configuration using only
  `preset`, `tstart`/`tstop`/`dt`, and at most one of
  `switching_mode`/`integrator`/`linear_solver`

#### Scenario: Deprecated bool field emits warning

- **GIVEN** a YAML file with `simulation.adaptive_timestep: true`
- **WHEN** the parser loads the file
- **THEN** the equivalent of `simulation.step_mode: variable` is
  applied
- **AND** a warning is emitted with code
  `PULSIM_YAML_W_DEPRECATED_FIELD` recommending `step_mode` instead

#### Scenario: Removed integrator value emits warning in v0.11

- **GIVEN** a YAML file with `simulation.integrator: bdf5`
- **WHEN** the parser loads the file in v0.11
- **THEN** the integrator is set to `Integrator::BDF5` (still
  supported in the deprecation window)
- **AND** a warning is emitted with code
  `PULSIM_YAML_W_DEPRECATED_FIELD` indicating BDF5 will be removed
  in v0.12 and recommending `trbdf2` or `rosenbrockw`

#### Scenario: Removed integrator value fails in v0.12

- **GIVEN** a YAML file with `simulation.integrator: bdf5`
- **WHEN** the parser loads the file in v0.12 (after the removal
  cycle)
- **THEN** an error is emitted with code
  `PULSIM_YAML_E_UNSUPPORTED_INTEGRATOR`
- **AND** the simulation does not load

### Requirement: Integrator YAML Vocabulary

The YAML parser SHALL accept exactly five integrator values in
canonical YAML form: `trapezoidal`, `bdf1`, `bdf2`, `trbdf2`,
`rosenbrockw` (case-insensitive).

The values `bdf3`, `bdf4`, `bdf5`, `gear`, `sdirk2` SHALL be
recognized with a deprecation warning in v0.11 and SHALL be
rejected with an error in v0.12.

#### Scenario: Curated integrator names parse cleanly

- **GIVEN** a YAML file with `simulation.integrator: trbdf2`
- **WHEN** the parser loads
- **THEN** the integrator is set
- **AND** no warnings or errors are emitted

#### Scenario: Case-insensitivity

- **GIVEN** YAML files with `integrator: TRBDF2`, `integrator: trbdf2`,
  `integrator: TrBdf2`
- **WHEN** each file is parsed
- **THEN** all three produce `Integrator::TRBDF2`
