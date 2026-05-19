## ADDED Requirements

### Requirement: Curated Integrator Set

The system SHALL expose exactly five integrator choices in its
public `Integrator` enumeration: `Trapezoidal`, `BDF1`, `BDF2`,
`TRBDF2`, and `RosenbrockW`. These represent the integrators that
have validated benchmark coverage on power-electronics topologies.

#### Scenario: Trapezoidal as second-order A-stable default

- **GIVEN** a user constructs `SimulationOptions{}` with no preset
- **WHEN** they inspect `opts.integrator`
- **THEN** the value is `Integrator::Trapezoidal`

#### Scenario: TRBDF2 as the preset Robust default

- **GIVEN** a user calls `SimulationOptions::from_preset(
  Preset::Robust, dt, tstop)`
- **WHEN** they inspect `opts.integrator`
- **THEN** the value is `Integrator::TRBDF2`

#### Scenario: BDF1 as the stiffness fallback

- **GIVEN** a transient run with stiffness detection enabled
- **WHEN** the stiffness detector triggers integrator switching
- **THEN** the integrator switches to `Integrator::BDF1`

## MODIFIED Requirements

### Requirement: Timestep Controller

The system SHALL implement an adaptive timestep controller that
combines LTE-based and Newton-iteration-based feedback. The
controller SHALL adjust the timestep based on the LTE estimate
versus tolerance, reduce the timestep when Newton iterations exceed
the target, increase the timestep when Newton iterations fall below
the minimum threshold, and apply smoothing to prevent timestep
oscillation.

The controller's tuning fields SHALL live under
`opts.advanced.timestep.*` (per the AdvancedOptions namespace
reorganization). The legacy boolean
`SimulationOptions::adaptive_timestep` SHALL be deprecated; the
canonical selector is `opts.step_mode = TransientStepMode::Variable
| TransientStepMode::Fixed`.

#### Scenario: LTE-based timestep reduction

- **GIVEN** a transient simulation where the LTE estimate exceeds
  the tolerance
- **WHEN** the timestep controller runs
- **THEN** the next timestep is reduced

#### Scenario: Newton-feedback timestep growth

- **GIVEN** a transient simulation where Newton consistently
  converges in fewer than the target iteration count
- **WHEN** the timestep controller runs
- **THEN** the next timestep grows by a bounded factor

#### Scenario: Step mode is the canonical enable switch

- **GIVEN** a user sets `opts.step_mode = TransientStepMode::Fixed`
- **WHEN** the simulator runs
- **THEN** the timestep controller is bypassed and the user's `dt`
  is used unchanged

#### Scenario: Legacy adaptive_timestep emits deprecation warning

- **GIVEN** legacy user code that sets `opts.adaptive_timestep = true`
- **WHEN** the field is accessed
- **THEN** the equivalent `opts.step_mode = TransientStepMode::Variable`
  is applied
- **AND** a deprecation warning is logged once per process

## ADDED Requirements

### Requirement: Deprecated Integrator Enum Values

The `Integrator` enum SHALL mark `BDF3`, `BDF4`, `BDF5`, `Gear`, and
`SDIRK2` as deprecated for one release cycle (v0.11), then SHALL
remove them in v0.12. Users explicitly selecting any of these values
SHALL receive a compile-time `[[deprecated]]` warning (C++) and a
YAML parser `PULSIM_YAML_W_DEPRECATED_FIELD` warning.

**Rationale**:
- `BDF3 / BDF4 / BDF5`: not A-stable for stiff switching topologies;
  zero benchmark coverage; oscillations on diode commutation make them
  unsafe defaults. Replacement: `TRBDF2` (order 2, A-stable) or
  `RosenbrockW`.
- `Gear`: literal alias for `BDF2` (zero behavioural difference).
  Replacement: `BDF2`.
- `SDIRK2`: research-grade implementation with zero benchmark coverage
  and no production validation. Replacement: `TRBDF2`.

#### Scenario: Deprecated integrator emits warning in v0.11

- **GIVEN** a user selects `Integrator::BDF5` (or `BDF3`, `BDF4`,
  `Gear`, `SDIRK2`) in v0.11
- **WHEN** the code compiles
- **THEN** the C++ compiler emits a `[[deprecated]]` warning with the
  suggested replacement
- **AND** the integrator still works at runtime (deprecation cycle)

#### Scenario: Deprecated integrator removed in v0.12

- **GIVEN** the same code in v0.12 (after the removal cycle)
- **WHEN** the code compiles
- **THEN** the enum value no longer exists and the compile fails
- **AND** the migration message points the user at the replacement
