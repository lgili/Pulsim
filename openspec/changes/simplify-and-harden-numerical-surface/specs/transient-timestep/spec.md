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

## REMOVED Requirements

### Requirement: BDF3-5 Integrators

**Reason**: `BDF3`, `BDF4`, and `BDF5` are theoretically supported
but are not A-stable for stiff switching topologies. They have zero
benchmark coverage in the Pulsim suite; analysis showed the
oscillations they produce on diode commutation and PWM edges make
them unsafe defaults for any power-electronics circuit.

**Migration**: Users explicitly selecting `Integrator::BDF3`,
`BDF4`, or `BDF5` SHALL receive a deprecation warning in v0.11.
v0.12 SHALL remove these enum values. Suggested replacement:
`Integrator::TRBDF2` for stiff dynamics requiring order ≥ 2, or
`Integrator::RosenbrockW` for explicitly stiff problems.

### Requirement: Gear Alias

**Reason**: The `Integrator::Gear` enum value is a literal alias
for `Integrator::BDF2` (verified in the C++ implementation). It
exists only as a back-compat hook from an early integration where
"Gear" was the documented name. Today it duplicates `BDF2` with no
behavioural difference.

**Migration**: Users selecting `Integrator::Gear` SHALL receive a
deprecation warning in v0.11. v0.12 SHALL remove the enum value.
Replacement: `Integrator::BDF2`.

### Requirement: SDIRK2 Integrator

**Reason**: The `Integrator::SDIRK2` (singly-diagonally-implicit
Runge-Kutta order 2) value exists in the enum but has zero benchmark
coverage and no documentation. The implementation is research-grade
and has not been validated on production circuits.

**Migration**: Users selecting `Integrator::SDIRK2` SHALL receive
a deprecation warning in v0.11. v0.12 SHALL remove the enum value.
Replacement: `Integrator::TRBDF2` (also second-order stiff-stable,
multi-stage, with full benchmark coverage).
