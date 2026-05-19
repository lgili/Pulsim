# transient-timestep Specification

## Purpose
TBD - created by archiving change improve-convergence-algorithms. Update Purpose after archive.
## Requirements
### Requirement: Richardson LTE Estimation

The system SHALL implement Richardson extrapolation for Local Truncation Error (LTE) estimation as an alternative to step-doubling.

Richardson LTE estimation SHALL:
- Use stored solution history (last 3 solutions)
- Compute LTE estimate without additional matrix solves
- Provide comparable accuracy to step-doubling
- Reduce computational cost by approximately 3x

#### Scenario: LTE computation with sufficient history

- **GIVEN** a transient simulation with at least 3 completed timesteps
- **WHEN** LTE is computed using Richardson extrapolation
- **THEN** the estimate uses polynomial extrapolation from history
- **AND** no additional Newton solves are performed
- **AND** the estimate is within 10% of step-doubling estimate

#### Scenario: LTE computation with insufficient history

- **GIVEN** a transient simulation in the first 2 timesteps
- **WHEN** LTE is computed
- **THEN** the system falls back to conservative estimate
- **AND** uses smaller timesteps until history is available

#### Scenario: Richardson vs step-doubling accuracy

- **GIVEN** a BDF2 integration with tolerance 1e-6
- **WHEN** both Richardson and step-doubling LTE are computed
- **THEN** Richardson estimate is within factor of 2 of step-doubling
- **AND** resulting timestep control produces similar accuracy

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

### Requirement: Event-Driven Timestep Control

The system SHALL detect switching events and adjust timesteps to hit event times precisely.

Event detection SHALL:
- Monitor control signals of all switches
- Detect threshold crossings between timesteps
- Use binary search to find exact crossing time
- Force timestep to land at event time

#### Scenario: Switch transition detection

- **GIVEN** a MOSFET switch with Vgs threshold = 5V
- **WHEN** Vgs changes from 4V to 6V between t1 and t2
- **THEN** a switch event is detected
- **AND** the crossing time is computed via bisection

#### Scenario: Timestep adjustment for event

- **GIVEN** current time = 10us and switch event at 12us
- **WHEN** the controller computes next timestep
- **THEN** the timestep is adjusted to land exactly at 12us
- **AND** the switch state is updated at the event time

#### Scenario: Multiple events in timestep

- **GIVEN** two switch events detected at 12us and 15us
- **WHEN** the controller computes next timestep
- **THEN** the timestep targets the earlier event (12us)
- **AND** subsequent step handles the 15us event

#### Scenario: PWM source breakpoints

- **GIVEN** a PWM source with period 10us and 50% duty cycle
- **WHEN** transient simulation starts
- **THEN** breakpoints are scheduled at 0, 5us, 10us, 15us, ...
- **AND** timesteps are adjusted to hit these breakpoints
- **AND** PWM edges are captured accurately

### Requirement: Solution History Management

The system SHALL maintain a history of recent solutions for LTE estimation and event detection.

#### Scenario: History buffer management

- **GIVEN** a transient simulation in progress
- **WHEN** each timestep completes
- **THEN** the solution vector is stored in a ring buffer
- **AND** buffer maintains last 3 solutions
- **AND** older solutions are discarded

#### Scenario: History with variable timesteps

- **GIVEN** timesteps [1us, 0.5us, 2us] with corresponding solutions
- **WHEN** LTE is computed
- **THEN** the computation accounts for non-uniform timesteps
- **AND** polynomial extrapolation uses correct time coefficients

### Requirement: Transient Options Extension

The system SHALL extend `TransientOptions` with new timestep control parameters.

New options SHALL include:
- `timestep_method`: Richardson or StepDoubling (default: Richardson)
- `event_detection`: Enable event detection (default: true)
- `event_tolerance`: Tolerance for event time bisection (default: 1e-9)
- `target_newton_iters`: Target Newton iterations (default: 5)
- `timestep_increase_factor`: Max increase per step (default: 1.5)
- `timestep_decrease_factor`: Decrease factor on slow convergence (default: 0.5)

#### Scenario: Use step-doubling for compatibility

- **GIVEN** TransientOptions with `timestep_method = StepDoubling`
- **WHEN** transient simulation runs
- **THEN** LTE is computed using step-doubling
- **AND** behavior matches previous implementation

#### Scenario: Disable event detection

- **GIVEN** TransientOptions with `event_detection = false`
- **WHEN** transient simulation runs
- **THEN** switch events are not detected
- **AND** timesteps are controlled only by LTE and Newton feedback
- **AND** simulation runs faster but may miss sharp transitions

#### Scenario: Aggressive timestep increase

- **GIVEN** TransientOptions with `timestep_increase_factor = 2.0`
- **WHEN** LTE and Newton iterations allow increase
- **THEN** timestep may double between steps
- **AND** simulation completes faster with acceptable accuracy loss

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

