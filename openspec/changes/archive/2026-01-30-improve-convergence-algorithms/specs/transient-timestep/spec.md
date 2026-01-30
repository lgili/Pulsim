# Transient Timestep Control

## ADDED Requirements

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

The system SHALL implement an adaptive timestep controller that combines LTE-based and Newton-iteration-based feedback.

The controller SHALL:
- Adjust timestep based on LTE estimate vs tolerance
- Reduce timestep when Newton iterations exceed target
- Increase timestep when Newton iterations are below threshold
- Apply smoothing to prevent timestep oscillation

#### Scenario: LTE-based timestep reduction

- **GIVEN** current timestep dt with computed LTE > tolerance
- **WHEN** the controller suggests next timestep
- **THEN** the new timestep is reduced by factor `(tolerance / LTE)^(1/3)`
- **AND** a safety factor of 0.9 is applied

#### Scenario: LTE-based timestep increase

- **GIVEN** current timestep dt with computed LTE < 0.1 * tolerance
- **WHEN** the controller suggests next timestep
- **THEN** the new timestep may be increased
- **AND** increase is limited to factor of 1.5

#### Scenario: Newton iteration feedback - slow convergence

- **GIVEN** target Newton iterations = 5 and actual iterations = 10
- **WHEN** the controller suggests next timestep
- **THEN** timestep is reduced by factor of 0.5
- **AND** this overrides LTE-based increase

#### Scenario: Newton iteration feedback - fast convergence

- **GIVEN** target Newton iterations = 5 and actual iterations = 2
- **WHEN** the controller suggests next timestep
- **THEN** timestep may be increased by factor up to 1.5
- **AND** this is combined with LTE-based adjustment

#### Scenario: Timestep smoothing

- **GIVEN** suggested timestep is 5x larger than previous timestep
- **WHEN** smoothing is applied
- **THEN** the actual timestep increase is limited to 2x
- **AND** gradual ramp-up prevents oscillation

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
