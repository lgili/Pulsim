## ADDED Requirements

### Requirement: Event Detection
The kernel SHALL detect discrete events during simulation.

#### Scenario: Zero-crossing detection
- **WHEN** a monitored signal crosses zero
- **THEN** the exact crossing time is located via interpolation
- **AND** the event is scheduled for processing

#### Scenario: Threshold crossing
- **WHEN** a signal crosses a threshold value
- **THEN** the crossing is detected like zero-crossing
- **AND** rising/falling edge can be distinguished

#### Scenario: Time-based events
- **WHEN** a time event is scheduled (e.g., PWM edge)
- **THEN** the simulator steps to that exact time
- **AND** multiple time events are ordered in a priority queue

### Requirement: Event Scheduling
The kernel SHALL maintain an event queue for discrete events.

#### Scenario: Event queue management
- **WHEN** events are scheduled
- **THEN** they are stored in a priority queue ordered by time
- **AND** equal-time events are ordered by priority

#### Scenario: Event insertion
- **WHEN** a new event is detected or scheduled
- **THEN** it is inserted into the queue
- **AND** duplicate events (same time, same source) are merged

#### Scenario: Event cancellation
- **WHEN** an event condition becomes invalid before execution
- **THEN** the event is cancelled
- **AND** no state change occurs

### Requirement: Event Processing
The kernel SHALL process events at their scheduled times.

#### Scenario: State discontinuity
- **WHEN** an event is processed (e.g., switch closes)
- **THEN** the simulation state may have a discontinuity
- **AND** integration is restarted from the new state

#### Scenario: Event iteration
- **WHEN** an event causes other events
- **THEN** cascading events at the same time are processed
- **AND** a maximum iteration count prevents infinite loops

#### Scenario: Post-event integration
- **WHEN** event processing completes
- **THEN** time integration resumes from the event time
- **AND** the new circuit topology is used

### Requirement: Switching Event Handling
The kernel SHALL handle power electronics switching events efficiently.

#### Scenario: Switch opening
- **WHEN** a switch opens
- **THEN** the switch resistance changes from Ron to Roff
- **AND** current through the switch is forced to near-zero

#### Scenario: Switch closing
- **WHEN** a switch closes
- **THEN** the switch resistance changes from Roff to Ron
- **AND** voltage across the switch is forced to near-zero

#### Scenario: Natural commutation
- **WHEN** a diode naturally commutates (current crosses zero)
- **THEN** the diode turns off
- **AND** the event is detected by zero-crossing of diode current

#### Scenario: Forced commutation
- **WHEN** a switch is commanded to change state
- **THEN** the switching time is determined by control signal
- **AND** any stored energy must be handled (snubber, clamping)

### Requirement: PWM Event Generation
The kernel SHALL support PWM waveform generation for switching.

#### Scenario: Fixed-frequency PWM
- **WHEN** a PWM source is defined with frequency and duty cycle
- **THEN** switching events are generated at carrier edges
- **AND** duty cycle can be constant or time-varying

#### Scenario: Carrier-based PWM
- **WHEN** PWM uses carrier comparison
- **THEN** events occur when modulating signal crosses carrier
- **AND** carrier can be sawtooth or triangle

#### Scenario: Space vector PWM
- **WHEN** SVPWM is used for three-phase
- **THEN** switching patterns follow space vector algorithm
- **AND** zero vectors are distributed optimally

### Requirement: Hybrid DAE Formulation
The kernel SHALL support hybrid continuous-discrete systems.

#### Scenario: Mode-dependent equations
- **WHEN** the circuit has multiple operating modes
- **THEN** each mode has its own DAE system
- **AND** mode transitions are triggered by events

#### Scenario: State consistency across modes
- **WHEN** a mode change occurs
- **THEN** state variables are mapped to the new mode
- **AND** algebraic constraints are re-initialized

#### Scenario: Event localization precision
- **WHEN** an event is detected between timesteps
- **THEN** bisection or interpolation locates the event time
- **AND** precision is controllable (default 1e-12 relative)

### Requirement: Event Logging
The kernel SHALL log events for analysis.

#### Scenario: Event recording
- **WHEN** an event occurs
- **THEN** it is recorded with: time, type, source, old/new state
- **AND** event log is available after simulation

#### Scenario: Event statistics
- **WHEN** simulation completes
- **THEN** event statistics are available: count per device, average period
- **AND** can detect abnormal switching patterns (e.g., chattering)
