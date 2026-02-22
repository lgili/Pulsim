## ADDED Requirements

### Requirement: GUI Power Semiconductor Parity Set

The device-model layer SHALL support GUI power semiconductor components that are currently unsupported: `BJT_NPN`, `BJT_PNP`, `THYRISTOR`, and `TRIAC`.

#### Scenario: Mixed semiconductor schematic executes without unsupported-component errors
- **GIVEN** a circuit containing NPN, PNP, SCR, and TRIAC devices with valid pin wiring and parameters
- **WHEN** the circuit is built and simulated
- **THEN** backend runtime model instantiation succeeds for each device
- **AND** no unsupported-component diagnostic is emitted for those component types

#### Scenario: SCR latching behavior
- **GIVEN** an SCR receives gate trigger current above threshold and anode current above holding current
- **WHEN** transient simulation progresses
- **THEN** the SCR enters conduction and remains latched while holding-current condition is satisfied

#### Scenario: TRIAC bidirectional conduction
- **GIVEN** a TRIAC with alternating main-terminal polarity
- **WHEN** gate trigger condition is met
- **THEN** conduction is supported in both current directions according to TRIAC model rules

### Requirement: Protection Device Behavioral Models

The device-model layer SHALL support `FUSE`, `CIRCUIT_BREAKER`, and `RELAY` with explicit state-transition behavior.

#### Scenario: Fuse I²t trip
- **GIVEN** a fuse configured with `rating` and `blow_i2t`
- **WHEN** accumulated current stress exceeds trip threshold
- **THEN** the fuse transitions to open state
- **AND** subsequent conduction follows configured open-state behavior

#### Scenario: Circuit breaker delayed trip
- **GIVEN** a breaker configured with `trip_current` and `trip_time`
- **WHEN** current exceeds threshold for at least the configured duration
- **THEN** the breaker transitions to tripped/open state

#### Scenario: Relay coil/contact coupling
- **GIVEN** a relay with coil and `COM/NO/NC` terminals
- **WHEN** coil excitation crosses pickup/dropout thresholds
- **THEN** contact state changes deterministically between NO and NC paths

### Requirement: Magnetic and Network Component Parity

The device-model layer SHALL support `SATURABLE_INDUCTOR`, `COUPLED_INDUCTOR`, and `SNUBBER_RC`.

#### Scenario: Saturable inductor current-dependent inductance
- **GIVEN** a saturable inductor with `inductance`, `saturation_current`, and `saturation_inductance`
- **WHEN** branch current crosses saturation threshold
- **THEN** effective inductance transitions according to model definition

#### Scenario: Coupled inductor mutual coupling
- **GIVEN** a coupled inductor with valid `l1`, `l2`, and coupling/mutual parameters
- **WHEN** transient simulation runs
- **THEN** coupling terms are applied so each winding influences the other according to configured coupling

#### Scenario: Snubber RC macro behavior
- **GIVEN** a snubber RC component across two nodes
- **WHEN** the circuit is assembled
- **THEN** the backend realizes equivalent R-C behavior consistent with canonical snubber topology

### Requirement: Analog and Control Block Model Coverage

The backend model layer SHALL provide behavioral support for GUI analog/control blocks: `OP_AMP`, `COMPARATOR`, `PI_CONTROLLER`, `PID_CONTROLLER`, `MATH_BLOCK`, `PWM_GENERATOR`, `INTEGRATOR`, `DIFFERENTIATOR`, `LIMITER`, `RATE_LIMITER`, `HYSTERESIS`, `LOOKUP_TABLE`, `TRANSFER_FUNCTION`, `DELAY_BLOCK`, `SAMPLE_HOLD`, and `STATE_MACHINE`.

#### Scenario: Closed-loop control chain
- **GIVEN** a control chain with PI/PID, limiter/rate limiter, and PWM generator
- **WHEN** simulation executes
- **THEN** each block updates output deterministically from configured parameters
- **AND** outputs can drive downstream switching/control elements

#### Scenario: Op-amp/comparator nonlinear limits
- **GIVEN** op-amp/comparator blocks with saturation or hysteresis settings
- **WHEN** inputs exceed threshold/rail conditions
- **THEN** outputs follow configured limiting and hysteresis behavior
