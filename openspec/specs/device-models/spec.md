# device-models Specification

## Purpose
TBD - created by archiving change improve-convergence-algorithms. Update Purpose after archive.
## Requirements
### Requirement: Diode Stamp with Limiting

The diode stamp function SHALL apply voltage limiting before computing current and conductance.

The stamp SHALL:
- Retrieve previous diode voltage from device state
- Apply voltage limiting to new voltage
- Compute I and G using limited voltage
- Store new voltage in device state

#### Scenario: Diode stamp with limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_diode() is called with V_new from Newton
- **THEN** V_limited = limit_diode_voltage(V_new, V_old)
- **AND** I and G are computed using V_limited
- **AND** V_old is updated to V_limited

### Requirement: MOSFET Stamp with Limiting

The MOSFET stamp function SHALL apply voltage limiting before computing currents.

#### Scenario: MOSFET stamp with Vgs and Vds limiting

- **GIVEN** MNA assembly with voltage limiting enabled
- **WHEN** stamp_mosfet() is called
- **THEN** Vgs_limited = limit_mosfet_vgs(Vgs_new, Vgs_old)
- **AND** Vds_limited = limit_mosfet_vds(Vds_new, Vds_old)
- **AND** drain current is computed using limited voltages

### Requirement: Modular Component Model Library
The system SHALL define each built-in electrical component model in a dedicated component file under a stable component-library path, expose model integration through registry/module contracts, and preserve compatibility aggregator includes for legacy callers.

#### Scenario: Legacy include compatibility after modularization
- **GIVEN** existing code that includes `pulsim/v1/device_base.hpp`
- **WHEN** the project is built after model modularization
- **THEN** all existing built-in component types remain available
- **AND** no caller migration is required for include-path compatibility

#### Scenario: Isolated model evolution per component
- **GIVEN** a change to one component model file
- **WHEN** tests and benchmarks are executed
- **THEN** only that component module and declared integration paths are impacted
- **AND** unrelated models do not require structural edits in the same file

#### Scenario: Registry-driven model integration
- **GIVEN** a new component model that satisfies device contract requirements
- **WHEN** it is added through model registry/module contracts
- **THEN** runtime discovers the model without mandatory edits in central orchestrator files
- **AND** incompatible model metadata is rejected with deterministic diagnostics

### Requirement: Controlled Numerical Regularization for Switching Models
The system SHALL support controlled, bounded numerical regularization for switching/nonlinear component models to improve convergence in pathological switching regimes without unbounded physical distortion.

#### Scenario: Automatic regularization in repeated switching-step failure
- **GIVEN** repeated transient failures near switching discontinuities
- **WHEN** recovery policy escalates through configured stages
- **THEN** bounded regularization parameters are applied to eligible component models
- **AND** each escalation is recorded in structured telemetry

#### Scenario: Regularization bounded by policy limits
- **GIVEN** auto-regularization is active
- **WHEN** the solver escalates regularization intensity
- **THEN** configured maximum bounds are never exceeded
- **AND** simulation aborts with typed diagnostics if convergence still fails

### Requirement: Semiconductor Loss Characterization Profiles
Thermal-capable semiconductor device models SHALL support explicit loss-characterization profiles for conduction and switching phenomena.

#### Scenario: MOSFET/IGBT datasheet characterization profile
- **GIVEN** a MOSFET or IGBT instance with datasheet-grade characterization
- **WHEN** runtime evaluates losses
- **THEN** conduction and switching terms are evaluated from the configured profile deterministically
- **AND** profile parameters are available to telemetry and diagnostics with stable identity

#### Scenario: Diode characterization profile with reverse recovery
- **GIVEN** a diode instance with reverse-recovery characterization
- **WHEN** runtime observes valid transition conditions
- **THEN** reverse-recovery energy is computed from profile data and included in loss decomposition

### Requirement: Gate-Condition and Calibration Scaling Semantics
Semiconductor loss profiles SHALL support deterministic scaling semantics for gate-condition or calibration factors.

#### Scenario: Gate-resistance scaling on switching energy
- **GIVEN** a profile with gate-condition reference and scaling coefficients
- **WHEN** runtime evaluates switching energy under configured gate conditions
- **THEN** scaling is applied deterministically before energy commit
- **AND** applied scaling factors are bounded by configured policy limits

### Requirement: Thermal Network Model Families per Component
Thermal-capable models SHALL support `single_rc`, `foster`, and `cauer` thermal-network families with consistent state semantics.

#### Scenario: Network family selection
- **WHEN** a component selects thermal network family in configuration
- **THEN** runtime instantiates matching network-state semantics for that component
- **AND** unsupported family selection fails validation deterministically

### Requirement: Deterministic Out-of-Range Handling for Loss Surfaces
Loss-surface model evaluation SHALL define deterministic out-of-range behavior for operating-variable queries.

#### Scenario: Operating point outside table bounds
- **WHEN** runtime queries a loss surface outside configured axis bounds
- **THEN** evaluation follows configured policy (for example clamp) deterministically
- **AND** strict policy modes can fail with typed diagnostics instead of silent extrapolation

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

### Requirement: Converter Power Device Coverage
Device models SHALL cover declared converter power switch classes with stable nonlinear stamping behavior.

#### Scenario: Declared power switch models in one converter case
- **WHEN** a converter uses supported diode, switch, MOSFET, and IGBT models
- **THEN** the simulation converges using v1 model implementations
- **AND** model-specific diagnostics are available for failed convergence

### Requirement: Electro-Thermal Loss Coupling
Device models SHALL support coupling between electrical loss calculations and thermal state updates for declared workflows.

#### Scenario: Thermal feedback enabled
- **WHEN** thermal feedback is enabled for supported devices
- **THEN** per-device losses update thermal states
- **AND** temperature-dependent device parameters are updated according to configured rules

### Requirement: External SPICE Calibration Envelope
Declared converter device-model workflows SHALL define calibration and validation envelopes against external SPICE references.

#### Scenario: LTspice parity check for device waveform
- **WHEN** a supported converter benchmark is compared to LTspice
- **THEN** waveform error metrics are checked against configured thresholds
- **AND** failures report which device-model observables exceeded limits

### Requirement: Loss Hooks for Segment and Event Commits
Device models SHALL expose deterministic loss contribution hooks for event transitions and accepted continuous segments.

#### Scenario: Switching-event loss contribution
- **WHEN** a switching-capable device commits an on/off transition event
- **THEN** the model returns switching-loss contribution terms for that event
- **AND** the runtime records the contribution exactly once per committed event

#### Scenario: Continuous-segment conduction loss contribution
- **WHEN** an accepted segment advances device currents/voltages
- **THEN** the model returns conduction-loss contribution terms for that segment
- **AND** rejected segment attempts do not contribute persistent loss energy

### Requirement: Temperature-Dependent Parameter Evaluation
Device models SHALL support deterministic temperature-dependent parameter evaluation for electrothermal coupling.

#### Scenario: Thermal state updates electrical parameters
- **WHEN** thermal coupling updates a device temperature state
- **THEN** the model applies bounded temperature scaling to configured parameters
- **AND** exposes the updated parameter state for the next accepted electrical segment

#### Scenario: Disabled thermal coupling
- **WHEN** thermal coupling is disabled for a device
- **THEN** temperature-dependent scaling is not applied
- **AND** the model remains numerically consistent with base electrical parameters

### Requirement: Thermal-Port Capability Declaration and Enforcement
Device models SHALL declare thermal-port capability, and this capability SHALL be enforced by parser/runtime validation.

#### Scenario: Thermal-capable model with thermal port enabled
- **GIVEN** a component type that declares thermal capability
- **WHEN** thermal port configuration is enabled for an instance
- **THEN** the runtime accepts thermal configuration and participates in electrothermal updates

#### Scenario: Non-thermal-capable model with thermal port enabled
- **GIVEN** a component type that declares no thermal capability
- **WHEN** thermal port configuration is enabled for an instance
- **THEN** the configuration is rejected with deterministic diagnostics

### Requirement: Consistent Thermal Parameter Semantics
Thermal-capable device models SHALL apply `rth`, `cth`, `temp_init`, `temp_ref`, and `alpha` with consistent electrothermal semantics.

#### Scenario: Loss-only thermal policy
- **WHEN** global policy is `LossOnly`
- **THEN** losses feed temperature evolution
- **AND** temperature-dependent electrical scaling is not applied

#### Scenario: Loss-with-temperature-scaling policy
- **WHEN** global policy is `LossWithTemperatureScaling`
- **THEN** losses feed temperature evolution
- **AND** bounded temperature-dependent scaling is applied using component thermal parameters

