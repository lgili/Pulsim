## ADDED Requirements
### Requirement: Event-Burst and Zero-Cross Guarding
Transient timestep control SHALL include dedicated guards for event bursts and zero-crossing regions.

#### Scenario: Event burst inside a macro interval
- **WHEN** multiple switching boundaries are detected in a short temporal window
- **THEN** the controller activates burst guard policy
- **AND** clips steps using deterministic limits that avoid runaway retry loops

#### Scenario: Zero-crossing chattering prevention
- **WHEN** threshold toggles oscillate around a crossing boundary
- **THEN** temporal hysteresis and anti-chattering policy are applied
- **AND** the simulator avoids alternating micro-steps without physical progress

### Requirement: Context-Aware LTE/Newton Arbitration
The timestep controller SHALL arbitrate LTE and Newton feedback according to event context.

#### Scenario: LTE indicates reject near hard discontinuity
- **WHEN** LTE indicates rejection within an event-adjacent guard window
- **THEN** controller applies event-context arbitration policy
- **AND** avoids pathological reject loops caused by discontinuity-local LTE artifacts

#### Scenario: Smooth window after event cluster
- **WHEN** simulation exits event-adjacent window and nonlinear health is restored
- **THEN** controller returns to nominal adaptive growth policy with deterministic ramp limits
