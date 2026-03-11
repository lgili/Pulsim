## ADDED Requirements
### Requirement: Bounded Regularization Profiles per Device Family
Device model regularization SHALL be configured and applied per device family with explicit physical bounds.

#### Scenario: Diode-heavy bridge with repeated convergence stress
- **WHEN** convergence policy escalates regularization for diode family
- **THEN** diode regularization parameters remain within configured physical bounds
- **AND** applied intensity is emitted in structured telemetry

#### Scenario: Magnetic nonlinear saturation stress
- **WHEN** nonlinear magnetic devices trigger stiffness-related failures
- **THEN** magnetic-specific stabilization profile is applied deterministically
- **AND** model semantics remain bounded by declared safety envelopes

### Requirement: Regularization Auditability Contract
Every automatic model regularization action SHALL be auditable.

#### Scenario: Recovery action modifies model parameters
- **WHEN** regularization modifies effective model parameters for a retry
- **THEN** telemetry records component-family scope, bounded intensity, and escalation stage
- **AND** final run summary includes the maximum applied intensity per family
