## ADDED Requirements
### Requirement: SUNDIALS Transient Backend
The v1 kernel SHALL provide a SUNDIALS transient backend with selectable solver family support for IDA, CVODE, and ARKODE when compiled with SUNDIALS.

#### Scenario: IDA backend selected for DAE transient
- **WHEN** transient backend mode is configured to SUNDIALS with solver family `IDA`
- **THEN** the simulator SHALL execute the transient using SUNDIALS IDA callbacks
- **AND** the result SHALL include backend telemetry identifying IDA as the active solver family

#### Scenario: SUNDIALS unavailable at build time
- **WHEN** backend mode requests SUNDIALS but the binary was compiled without SUNDIALS
- **THEN** the simulator SHALL fail deterministically with an explicit backend-unavailable diagnostic
- **AND** no undefined behavior or silent fallback SHALL occur

### Requirement: Deterministic Native-to-SUNDIALS Escalation
The v1 kernel SHALL support deterministic escalation from native transient integration to SUNDIALS based on configured retry thresholds.

#### Scenario: Native retries exhausted in auto mode
- **WHEN** backend mode is `Auto` and native transient retries exceed configured threshold
- **THEN** the simulator SHALL reinitialize and continue using configured SUNDIALS solver family
- **AND** fallback trace SHALL record backend escalation with deterministic reason code and action text

#### Scenario: Auto mode succeeds without escalation
- **WHEN** native transient integration converges within configured thresholds
- **THEN** SUNDIALS SHALL NOT be invoked
- **AND** telemetry SHALL report native backend as final execution path

### Requirement: SUNDIALS Backend Telemetry
The v1 kernel SHALL expose structured SUNDIALS telemetry counters in simulation results.

#### Scenario: Successful SUNDIALS run
- **WHEN** a transient run completes with SUNDIALS backend
- **THEN** telemetry SHALL include backend name, solver family, nonlinear iteration counters, and backend recovery/reinitialization counters
- **AND** telemetry SHALL be accessible alongside existing linear/nonlinear telemetry fields

#### Scenario: Failed SUNDIALS run
- **WHEN** SUNDIALS transient execution fails
- **THEN** result status/message SHALL include mapped solver failure reason
- **AND** fallback trace SHALL include the final backend failure event
