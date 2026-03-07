## ADDED Requirements
### Requirement: Direct SUNDIALS DAE Formulation
The v1 kernel SHALL provide a direct SUNDIALS formulation that evaluates transient equations from runtime circuit residual/Jacobian callbacks without using the projected-wrapper approximation.

#### Scenario: Direct IDA on stiff switched converter
- **WHEN** `simulation.sundials.formulation` is set to `direct` with family `ida`
- **THEN** the simulator uses direct residual/Jacobian callbacks from runtime circuit assembly
- **AND** does not route through `project_rhs` projected-wrapper callbacks

#### Scenario: Direct formulation fallback
- **WHEN** direct SUNDIALS formulation fails under configured retry/fallback policy
- **THEN** the simulator records a deterministic fallback reason
- **AND** may fall back to projected-wrapper or native backend according to configured order

### Requirement: SUNDIALS Formulation Telemetry
The v1 kernel SHALL report SUNDIALS formulation mode and solver-internal counters needed for accuracy/performance validation.

#### Scenario: Telemetry after direct SUNDIALS run
- **WHEN** a transient simulation completes with SUNDIALS backend
- **THEN** telemetry includes selected formulation mode and solver family
- **AND** includes counters for function evaluations, Jacobian evaluations, nonlinear iterations, and error-test failures

### Requirement: Event-Consistent Direct Reinitialization
The v1 kernel SHALL preserve state consistency for direct SUNDIALS formulation across PWM/switch events and warm-start transitions.

#### Scenario: Direct mode with switching events
- **WHEN** a PWM boundary or switch threshold event triggers solver reinitialization
- **THEN** the reinitialized direct SUNDIALS state remains consistent with runtime circuit history
- **AND** subsequent transient steps continue without introducing non-physical discontinuities from formulation mismatch
