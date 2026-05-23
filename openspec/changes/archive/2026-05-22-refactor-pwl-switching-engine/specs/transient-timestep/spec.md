## ADDED Requirements

### Requirement: PWL Tustin Stepping
The transient kernel SHALL provide a Tustin (trapezoidal) PWL step that advances `(M + dt/2 N) x_{n+1} = (M − dt/2 N) x_n + dt/2 (b_{n+1} + b_n)` in a single linear solve.

#### Scenario: PWL Tustin step
- **GIVEN** a stable topology with state-space `(M, N, b)`
- **WHEN** a PWL Tustin step of size `dt` is taken
- **THEN** exactly one linear solve is performed
- **AND** Newton iteration is not invoked

#### Scenario: PWL backward-Euler fallback
- **GIVEN** stiffness detection flags numerical damping is needed
- **WHEN** the integrator is configured as `pwl_integrator: backward_euler`
- **THEN** the step uses backward-Euler discretization on the same `(M, N, b)`
- **AND** stability is preserved at the cost of first-order accuracy

### Requirement: PWL Event Bisection Tolerance
The kernel SHALL bisect event times to a configurable tolerance (default `1e-12`) when an event is detected within a candidate step.

#### Scenario: Default tolerance
- **WHEN** a switch event occurs within a candidate step
- **THEN** the bisection narrows the event time to ≤`1e-12` of true crossing
- **AND** the accepted step boundary equals the bisected event time within tolerance

#### Scenario: Configured tolerance
- **GIVEN** `simulation.event_tolerance = 1e-9`
- **WHEN** an event is detected
- **THEN** bisection narrows to ≤`1e-9`
- **AND** the configured tolerance is reflected in `BackendTelemetry`
