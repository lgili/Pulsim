## ADDED Requirements

### Requirement: Advanced Linear Solver Stack
The v1 kernel SHALL provide both direct and iterative linear solvers with runtime selection and robust fallback.

#### Scenario: Large sparse circuit prefers iterative solver
- **WHEN** a circuit exceeds the configured size/nnz thresholds
- **THEN** the solver selects an iterative method (GMRES/BiCGSTAB/CG)
- **AND** applies a preconditioner (ILU0/Jacobi) if configured

#### Scenario: Iterative solve fails
- **WHEN** an iterative solve fails to converge within limits
- **THEN** the solver SHALL fall back to a direct method (KLU/Eigen SparseLU)
- **AND** record the fallback in solver telemetry

### Requirement: Nonlinear Solver Acceleration
The v1 kernel SHALL support nonlinear acceleration strategies beyond basic Newton iteration.

#### Scenario: Difficult nonlinear circuit
- **WHEN** Newton stalls or oscillates
- **THEN** the solver SHALL apply an acceleration method (Anderson or Broyden)
- **AND** may switch to Newton-Krylov with the same tolerances

#### Scenario: Aggressive steps increase residual
- **WHEN** a Newton step increases residual error
- **THEN** the solver SHALL apply line search or trust-region damping
- **AND** retry the step within configured limits

### Requirement: Solver Auto-Selection and Fallback Order
The v1 kernel SHALL allow a configurable solver selection order with deterministic fallback.

#### Scenario: User-defined solver order
- **WHEN** the configuration specifies a solver order
- **THEN** the kernel SHALL try solvers in that order
- **AND** stop at the first successful strategy

#### Scenario: Deterministic fallback
- **WHEN** multiple solvers are enabled
- **THEN** the fallback order SHALL be deterministic for reproducible results

### Requirement: Stiffness-Aware Transient Integration
The v1 kernel SHALL detect stiffness indicators and adapt integration order and timestep accordingly.

#### Scenario: Stiff switching transient
- **WHEN** stiffness is detected (e.g., repeated step rejection or large Jacobian condition changes)
- **THEN** the solver SHALL reduce timestep and/or lower BDF order
- **AND** continue with stability-focused settings until recovery

### Requirement: Solver Telemetry
The v1 kernel SHALL expose solver telemetry for debugging and regression tracking.

#### Scenario: Telemetry capture
- **WHEN** a simulation completes
- **THEN** the result SHALL include counts of nonlinear iterations, linear iterations, and fallback events
- **AND** the selected solver policies SHALL be reported in a structured form
