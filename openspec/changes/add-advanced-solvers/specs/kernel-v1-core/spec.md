## ADDED Requirements

### Requirement: Primary and Fallback Solver Order
The v1 kernel SHALL support separate primary and fallback solver orders for deterministic selection.

#### Scenario: Primary order succeeds
- **WHEN** the primary solver order succeeds
- **THEN** fallback order SHALL NOT be used

#### Scenario: Primary order fails
- **WHEN** the primary solver order fails
- **THEN** the fallback order SHALL be attempted in deterministic order

### Requirement: SPD‑Safe Conjugate Gradient
CG SHALL only be used when the linear system is symmetric positive definite (SPD).

#### Scenario: Non‑SPD matrix
- **WHEN** the matrix is not SPD
- **THEN** CG SHALL be rejected and a fallback solver SHALL be selected

### Requirement: Jacobian‑Free Newton–Krylov
The v1 kernel SHALL support JFNK with Jacobian‑vector products and iterative linear solvers.

#### Scenario: JFNK enabled
- **WHEN** JFNK is enabled
- **THEN** the solver SHALL compute J·v without assembling the full Jacobian
- **AND** use an iterative Krylov method

### Requirement: Stiff‑Stable Integrators
The v1 kernel SHALL provide TR‑BDF2 and Rosenbrock‑W/SDIRK integrators for stiff systems.

#### Scenario: TR‑BDF2 selection
- **WHEN** TR‑BDF2 is selected
- **THEN** the integrator SHALL remain stable on stiff switching transients

#### Scenario: Rosenbrock selection
- **WHEN** Rosenbrock‑W/SDIRK is selected
- **THEN** the integrator SHALL maintain stability for stiff DAEs

### Requirement: Periodic Steady‑State Solvers
The v1 kernel SHALL provide periodic steady‑state solvers for switching converters.

#### Scenario: Shooting method
- **WHEN** the shooting method is enabled
- **THEN** the solver SHALL converge to a periodic steady‑state waveform

#### Scenario: Harmonic balance
- **WHEN** harmonic balance is enabled
- **THEN** the solver SHALL compute steady‑state frequency‑domain solution
