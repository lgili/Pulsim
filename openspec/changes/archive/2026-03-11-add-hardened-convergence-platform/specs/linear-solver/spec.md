## ADDED Requirements
### Requirement: Health-Driven Linear Solver Policy
Linear solver selection SHALL consider numeric health signals in addition to size/nnz heuristics.

#### Scenario: Conditioning degradation in active solver
- **WHEN** conditioning and residual indicators cross degradation thresholds
- **THEN** solver policy escalates deterministically to a safer candidate solver/preconditioner
- **AND** emits structured policy-transition telemetry

#### Scenario: Healthy regime retains fast solver
- **WHEN** health indicators remain within configured safe bounds
- **THEN** active solver and cache reuse remain on the efficient path
- **AND** no unnecessary solver churn occurs

### Requirement: Structured Linear Failure Taxonomy for Policy Engine
Linear failures SHALL expose typed reason classes consumable by the convergence policy engine.

#### Scenario: Iterative breakdown classification
- **WHEN** iterative solve fails due to breakdown or stagnation
- **THEN** failure reason class is exported in structured form
- **AND** convergence policy can branch recovery by class without parsing error strings
