## ADDED Requirements

### Requirement: Missing-Component Parity Matrix

The benchmark/validation suite SHALL include parity coverage for all component types that were previously missing from backend support.

#### Scenario: Per-component smoke matrix
- **WHEN** parity matrix tests are executed
- **THEN** each newly covered component type has at least one executable smoke scenario
- **AND** failures identify the component type and family explicitly

### Requirement: Family-Level Behavioral Validation

The suite SHALL include behavioral validation scenarios for each family: power semiconductors, protection, magnetic/networks, control/analog, and instrumentation/routing.

#### Scenario: Family regression gate
- **WHEN** CI runs benchmark validation
- **THEN** each family-level suite passes configured behavior checks
- **AND** regressions fail the gate with stable diagnostics

### Requirement: Unsupported-Component Regression Guard

Validation SHALL enforce that supported mode does not regress to unsupported-component errors for the declared GUI parity set.

#### Scenario: Unsupported error regression
- **WHEN** a parity fixture containing declared component types is built and simulated
- **THEN** no `Unsupported component type` error is emitted for declared types
