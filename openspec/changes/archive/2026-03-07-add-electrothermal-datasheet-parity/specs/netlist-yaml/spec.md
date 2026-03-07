## ADDED Requirements
### Requirement: Datasheet Loss-Model YAML Surface
The YAML schema SHALL provide a backend-complete semiconductor loss-model block that supports both scalar and datasheet-grade characterization.

#### Scenario: Datasheet-grade loss block accepted
- **WHEN** YAML defines a component loss-model block with supported axes/tables for conduction and switching terms
- **THEN** parser maps the block into runtime characterization structures
- **AND** strict validation confirms dimensional consistency and required fields

#### Scenario: Scalar loss block remains valid
- **WHEN** YAML uses legacy scalar `eon/eoff/err` style fields
- **THEN** parsing succeeds with backward-compatible mapping
- **AND** runtime behavior remains compatible with existing scalar workflows

### Requirement: Thermal Network YAML Surface
The YAML schema SHALL support thermal-network family selection and staged thermal parameters for thermal-enabled components.

#### Scenario: Foster or Cauer network declaration
- **WHEN** YAML declares thermal network family `foster` or `cauer` with stage parameters
- **THEN** parser validates stage schema and maps to runtime thermal-network configuration
- **AND** invalid stage dimensions or ranges fail with deterministic diagnostics

#### Scenario: Single-RC declaration compatibility
- **WHEN** YAML provides classic single `rth/cth` thermal fields
- **THEN** parser maps them as `single_rc` behavior
- **AND** migration remains backward compatible

### Requirement: Deterministic Table and Axis Validation
The YAML parser SHALL validate datasheet table schema deterministically, including axis monotonicity and dimension matching.

#### Scenario: Invalid table dimensions
- **WHEN** table dimensions do not match declared axes cardinality
- **THEN** parsing fails with deterministic reason code
- **AND** diagnostics include exact YAML field path for the mismatch

#### Scenario: Non-monotonic axis values
- **WHEN** an axis expected to be monotonic is not monotonic
- **THEN** strict validation fails with deterministic diagnostics

### Requirement: GUI-Agnostic Backend-Complete Contract
YAML electrothermal schema SHALL remain GUI-agnostic so simulation correctness does not depend on frontend heuristics.

#### Scenario: Headless execution with full electrothermal definition
- **WHEN** a netlist provides full loss and thermal definitions and runs via backend/Python only
- **THEN** simulation produces complete electrothermal outputs without GUI participation
- **AND** all required physical calculations are backend-owned
