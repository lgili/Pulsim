## ADDED Requirements
### Requirement: Convergence Class Stress Matrix
Benchmark and validation tooling SHALL provide a stress matrix indexed by convergence challenge class.

#### Scenario: Execute challenge-class matrix
- **WHEN** the stress matrix is executed
- **THEN** it includes at least classes `diode_heavy`, `switch_heavy`, `zero_cross`, `magnetic_nonlinear`, and `closed_loop_control`
- **AND** reports pass/fail and KPI outputs by class and runtime mode

### Requirement: Phase-Gated Progression for Robustness Program
CI SHALL enforce phase gates (A..F, ADV) for convergence-platform rollout.

#### Scenario: Phase gate regression
- **WHEN** required KPI thresholds fail for active phase
- **THEN** phase gate fails deterministically
- **AND** progression to next phase is blocked until restored

#### Scenario: Phase gate pass with artifact completeness
- **WHEN** a phase run meets KPI criteria
- **THEN** gate passes only if required telemetry and documentation artifacts are present
- **AND** artifacts are published in machine-readable form

### Requirement: Deterministic Reproducibility for Hard Cases
Hard-case benchmark runs SHALL provide deterministic reproducibility metadata and thresholds.

#### Scenario: Repeat hard-case run on same runner class
- **WHEN** the same hard-case matrix is re-executed under equivalent environment fingerprint
- **THEN** convergence-class KPIs remain within configured reproducibility tolerances
- **AND** deviations beyond tolerance fail reproducibility checks

### Requirement: Cross-Class Non-Regression Budget
Benchmark gates SHALL enforce explicit cross-class non-regression budgets to prevent overfitting convergence heuristics to a single circuit.

#### Scenario: Local gain causes cross-class regression
- **WHEN** a change improves one challenge class but exceeds regression budget in another stable class
- **THEN** the gate fails deterministically
- **AND** the report identifies improved and regressed classes side-by-side
