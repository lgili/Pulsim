## ADDED Requirements
### Requirement: Convergence Playbook Documentation
The project SHALL publish a convergence playbook describing strategy selection, tuning guidance, and failure triage for challenging circuits.

#### Scenario: User investigates transient convergence failure
- **WHEN** a user receives structured convergence diagnostics
- **THEN** the playbook maps diagnostics to recommended mitigation paths
- **AND** includes strict vs balanced vs robust profile guidance

### Requirement: Reference Example Corpus for Hard Scenarios
The project SHALL maintain runnable reference examples for each hard convergence class.

#### Scenario: Run diode-heavy zero-cross example
- **WHEN** the documented diode-heavy zero-cross example is executed
- **THEN** it reproduces expected behavior and emits convergence telemetry
- **AND** serves as regression reference for future solver changes

#### Scenario: Run closed-loop control example with C-Block
- **WHEN** the documented closed-loop C-Block example is executed
- **THEN** it demonstrates stable policy behavior under control/electrical interaction stress
- **AND** includes expected KPI ranges for validation

### Requirement: Migration and Tuning Guide
The project SHALL provide a migration/tuning guide for existing users adopting new convergence profiles and telemetry.

#### Scenario: Existing workflow migrates to policy profiles
- **WHEN** a user migrates from legacy defaults to policy-driven profiles
- **THEN** the guide provides deterministic mapping steps and backward-compatible defaults
- **AND** highlights any behavior changes and verification checklist
