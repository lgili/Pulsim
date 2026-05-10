## ADDED Requirements

### Requirement: runtime_circuit.hpp Implementation Split
The `core/include/pulsim/v1/runtime_circuit.hpp` header SHALL be split such that method bodies move to a corresponding `.cpp`, with explicit template instantiation where applicable.

#### Scenario: Header trimmed
- **WHEN** the project is built
- **THEN** `runtime_circuit.hpp` contains declarations and public templates only
- **AND** method bodies live in `core/src/v1/runtime_circuit.cpp`
- **AND** the header is below 1000 lines

#### Scenario: Explicit instantiation
- **GIVEN** any client TU that includes `runtime_circuit.hpp`
- **WHEN** the TU is compiled
- **THEN** `extern template` declarations prevent re-instantiation of common specializations
- **AND** total client compile time is reduced ≥40% on a representative file

### Requirement: Bloated Header Audit
Headers exceeding 1500 lines (`high_performance.hpp`, `integration.hpp`) SHALL be audited and trimmed by moving inline implementations to `.cpp`, removing dead code, or splitting by concern.

#### Scenario: Header below threshold post-trim
- **WHEN** the audit completes
- **THEN** no kernel header in `core/include/pulsim/v1/` exceeds 1500 lines
- **AND** dead utilities removed from headers are documented in the change history

### Requirement: Build-Time Regression Alerts
CI SHALL track clean-build wallclock per platform and alert when growth exceeds 10% across two consecutive runs without an associated justification.

#### Scenario: Build-time regression
- **WHEN** a PR causes clean-build wallclock to grow >10% on any platform
- **THEN** the CI emits an alert linking the PR
- **AND** the PR description must include justification or a fix
