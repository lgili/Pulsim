## ADDED Requirements
### Requirement: Python Configuration for SUNDIALS Backend
Python bindings SHALL expose SUNDIALS backend configuration fields equivalent to runtime simulation options.

#### Scenario: Configure SUNDIALS backend from Python
- **WHEN** Python code sets transient backend mode to `SUNDIALS` or `Auto` and selects solver family/tolerances
- **THEN** `Simulator` and procedural APIs SHALL run with the same backend configuration semantics as kernel options
- **AND** invalid backend-family combinations SHALL produce clear Python exceptions

### Requirement: Python Telemetry for Backend Selection
Python bindings SHALL expose backend telemetry including whether SUNDIALS was used and why escalation happened.

#### Scenario: Inspect backend telemetry after run
- **WHEN** Python code accesses transient simulation result telemetry
- **THEN** it SHALL read backend selection, solver family, escalation counters, and backend failure diagnostics when applicable
- **AND** these fields SHALL be structured (not string-parsed)

### Requirement: Backward-Compatible Defaults Without SUNDIALS
Python APIs SHALL remain backward-compatible on builds without SUNDIALS.

#### Scenario: Existing script on non-SUNDIALS build
- **WHEN** an existing Python script uses default transient APIs without backend overrides
- **THEN** behavior SHALL remain compatible with the native backend defaults
- **AND** capability inspection SHALL report SUNDIALS unavailable without breaking imports
