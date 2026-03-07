## ADDED Requirements
### Requirement: Averaged-Converter Typed Python Configuration
Python bindings SHALL expose typed configuration surfaces for averaged converter modeling mode.

#### Scenario: Configure averaged mode from Python
- **WHEN** Python code configures averaged-converter options using canonical typed fields/enums
- **THEN** configuration maps deterministically to kernel runtime options
- **AND** invalid enum/value assignments are rejected with deterministic diagnostics.

#### Scenario: Backward compatibility for switching mode code
- **GIVEN** Python code that uses existing transient APIs without averaged settings
- **WHEN** the code executes
- **THEN** behavior remains backward compatible
- **AND** no averaged-specific fields are required.

### Requirement: Structured Averaged-Mode Diagnostics in Python
Python bindings SHALL expose structured diagnostics for averaged-mode validation and runtime failures.

#### Scenario: Mapping/envelope failure surfaces
- **WHEN** averaged-mode execution fails due to mapping errors or envelope violations
- **THEN** Python receives typed diagnostic fields (reason code, context, message)
- **AND** callers can classify failure without parsing free-form logs.

### Requirement: Averaged-Mode Telemetry and Channel Access in Python
Python results SHALL expose averaged-state channels and mapping/envelope telemetry in deterministic structures.

#### Scenario: Successful averaged run publishes telemetry
- **WHEN** averaged-mode simulation succeeds
- **THEN** Python result objects expose averaged channels and canonical metadata
- **AND** mapping/envelope telemetry fields are available for assertions and reports.
