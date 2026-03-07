## ADDED Requirements
### Requirement: Frequency-Analysis YAML Surface
The YAML schema SHALL define a canonical `simulation.frequency_analysis` block for frequency-domain workflows.

#### Scenario: Valid frequency-analysis block
- **GIVEN** a netlist with a valid `simulation.frequency_analysis` definition
- **WHEN** YAML parsing executes
- **THEN** parser maps mode, anchoring, sweep, perturbation, injection, and measurement fields into runtime options
- **AND** mapping is deterministic across runs

#### Scenario: Block omitted
- **GIVEN** a netlist without `simulation.frequency_analysis`
- **WHEN** parsing executes
- **THEN** standard non-frequency workflows remain backward compatible
- **AND** parser does not require AC sweep fields for transient/DC runs

### Requirement: Deterministic Sweep Schema Validation
The YAML parser SHALL enforce deterministic validation for frequency sweep parameters.

#### Scenario: Invalid frequency range
- **WHEN** `f_start_hz` is non-positive or `f_stop_hz < f_start_hz`
- **THEN** parsing fails with deterministic field-path diagnostics

#### Scenario: Invalid sweep density
- **WHEN** `points` is missing, non-integer, or below minimum supported value
- **THEN** parsing fails with deterministic diagnostics and expected-range details

### Requirement: Anchor and Mode Compatibility Validation
The YAML parser SHALL validate compatibility between requested analysis mode, anchoring strategy, and circuit declarations.

#### Scenario: Unsupported mode-anchor combination
- **WHEN** YAML requests an unsupported mode/anchor combination for the declared circuit context
- **THEN** parsing or preflight validation fails deterministically
- **AND** diagnostics include migration guidance toward supported combinations
