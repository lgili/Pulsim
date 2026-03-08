## ADDED Requirements
### Requirement: Canonical YAML Surface for Waveform Post-Processing
The YAML schema SHALL define a canonical `simulation.post_processing` block for waveform post-processing configuration.

#### Scenario: Valid post-processing block
- **GIVEN** a YAML netlist with valid post-processing jobs
- **WHEN** parsing executes
- **THEN** parser maps job definitions deterministically into runtime options
- **AND** no external script-only configuration is required.

#### Scenario: Block omitted
- **GIVEN** a YAML netlist without `simulation.post_processing`
- **WHEN** parsing executes
- **THEN** simulation workflows remain backward compatible
- **AND** parser does not require post-processing fields.

### Requirement: Strict Job Schema Validation
The YAML parser SHALL enforce strict deterministic validation for post-processing job fields.

#### Scenario: Invalid job type or metric identifier
- **WHEN** a post-processing job contains unsupported `kind`, `metric`, or equivalent constrained values
- **THEN** parsing fails with deterministic diagnostics listing accepted values
- **AND** diagnostics include exact YAML field path.

#### Scenario: Missing required job field
- **WHEN** required job fields (signal bindings, window spec, harmonic settings, etc.) are missing
- **THEN** parsing fails deterministically with field-level diagnostics
- **AND** no partial ambiguous mapping is applied.

### Requirement: Deterministic Window and Sampling Contract Validation
The YAML parser SHALL validate window and sampling contracts required by post-processing jobs.

#### Scenario: Invalid time/index/cycle window bounds
- **WHEN** window bounds are inconsistent or non-physical
- **THEN** parsing fails with deterministic diagnostics
- **AND** error context identifies the invalid window fields.

#### Scenario: Invalid spectral prerequisites in static config
- **WHEN** static configuration violates explicit spectral constraints (for example harmonic count/range constraints)
- **THEN** parser fails deterministically
- **AND** diagnostics include expected numeric ranges.

### Requirement: Typed Diagnostic Code Consistency
YAML diagnostics for post-processing configuration SHALL use stable machine-readable reason codes.

#### Scenario: Strict mode failure for malformed post-processing block
- **WHEN** strict parsing encounters malformed post-processing configuration
- **THEN** parser emits deterministic coded diagnostics
- **AND** benchmark/tooling consumers can classify failures without regex heuristics.
