## ADDED Requirements
### Requirement: Typed Waveform Post-Processing API in Python
Python bindings SHALL expose typed configuration and execution surfaces for waveform post-processing jobs.

#### Scenario: Configure and run post-processing jobs from Python
- **WHEN** Python code defines post-processing jobs (FFT/THD/time metrics/power-efficiency/loop metrics) using canonical typed fields
- **THEN** jobs execute through backend-owned logic
- **AND** result objects are returned in structured form without requiring ad-hoc script parsing.

#### Scenario: Invalid enum/value assignment
- **WHEN** Python code sets unsupported job type, window mode, or metric identifier
- **THEN** bindings fail deterministically with structured configuration diagnostics
- **AND** diagnostics include field context and accepted values.

### Requirement: Structured Python Result Surface
Python results SHALL expose post-processing outputs as structured, deterministic objects.

#### Scenario: Successful post-processing output
- **WHEN** post-processing executes successfully
- **THEN** Python receives structured scalar metrics, spectrum/harmonic payloads, and per-job metadata
- **AND** output ordering is deterministic for equivalent inputs.

#### Scenario: Metadata-driven channel interpretation
- **WHEN** Python consumers inspect post-processing outputs
- **THEN** units, domains, and source-signal bindings are available as structured metadata
- **AND** consumers do not need regex-based inference from label strings.

### Requirement: Structured Diagnostics and Undefined-Metric Reasons
Python bindings SHALL expose typed diagnostics for invalid jobs and undefined metrics.

#### Scenario: Invalid sampling/window prerequisites
- **WHEN** post-processing requests violate sampling/window constraints
- **THEN** Python receives deterministic typed diagnostics
- **AND** partial ambiguous output is not emitted for failed jobs.

#### Scenario: Metric undefined by physical/numerical conditions
- **WHEN** a metric cannot be defined (for example THD with zero fundamental component)
- **THEN** Python result includes deterministic undefined-reason code
- **AND** behavior is stable across repeated executions.

### Requirement: Backward-Compatible Runtime Workflows
Python transient/frequency workflows SHALL remain backward compatible when post-processing is not configured.

#### Scenario: Existing script without post-processing config
- **WHEN** an existing Python simulation script executes without post-processing jobs
- **THEN** behavior remains backward compatible
- **AND** no new mandatory fields are required.
