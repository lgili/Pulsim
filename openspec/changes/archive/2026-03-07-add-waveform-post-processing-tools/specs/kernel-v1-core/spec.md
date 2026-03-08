## ADDED Requirements
### Requirement: Canonical Waveform Post-Processing Pipeline in v1 Kernel
The v1 kernel SHALL provide a backend-owned waveform post-processing pipeline that operates on canonical simulation/frequency result surfaces.

#### Scenario: Deterministic post-processing execution
- **GIVEN** valid post-processing job definitions and resolved source signals
- **WHEN** post-processing executes
- **THEN** the kernel computes requested metrics deterministically
- **AND** output ordering and key naming are stable for equivalent inputs.

#### Scenario: Unsupported source contract
- **WHEN** a post-processing job references unsupported or missing source data
- **THEN** execution fails with typed deterministic diagnostics
- **AND** no partial ambiguous result object is emitted for that failed job.

### Requirement: Deterministic Windowing and Sampling Semantics
The v1 kernel SHALL enforce explicit deterministic semantics for analysis windows and sampling prerequisites.

#### Scenario: Valid window specification
- **WHEN** a job specifies a valid window (time/index/cycle mode) with sufficient samples
- **THEN** the kernel resolves deterministic sample bounds
- **AND** all derived metrics use those exact bounds.

#### Scenario: Invalid or insufficient window
- **WHEN** a job window is invalid, empty, or undersampled for requested analysis
- **THEN** the job fails with typed diagnostics (`invalid_window` or `insufficient_samples` equivalent)
- **AND** failure reason includes deterministic context fields.

### Requirement: Spectral and Harmonic Metric Engine
The v1 kernel SHALL support FFT/harmonic analysis and THD computation with explicit configuration and deterministic behavior.

#### Scenario: FFT/THD on known harmonic waveform
- **GIVEN** a waveform with known harmonic composition
- **WHEN** FFT and THD jobs execute
- **THEN** fundamental, harmonic bins, and THD are computed within configured tolerances
- **AND** spectral outputs include deterministic frequency-axis metadata.

#### Scenario: Undefined THD reason code
- **WHEN** THD cannot be defined due to zero/undefined fundamental component
- **THEN** the kernel emits deterministic undefined-metric reason codes
- **AND** does not silently report arbitrary fallback values.

### Requirement: Time-Domain and Power Metric Engine
The v1 kernel SHALL provide deterministic time-domain and power metrics for post-processing workflows.

#### Scenario: Time-domain metric set
- **WHEN** a job requests metrics such as RMS, mean, min/max, peak-to-peak, crest factor, or ripple
- **THEN** kernel outputs canonical metric keys and values with units
- **AND** metric definitions remain stable across releases unless explicitly versioned.

#### Scenario: Efficiency and power-factor metric set
- **WHEN** input/output power signal bindings are valid
- **THEN** kernel computes average power, derived efficiency, and power-factor metrics deterministically
- **AND** invalid bindings fail with typed diagnostics.

### Requirement: Post-Processing Runtime Discipline
Post-processing execution SHALL maintain allocation-aware behavior and expose telemetry for non-regression gating.

#### Scenario: Repeated post-processing with equivalent jobs
- **WHEN** equivalent post-processing jobs are executed repeatedly on the same machine class
- **THEN** runtime telemetry indicates stable execution cost and deterministic outputs
- **AND** regressions beyond configured KPI thresholds fail CI gates.
