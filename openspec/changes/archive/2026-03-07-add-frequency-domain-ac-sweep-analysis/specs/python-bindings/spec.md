## ADDED Requirements
### Requirement: Python AC Sweep API Exposure
Python bindings SHALL expose frequency-domain analysis through canonical class-based and procedural APIs.

#### Scenario: Run AC sweep from class API
- **WHEN** Python code invokes AC sweep through canonical `Simulator` workflow
- **THEN** the run executes using the v1 kernel frequency-analysis path
- **AND** returned structures include typed sweep results and diagnostics

#### Scenario: Procedural compatibility entrypoint
- **WHEN** Python code uses procedural AC sweep entrypoints provided by bindings
- **THEN** behavior maps to the same kernel execution semantics as class-based API
- **AND** no console-text parsing is required to consume results

### Requirement: Structured Frequency-Domain Result Objects
Python bindings SHALL expose structured AC sweep result fields for frequency vector, complex response, magnitude/phase arrays, and derived metrics.

#### Scenario: Consume response data for plotting/reporting
- **WHEN** Python tooling reads AC sweep results
- **THEN** frequency and response arrays are available in structured fields
- **AND** metadata includes response quantity/unit context for frontend routing

#### Scenario: Undefined metrics are explicit
- **WHEN** crossover or stability margins are not mathematically defined for a response
- **THEN** result fields expose explicit undefined status/reason
- **AND** consumers can branch without heuristic checks

### Requirement: Structured Error Surface for AC Sweep
Python bindings SHALL propagate deterministic typed diagnostics for AC sweep parsing/preflight/runtime failures.

#### Scenario: Invalid frequency-analysis configuration
- **WHEN** Python submits invalid AC sweep options
- **THEN** bindings raise structured exceptions with deterministic reason codes and field context

#### Scenario: Kernel-side AC sweep failure
- **WHEN** kernel reports typed failure during AC sweep execution
- **THEN** Python receives equivalent failure reason and contextual details
- **AND** benchmark tooling can classify failures without regex parsing
