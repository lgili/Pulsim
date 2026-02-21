## MODIFIED Requirements

### Requirement: Single v1 Core Engine
The system SHALL use `pulsim/v1` as the sole simulation kernel for DC and transient analysis across all supported runtime entrypoints.

#### Scenario: Python runtime invokes simulation
- **WHEN** a simulation is executed through supported Python APIs
- **THEN** the execution path uses `pulsim/v1` classes and algorithms
- **AND** no alternate legacy kernel path is used

#### Scenario: Internal tooling invokes simulation
- **WHEN** benchmarks or internal helpers run simulations
- **THEN** they use the same `pulsim/v1` runtime path as Python-facing flows

## ADDED Requirements

### Requirement: Legacy Feature Migration Gate
Legacy functionality SHALL only be removed after equivalent v1 behavior exists and is validated.

#### Scenario: Legacy-only capability identified
- **WHEN** a capability exists only in legacy code
- **THEN** it is classified as `migrate`, `drop`, or `defer` in a migration matrix
- **AND** any `migrate` item is ported and tested in v1 before legacy deletion

### Requirement: Converter Component Support Matrix
The v1 kernel SHALL maintain a declared support matrix for converter-critical components and analyses.

#### Scenario: Supported converter workflow
- **WHEN** a converter benchmark uses components listed in the declared support matrix
- **THEN** the v1 runtime executes the case without falling back to legacy implementations
- **AND** emits structured solver telemetry for the run

### Requirement: Electro-Thermal Coupled Simulation
The v1 kernel SHALL support coupled electrical and thermal simulation for declared converter workflows.

#### Scenario: Coupled run enabled
- **WHEN** electro-thermal coupling is enabled for a converter case
- **THEN** electrical losses feed thermal states during simulation
- **AND** temperature-dependent model effects are applied according to configured coupling policy

### Requirement: Stress Convergence and Determinism Envelope
The v1 kernel SHALL pass tiered stress simulations with deterministic outcomes for fixed configurations.

#### Scenario: Repeated stress execution
- **WHEN** the same stress case is run multiple times on the same hardware class with fixed settings
- **THEN** status and key deterministic metrics (step count, solver path, error metrics) remain reproducible
