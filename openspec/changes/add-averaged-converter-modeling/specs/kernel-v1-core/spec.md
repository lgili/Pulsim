## ADDED Requirements
### Requirement: Averaged Converter Runtime Mode in v1 Kernel
The v1 kernel SHALL provide a first-class averaged-converter runtime mode for transient control-design workflows.

#### Scenario: Supported topology executes in averaged mode
- **GIVEN** a valid netlist configured for averaged-converter mode with a supported topology
- **WHEN** transient simulation executes
- **THEN** the kernel advances averaged state equations deterministically
- **AND** the run completes through canonical transient result surfaces.

#### Scenario: Unsupported topology is rejected
- **GIVEN** averaged mode is requested for a topology outside the supported MVP set
- **WHEN** validation or runtime preflight executes
- **THEN** the kernel returns a typed deterministic diagnostic (`AveragedModelUnsupportedTopology`)
- **AND** no partial ambiguous averaged-state output is emitted.

### Requirement: Deterministic Switching-to-Averaged Mapping
The v1 kernel SHALL map switching-reference configuration to averaged-model parameters through an explicit deterministic mapping contract.

#### Scenario: Valid mapping payload
- **GIVEN** all required mapped fields for the selected topology are present and valid
- **WHEN** averaged-mode preflight executes
- **THEN** the kernel builds averaged model parameters deterministically
- **AND** mapping summary telemetry is produced for audit/debug workflows.

#### Scenario: Missing mapped element
- **GIVEN** at least one required mapped element is missing or malformed
- **WHEN** averaged-mode preflight executes
- **THEN** the run fails with deterministic field-level diagnostics (`AveragedModelMappingFailure`)
- **AND** diagnostics include the missing mapping key.

### Requirement: Operating-Envelope Policy Enforcement
The v1 kernel SHALL enforce averaged-model operating-envelope policy with explicit deterministic behavior.

#### Scenario: Strict envelope policy rejects out-of-envelope operation
- **GIVEN** envelope policy is `strict`
- **AND** runtime detects averaged-model assumptions are violated (for example CCM envelope violation)
- **WHEN** execution reaches out-of-envelope conditions
- **THEN** execution fails with typed diagnostics (`AveragedModelOutOfEnvelope`)
- **AND** no silent fallback to switched mode occurs.

#### Scenario: Warn envelope policy continues with telemetry
- **GIVEN** envelope policy is `warn`
- **AND** runtime detects out-of-envelope behavior
- **WHEN** simulation continues
- **THEN** the run remains deterministic and successful when numerically feasible
- **AND** structured warnings/telemetry explicitly mark the envelope violation.

### Requirement: Averaged-State Result and Metadata Contract
The v1 kernel SHALL expose averaged-state channels and metadata in canonical result surfaces.

#### Scenario: Averaged-state channels published
- **WHEN** averaged-mode simulation succeeds
- **THEN** result channels include deterministic averaged quantities (for example inductor current average, capacitor voltage average, duty-effective signals where defined)
- **AND** each channel includes unit/domain/source metadata for frontend routing.

#### Scenario: Deterministic channel ordering across repeated runs
- **GIVEN** identical circuit, options, and machine class
- **WHEN** averaged mode is executed repeatedly
- **THEN** averaged channel names and ordering are deterministic
- **AND** sample-wise numeric drift remains within determinism tolerance gates.

### Requirement: Averaged-Mode Runtime Discipline
Averaged-mode execution SHALL preserve allocation-bounded hot-loop behavior and expose telemetry for non-regression gating.

#### Scenario: Repeated averaged execution reuses runtime structures
- **WHEN** identical averaged-mode simulations run repeatedly
- **THEN** runtime reuses reusable structures deterministically
- **AND** telemetry exposes metrics needed for CI runtime/allocation regression gates.
