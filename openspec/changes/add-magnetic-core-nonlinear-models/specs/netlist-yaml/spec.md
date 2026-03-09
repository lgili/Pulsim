## ADDED Requirements
### Requirement: Canonical YAML Magnetic-Core Block
The YAML schema SHALL define a canonical `component.magnetic_core` block for nonlinear magnetic-core configuration on supported magnetic component types.

#### Scenario: Valid magnetic-core block on supported component
- **GIVEN** a supported inductor/transformer component with valid `magnetic_core` configuration
- **WHEN** YAML parsing runs in strict mode
- **THEN** parsing succeeds and maps canonical magnetic-core fields to runtime structures
- **AND** parser output preserves deterministic field mapping.

#### Scenario: Unsupported component uses magnetic_core
- **GIVEN** a non-magnetic component with `magnetic_core` block
- **WHEN** strict parsing runs
- **THEN** parsing fails with deterministic typed diagnostics
- **AND** diagnostics include the component type and field path.

### Requirement: Deterministic Magnetic-Core Parameter Validation
The YAML parser SHALL enforce deterministic range, dimensional, and compatibility checks for nonlinear magnetic-core parameters.

#### Scenario: Invalid table dimensions
- **GIVEN** a magnetic-core configuration with inconsistent table axis dimensions
- **WHEN** strict parsing runs
- **THEN** parsing fails deterministically
- **AND** diagnostics identify the exact invalid field path and expected dimensions.

#### Scenario: Invalid range or missing required field
- **GIVEN** a magnetic-core configuration with missing required parameters or out-of-range values
- **WHEN** strict parsing runs
- **THEN** parsing fails with stable reason codes
- **AND** diagnostics include actionable correction context.

### Requirement: Deterministic Magnetic Policy Defaults
When optional magnetic-core policy fields are omitted, the YAML parser SHALL apply deterministic defaults documented by the schema.

#### Scenario: Optional policy omitted
- **GIVEN** a valid magnetic-core configuration without optional policy overrides
- **WHEN** parsing runs
- **THEN** parser applies canonical default policy values deterministically
- **AND** behavior is reproducible across parser runs with identical input.
