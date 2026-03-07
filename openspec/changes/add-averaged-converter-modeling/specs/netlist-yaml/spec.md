## ADDED Requirements
### Requirement: Averaged-Converter YAML Surface
The YAML schema SHALL define a canonical `simulation.averaged_converter` block for averaged converter modeling workflows.

#### Scenario: Valid averaged-converter block
- **GIVEN** a netlist with a valid `simulation.averaged_converter` block
- **WHEN** parsing executes
- **THEN** parser maps averaged-mode fields deterministically into runtime options
- **AND** no legacy ad-hoc averaged flags are required.

#### Scenario: Block omitted
- **GIVEN** a netlist without `simulation.averaged_converter`
- **WHEN** parsing executes
- **THEN** standard switching transient workflows remain backward compatible
- **AND** parser does not require averaged fields.

### Requirement: Deterministic Mapping Schema Validation
The YAML parser SHALL enforce strict deterministic validation for topology-specific mapped fields required by averaged mode.

#### Scenario: Missing required mapped field in strict mode
- **GIVEN** strict parser mode is enabled
- **WHEN** required averaged mapping fields are missing or malformed
- **THEN** parsing fails with deterministic field-path diagnostics
- **AND** diagnostics identify the missing key and expected type/range.

#### Scenario: Invalid enum/value in averaged configuration
- **WHEN** averaged block includes unsupported `topology`, `envelope_policy`, or equivalent constrained fields
- **THEN** parsing fails with deterministic diagnostics that include accepted values.

### Requirement: Envelope Policy YAML Contract
The YAML schema SHALL support explicit averaged-model envelope policy settings with deterministic parser behavior.

#### Scenario: Strict envelope policy accepted
- **WHEN** `simulation.averaged_converter.envelope_policy: strict` is configured
- **THEN** parser maps policy deterministically to runtime options.

#### Scenario: Warn envelope policy accepted
- **WHEN** `simulation.averaged_converter.envelope_policy: warn` is configured
- **THEN** parser maps policy deterministically to runtime options
- **AND** this policy is available in runtime telemetry for report workflows.
