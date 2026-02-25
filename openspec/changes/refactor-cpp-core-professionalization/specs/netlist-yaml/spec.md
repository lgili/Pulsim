## ADDED Requirements
### Requirement: Schema Evolution Lifecycle
The YAML netlist capability SHALL define a versioned schema-evolution lifecycle with deterministic behavior for added, deprecated, and removed fields.

#### Scenario: Deprecated field inside migration window
- **WHEN** a YAML netlist uses a deprecated field still within the supported migration window
- **THEN** parsing succeeds with a structured deprecation diagnostic
- **AND** diagnostic output includes canonical replacement guidance

#### Scenario: Removed field after migration window
- **WHEN** a YAML netlist uses a field removed from the active schema version
- **THEN** parsing fails deterministically
- **AND** failure diagnostics include migration guidance to supported fields

### Requirement: Deterministic Structural Diagnostics
YAML parsing SHALL report strict diagnostics with stable reason codes and precise field-path context.

#### Scenario: Unknown field in strict mode
- **WHEN** strict validation is enabled and YAML contains an unknown field
- **THEN** parsing fails with a deterministic reason code
- **AND** diagnostics include the exact YAML path of the invalid field

#### Scenario: Invalid type at known field
- **WHEN** a known field has a value type that violates schema constraints
- **THEN** parsing fails with a deterministic type-mismatch diagnostic
- **AND** expected and received type classes are reported

### Requirement: Extension Namespace Validation
The YAML schema SHALL support explicit extension namespaces validated against registered runtime capabilities.

#### Scenario: Valid extension block
- **WHEN** YAML provides an extension block under a recognized namespace and valid contract
- **THEN** parsing maps the extension to runtime options deterministically
- **AND** extension metadata is available in structured parse results

#### Scenario: Unknown extension namespace
- **WHEN** YAML provides an extension namespace not recognized by the active registry
- **THEN** parsing fails (or warns in non-strict mode per policy) with deterministic diagnostics
- **AND** diagnostics identify the unsupported namespace

### Requirement: Compatibility Corpus Enforcement
The YAML parser SHALL maintain a compatibility corpus to prevent behavioral drift for supported legacy netlists.

#### Scenario: Supported legacy corpus run
- **WHEN** compatibility tests execute against the approved legacy YAML corpus
- **THEN** all supported netlists parse with equivalent semantic mappings
- **AND** regressions are flagged before merge
