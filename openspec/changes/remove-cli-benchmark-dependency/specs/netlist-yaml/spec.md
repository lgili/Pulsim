# netlist-yaml

## ADDED Requirements

### Requirement: Unified YAML Parsing Across Python and Kernel Paths
The system SHALL use the same v1 YAML parsing semantics for Python runtime execution as for kernel-native execution paths.

#### Scenario: Parse benchmark YAML from Python runtime
- **WHEN** benchmark tooling loads a YAML netlist through Python bindings
- **THEN** parsing behavior and option mapping match the v1 YAML parser semantics
- **AND** supported solver/integrator/periodic fields map identically

### Requirement: Strict Diagnostic Parity in Python
Strict validation diagnostics from the YAML parser SHALL be exposed consistently in Python runtime paths.

#### Scenario: Unknown solver field in strict mode
- **WHEN** a YAML netlist contains an unknown field and strict mode is enabled via Python-exposed parser options
- **THEN** parsing fails with a clear diagnostic message
- **AND** benchmark tooling surfaces that diagnostic as an explicit failure reason
