## ADDED Requirements

### Requirement: Converter Coverage Schema Completeness
The YAML schema SHALL represent all declared converter component and solver fields required by the support matrix.

#### Scenario: Declared converter case in YAML
- **WHEN** a netlist uses only fields listed in the declared converter support matrix
- **THEN** parsing succeeds in strict mode
- **AND** options map to v1 runtime without requiring legacy parsers

### Requirement: Electro-Thermal Configuration in YAML
The YAML format SHALL support explicit electro-thermal simulation configuration for declared workflows.

#### Scenario: Coupled electro-thermal netlist
- **WHEN** a netlist enables thermal coupling settings and thermal model parameters
- **THEN** parser maps these settings to v1 simulation options and thermal runtime configuration
- **AND** invalid thermal fields produce strict diagnostics

### Requirement: Explicit Legacy JSON Rejection Diagnostics
The YAML parser SHALL emit actionable diagnostics when users provide legacy JSON netlists in supported workflows.

#### Scenario: JSON netlist provided to YAML loader
- **WHEN** a user passes a JSON-format netlist to the v1 YAML parser path
- **THEN** parsing fails with a message that JSON netlists are unsupported
- **AND** the diagnostic points to the YAML migration guidance
