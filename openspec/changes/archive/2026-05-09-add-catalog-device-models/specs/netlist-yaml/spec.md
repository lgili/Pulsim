## ADDED Requirements

### Requirement: Catalog Device Component Types
The YAML netlist schema SHALL accept new component types `mosfet_catalog`, `igbt_catalog`, `diode_catalog`, each requiring a `model` reference or inline `parameters` block.

#### Scenario: Catalog device by model name
- **GIVEN** a netlist component:
```yaml
- type: mosfet_catalog
  name: Q1
  nodes: [drain, gate, source]
  model: wolfspeed/C3M0065090J
```
- **WHEN** the parser loads the netlist
- **THEN** the device is instantiated from the named catalog YAML
- **AND** the device participates in the simulation

#### Scenario: Catalog device with inline params
- **GIVEN** a netlist component with `parameters: { rds_on_25c: 19e-3, vth: 3.5, ... }`
- **WHEN** the parser loads
- **THEN** the device is instantiated with the inline parameter set
- **AND** missing required parameters produce a clear diagnostic

#### Scenario: Catalog device with override
- **GIVEN** a netlist component with both `model: vendor/part` and `parameters: { rds_on_25c: 25e-3 }`
- **WHEN** the parser loads
- **THEN** the inline parameter overrides the catalog default
- **AND** all other catalog parameters are preserved

### Requirement: Strict Validation of Catalog References
Strict validation SHALL fail with a deterministic diagnostic when a catalog `model:` reference cannot be resolved.

#### Scenario: Unknown model
- **GIVEN** a netlist with `model: vendor/nonexistent_part`
- **WHEN** strict mode parsing runs
- **THEN** parsing fails with reason `catalog_model_not_found`
- **AND** the diagnostic suggests `pulsim catalog list <vendor>` to inspect available parts

### Requirement: Catalog Parameter Schema in YAML
The YAML schema SHALL document required and optional parameters for each catalog device type, validated by strict mode.

#### Scenario: Missing required parameter
- **GIVEN** an inline `mosfet_catalog` lacking `rds_on_25c`
- **WHEN** strict parsing runs
- **THEN** parsing fails with `missing_required_parameter` and the missing field name
- **AND** the diagnostic includes the schema YAML path
