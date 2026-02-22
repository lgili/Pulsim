## ADDED Requirements

### Requirement: YAML Coverage for Missing GUI Component Types

The YAML parser SHALL accept and validate all currently missing GUI component types:

- `BJT_NPN`, `BJT_PNP`, `THYRISTOR`, `TRIAC`, `SWITCH`
- `OP_AMP`, `COMPARATOR`
- `FUSE`, `CIRCUIT_BREAKER`, `RELAY`
- `PI_CONTROLLER`, `PID_CONTROLLER`, `MATH_BLOCK`, `PWM_GENERATOR`, `INTEGRATOR`, `DIFFERENTIATOR`, `LIMITER`, `RATE_LIMITER`, `HYSTERESIS`, `LOOKUP_TABLE`, `TRANSFER_FUNCTION`, `DELAY_BLOCK`, `SAMPLE_HOLD`, `STATE_MACHINE`
- `SATURABLE_INDUCTOR`, `COUPLED_INDUCTOR`, `SNUBBER_RC`
- `VOLTAGE_PROBE`, `CURRENT_PROBE`, `POWER_PROBE`, `ELECTRICAL_SCOPE`, `THERMAL_SCOPE`, `SIGNAL_MUX`, `SIGNAL_DEMUX`

#### Scenario: Parse full GUI-parity netlist
- **WHEN** a YAML netlist includes at least one instance of each listed component type with valid pins and parameters
- **THEN** parsing succeeds without unsupported-type errors
- **AND** each component is mapped to its backend runtime representation

### Requirement: Strict Pin and Parameter Validation for New Types

Strict mode SHALL validate pin count, required parameters, and parameter ranges for newly supported component types.

#### Scenario: Invalid relay pin configuration
- **WHEN** a relay component is defined with missing `COM/NO/NC` terminals
- **THEN** parsing fails with a diagnostic naming the missing pins

#### Scenario: Invalid saturation parameters
- **WHEN** a saturable inductor is defined with non-physical parameter combination
- **THEN** parsing fails with clear parameter-range diagnostics

### Requirement: Canonical Name and Alias Mapping

The parser SHALL support canonical backend names and GUI/YAML aliases for new component families while producing canonical internal representation.

#### Scenario: Alias normalization
- **WHEN** a netlist uses accepted aliases for a new component type
- **THEN** the parser normalizes the alias to canonical type identifier
- **AND** downstream runtime and diagnostics reference canonical naming
