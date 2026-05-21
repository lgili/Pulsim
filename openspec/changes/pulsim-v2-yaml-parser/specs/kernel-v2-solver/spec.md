## ADDED Requirements

### Requirement: pulsim::v2::yaml loader

A `pulsim::v2::yaml` namespace SHALL provide a YAML circuit loader exposing `load_file(path) → LoadedCircuit` and `load_string(yaml_text) → LoadedCircuit`.

`LoadedCircuit` MUST contain:
- `builder`: a fully-populated `CircuitBuilder` reflecting the YAML's `circuit.devices:` list.
- `options`: a `SimulationOptions` populated from the YAML's optional `simulation:` block (defaults to a zero-initialised `SimulationOptions` if the block is missing).

The loader MUST support every device type exposed by `CircuitBuilder` as of Layer 2 V2: `voltage_source`, `resistor`, `capacitor`, `inductor`, `diode`, `nonlinear_diode`, `switch`, `mosfet`, `mosfet_with_body_diode`, `igbt`, `transformer`.

#### Scenario: Round-trip a resistor via YAML matches the direct builder

- **GIVEN** a YAML string with a single resistor (R=100Ω, from=a to=b)
- **WHEN** the user calls `load_string(yaml)`
- **THEN** `LoadedCircuit::builder.num_branches` SHALL equal 1
- **AND** the pool's resistor entry SHALL have `G = 1/100`.

#### Scenario: simulation: block populates SimulationOptions

- **GIVEN** a YAML with `simulation: { t_start: 0, t_end: 1e-3, dt: 1e-7, enable_newton_line_search: true }`
- **WHEN** the user calls `load_string(yaml)`
- **THEN** `LoadedCircuit::options.t_end` SHALL equal `1e-3`
- **AND** `LoadedCircuit::options.dt` SHALL equal `1e-7`
- **AND** `LoadedCircuit::options.enable_newton_line_search` SHALL be `true`.

### Requirement: YAML loader validation errors

The YAML loader MUST throw `std::runtime_error` on:
- Missing required fields per device type (e.g. a resistor without `R`).
- Unknown `type:` value in a device entry.
- Malformed YAML (re-wrapped from yaml-cpp's parse errors).

The error message MUST include the offending device's `name` if provided, otherwise the device's index in the YAML's `devices:` list.

#### Scenario: Missing R on a resistor throws with device name

- **GIVEN** a YAML with a resistor entry missing the `R:` field but with `name: R_load`
- **WHEN** the user calls `load_string(yaml)`
- **THEN** the call SHALL throw `std::runtime_error`
- **AND** the exception message SHALL contain the string `"R_load"`.

#### Scenario: Unknown device type throws with the type name

- **GIVEN** a YAML with a device entry containing `type: thyristor_v999`
- **WHEN** the user calls `load_string(yaml)`
- **THEN** the call SHALL throw `std::runtime_error`
- **AND** the exception message SHALL contain `"thyristor_v999"`.
