## ADDED Requirements

### Requirement: Device-Variant Integration for Magnetic Components

The library SHALL register `SaturableTransformerDevice` and `HysteresisInductorDevice` in the `RuntimeCircuit::DeviceVariant` union, each wrapping the matching math object from `pulsim::v1::magnetic::` and exposing a `Circuit::add_*` method (C++ + Python).

#### Scenario: Saturable transformer is a first-class device
- **GIVEN** the consolidated codebase
- **WHEN** a user calls `Circuit::add_saturable_transformer(name, nodes, params)` or its Python equivalent
- **THEN** a `SaturableTransformerDevice` is registered in the circuit
- **AND** the transient inrush current reflects the B-H saturation curve of the underlying `magnetic::SaturableTransformer` math object

#### Scenario: Hysteresis inductor is a first-class device
- **GIVEN** the consolidated codebase
- **WHEN** a user calls `Circuit::add_hysteresis_inductor(name, nodes, params)`
- **THEN** a `HysteresisInductorDevice` is registered in the circuit
- **AND** the device's per-step state captures the hysteresis-loop traversal (energy dissipated per cycle matches the loop area within 5%)
